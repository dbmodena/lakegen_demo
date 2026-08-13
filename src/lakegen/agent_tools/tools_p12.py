import json
from pathlib import Path
from typing import Callable
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping

from lakegen.core.types import SolrMetadata
from lakegen.agent_tools.tools_p2 import (
    MAX_INSPECTIONS_PER_FILE,
    _find_schema_matches,
    _inspect_columns,
)
from src.client_solr import LocalSolrClient
from lakegen.phases.utils import match_local_csv, solr_metadata_from_doc, format_candidate_context
from lakegen.core.resources import get_table_retrieval_service
from lakegen.retrieval import (
    EmbeddingGenerationError,
    RetrievalConfig,
    RetrievalRun,
    RetrievalMode,
)


class ConfirmUnifiedSelectionSchema(BaseModel):
    reasoning: str = Field(description="MANDATORY. Write a brief explanation IN ENGLISH explaining why these specific tables were selected and how they answer the question.")
    tables: list[str] = Field(description="A list of ALL the exact file names needed (e.g., ['sales.parquet', 'dates.parquet']). Do not omit any table you need!")


class P12State:
    """State tracker for the Phase 1 & 2 unified agent."""
    def __init__(self):
        self.all_candidates: list[str] = []
        self.solr_meta: SolrMetadata = {}
        self.used_keywords: list[str] = []
        self.keyword_history: list[list[str]] = []
        self.best_ranks: dict[str, int] = {}
        self.search_cache: dict[tuple[str, ...], str] = {}
        self.search_attempts: list[dict[str, object]] = []
        self.semantic_failure: str | None = None
        self.inspection_cache: dict[str, str] = {}
        self.inspection_counts: dict[str, int] = {}


class Phase12ToolsManager:
    """Manager for Phase 1 & 2 unified tools to avoid closures and improve testability."""
    
    def __init__(
        self,
        state: P12State,
        solr_client: LocalSolrClient,
        all_files: list[str],
        csv_dir: Path,
        question: str = "",
        retrieval_config: RetrievalConfig | None = None,
        retrieval_observer: Callable[[RetrievalRun], None] | None = None,
    ):
        self.state = state
        self.solr_client = solr_client
        self.all_files = all_files
        self.csv_dir = Path(csv_dir)
        self.question = question
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.retrieval_observer = retrieval_observer

    def _search_cache_key(self, keywords: list[str]) -> tuple[str, ...]:
        # The agent-facing contract allows one retrieval request regardless of
        # the configured strategy.  Mode-specific use of the concepts remains
        # internal to the retriever.
        return ("configured-retrieval",)

    def _search_tool_description(self) -> str:
        return (
            "Search for relevant tables using the retrieval strategy configured by "
            "the experiment. Provide 1-2 concise dataset concepts in the portal's "
            "native language. The tool applies the original question and the "
            "configured retrieval parameters automatically. Call the tool once and "
            "evaluate only the returned candidates."
        )

    def search_solr(self, concepts_str: str = "") -> str:
        """Search for relevant tables using one or two dataset concepts."""
        try:
            supplied_concepts = [
                item.strip() for item in concepts_str.split(" ") if item.strip()
            ]
            keywords = list(supplied_concepts)
            if self.retrieval_config.mode == RetrievalMode.SEMANTIC:
                keywords = []
            elif not keywords:
                return "No concepts provided. Search with one or two dataset concepts."
            if (
                self.retrieval_config.mode != RetrievalMode.KEYWORD
                and self.state.semantic_failure is not None
            ):
                return (
                    "Configured retrieval skipped: representation generation already "
                    "failed for this request. " + self.state.semantic_failure
                )
            key = self._search_cache_key(keywords)
            if key in self.state.search_cache:
                return (
                    "Search skipped: the configured retrieval was already run.\n"
                    + self.state.search_cache[key]
                )
            self.state.used_keywords = supplied_concepts
            self.state.keyword_history.append(keywords)
            attempt = len(self.state.keyword_history)

            if self.retrieval_observer is None:
                retriever = get_table_retrieval_service(
                    self.solr_client,
                    self.retrieval_config,
                )
            else:
                retriever = get_table_retrieval_service(
                    self.solr_client,
                    self.retrieval_config,
                    observer=self.retrieval_observer,
                )
            # Solr candidates must first be mapped and de-duplicated against
            # local files.  Request a wider ranked list here and apply the
            # workflow's final top_k only after that mapping below.
            fetch_k = max(15, self.retrieval_config.top_k)
            hits = retriever.retrieve(
                question=self.question,
                keywords=keywords,
                top_k=fetch_k,
                lexical_fetch_k=fetch_k,
                q_op="AND",
            )
            search_mode = (
                "AND" if self.retrieval_config.mode == RetrievalMode.KEYWORD
                else self.retrieval_config.mode.value
            )

            if (
                not hits
                and len(keywords) > 1
                and self.retrieval_config.mode == RetrievalMode.KEYWORD
            ):
                hits = retriever.retrieve(
                    question=self.question,
                    keywords=keywords,
                    top_k=fetch_k,
                    lexical_fetch_k=fetch_k,
                    q_op="OR",
                )
                search_mode = "OR fallback"

            current_candidates: list[str] = []
            for hit in hits:
                doc = hit.document
                matched = match_local_csv(doc, self.all_files)
                if matched is None or matched in current_candidates:
                    continue
                current_candidates.append(matched)
                previous_rank = self.state.best_ranks.get(matched)
                if previous_rank is None or hit.rank < previous_rank:
                    self.state.best_ranks[matched] = hit.rank
                    self.state.solr_meta[matched] = solr_metadata_from_doc(doc)
                    self.state.solr_meta[matched]["retrieval"] = hit.to_log_dict()
                    self.state.solr_meta[matched]["best_attempt"] = attempt
                    self.state.solr_meta[matched]["best_keywords"] = list(keywords)
                if matched not in self.state.all_candidates:
                    self.state.all_candidates.append(matched)
                if len(current_candidates) >= self.retrieval_config.top_k:
                    break

            # Preserve the retriever/Solr order after local-file mapping and
            # de-duplication.  The configured top_k is the only final cutoff;
            # no workflow-level schema heuristic re-ranks the candidates.
            candidates = self.state.all_candidates[: self.retrieval_config.top_k]
            self.state.search_attempts.append(
                {
                    "attempt": attempt,
                    "keywords": list(keywords),
                    "search_mode": search_mode,
                    "current_candidates": list(current_candidates),
                    "accumulated_candidates": list(candidates),
                }
            )

            if not current_candidates and not candidates:
                response = (
                    f"Attempt: {attempt}\nConcepts supplied: {self.state.used_keywords}\n"
                    "No tables found. Evaluate the empty candidate set."
                )
                self.state.search_cache[key] = response
                return response

            response = (
                f"Attempt: {attempt}\nConcepts supplied: {self.state.used_keywords}\n\n"
                "Candidates accumulated across all attempts (best schema/rank retained):\n"
                + format_candidate_context(candidates, self.state.solr_meta)
            )
            self.state.search_cache[key] = response
            return response
        except EmbeddingGenerationError as exc:
            detail = str(exc)
            cause = exc.__cause__
            if cause is not None:
                detail = f"{detail}: {cause}"
            self.state.semantic_failure = detail
            return (
                "Error generating the configured retrieval representation: "
                f"{detail}. The retrieval request has finished and must not be repeated."
            )
        except Exception as exc:
            return f"Error querying Solr: {exc}."

    def inspect_columns(self, file_name: str | None = None, filename: str | None = None) -> str:
        """
        Returns a compact profile for one table in the active dataset.
        Shows row count, bounded min/max coverage for temporal columns, column
        types, and sample values for low-cardinality categorical columns. At
        most two requests per file are useful; repeated requests use a cache.
        If the question has a date or time range, compare it with the reported
        temporal coverage before selecting the table.
        Use this only after identifying a valid table file with search_solr.
        """
        name = file_name or filename
        if not name:
            return "Error: file_name or filename parameter is required."
        name = name.strip()
        key = name.casefold()
        count = self.state.inspection_counts.get(key, 0) + 1
        self.state.inspection_counts[key] = count
        if count > MAX_INSPECTIONS_PER_FILE:
            return (
                f"Inspection skipped: {name} has already been inspected "
                f"{MAX_INSPECTIONS_PER_FILE} times. Use the cached schema and "
                "continue with confirm_unified_selection."
            )
        if key not in self.state.inspection_cache:
            self.state.inspection_cache[key] = _inspect_columns(self.csv_dir, name)
            return self.state.inspection_cache[key]
        return (
            f"Cached inspection (attempt {count}/{MAX_INSPECTIONS_PER_FILE}):\n"
            + self.state.inspection_cache[key]
        )

    def find_schema_matches(self, file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine to identify matching columns, then verify their practical
        joinability through overlap, coverage, uniqueness, cardinality, and
        estimated join expansion.
        """
        return _find_schema_matches(self.csv_dir, file_name_1, file_name_2)

    def confirm_unified_selection(self, reasoning: str, tables: list[str]) -> str:
        """
        CRITICAL: Use this tool ONLY when you have identified the required files after searching solr and inspecting them.
        Calling this tool terminates execution and confirms the selection.
        """
        final_tables = ", ".join(str(t) for t in tables)

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.search_solr,
                description=self._search_tool_description(),
            ),
            FunctionTool.from_defaults(fn=self.inspect_columns),
            FunctionTool.from_defaults(fn=self.find_schema_matches),
            FunctionTool.from_defaults(fn=self.confirm_unified_selection, fn_schema=ConfirmUnifiedSelectionSchema, return_direct=True),
        ]


def make_p12_tools(
    state: P12State,
    solr_client: LocalSolrClient,
    all_files: list[str],
    csv_dir: Path,
    question: str = "",
    retrieval_config: RetrievalConfig | None = None,
    retrieval_observer: Callable[[RetrievalRun], None] | None = None,
):
    """
    Build the tools for the unified Phase 1 & 2 agent and return an ObjectRetriever.
    The retriever will dynamically fetch the top relevant tools based on the agent's intent.
    """
    manager = Phase12ToolsManager(
        state,
        solr_client,
        all_files,
        csv_dir,
        question=question,
        retrieval_config=retrieval_config,
        retrieval_observer=retrieval_observer,
    )
    return manager.get_tools()
