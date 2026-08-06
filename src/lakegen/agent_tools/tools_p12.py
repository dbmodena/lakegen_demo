import json
import re
from pathlib import Path
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping

from lakegen.core.types import SolrMetadata
from lakegen.agent_tools.tools_p2 import _inspect_columns, _find_schema_matches
from src.client_solr import LocalSolrClient
from lakegen.phases.utils import match_local_csv, solr_metadata_from_doc, format_candidate_context
from lakegen.core.resources import get_table_retrieval_service
from lakegen.retrieval import RetrievalConfig, RetrievalMode


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


def _schema_overlap(question: str, metadata: dict) -> int:
    question_tokens = set(re.findall(r"[a-z0-9]+", question.casefold()))
    schema = " ".join(map(str, metadata.get("columns.name", []))).casefold()
    schema_tokens = set(re.findall(r"[a-z0-9]+", schema))
    return len(question_tokens & schema_tokens)


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
    ):
        self.state = state
        self.solr_client = solr_client
        self.all_files = all_files
        self.csv_dir = Path(csv_dir)
        self.question = question
        self.retrieval_config = retrieval_config or RetrievalConfig()

    def _search_cache_key(self, keywords: list[str]) -> tuple[str, ...]:
        if self.retrieval_config.mode == RetrievalMode.SEMANTIC:
            return ("semantic-question", self.question.strip().casefold())
        return (
            f"{self.retrieval_config.mode.value}-keywords",
            *sorted({keyword.casefold() for keyword in keywords}),
        )

    def _search_tool_description(self) -> str:
        if self.retrieval_config.mode == RetrievalMode.SEMANTIC:
            return (
                "Search for relevant tables with semantic KNN retrieval. The complete "
                "original user question is embedded automatically and exactly once. "
                "Do not invent or vary keywords; call this tool once with an empty "
                "string. Returns matching local table names and metadata."
            )
        if self.retrieval_config.mode == RetrievalMode.HYBRID:
            return (
                "Search for relevant tables with hybrid retrieval. Provide 1-2 "
                "metadata-oriented keywords in the portal's native language for the "
                "BM25 branch; the semantic branch automatically embeds the complete "
                "original user question. Returns matching local table names and metadata."
            )
        return (
            "Search for relevant tables with BM25. Provide 1-2 metadata-oriented "
            "keywords in the portal's native language. The generated keywords are "
            "used directly, with strict AND and an automatic OR fallback."
        )

    def search_solr(self, keywords_str: str = "") -> str:
        """
        Search Solr using the retrieval mode configured for this workflow.

        Keyword and hybrid modes consume the supplied metadata keywords. Semantic
        mode ignores this argument and embeds the complete original user question.
        """
        try:
            keywords = [k.strip() for k in keywords_str.split(" ") if k.strip()]
            if self.retrieval_config.mode == RetrievalMode.SEMANTIC:
                keywords = []
            elif not keywords:
                return "No keywords provided. Search with one or two dataset concepts."
            self.state.used_keywords = keywords
            key = self._search_cache_key(keywords)
            if key in self.state.search_cache:
                repeated_signal = (
                    "the original question"
                    if self.retrieval_config.mode == RetrievalMode.SEMANTIC
                    else f"keywords {keywords}"
                )
                return (
                    f"Search skipped: {repeated_signal} was already tried.\n"
                    + self.state.search_cache[key]
                )
            self.state.keyword_history.append(keywords)
            attempt = len(self.state.keyword_history)

            retriever = get_table_retrieval_service(
                self.solr_client, self.retrieval_config
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

            self.state.all_candidates.sort(
                key=lambda name: (
                    -_schema_overlap(self.question, self.state.solr_meta.get(name, {})),
                    self.state.best_ranks.get(name, 10**9),
                    name,
                )
            )
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
                    f"Attempt: {attempt}\nKeywords used: {keywords}\n"
                    "No tables found. Try with fewer or different keywords."
                )
                self.state.search_cache[key] = response
                return response

            response = (
                f"Attempt: {attempt}\nKeywords used: {keywords}\n"
                f"Retriever: {self.retrieval_config.mode}\n"
                f"Search mode: {search_mode}\n\n"
                "Candidates accumulated across all attempts (best schema/rank retained):\n"
                + format_candidate_context(candidates, self.state.solr_meta)
            )
            self.state.search_cache[key] = response
            return response
        except Exception as e:
            return f"Error querying Solr: {str(e)}. Try different keywords."

    def inspect_columns(self, file_name: str | None = None, filename: str | None = None) -> str:
        """
        Returns a compact profile for one table in the active dataset.
        Shows row count, full-file min/max coverage for temporal columns, column
        types, and sample values for low-cardinality categorical columns.
        If the question has a date or time range, compare it with the reported
        temporal coverage before selecting the table.
        Use this only after identifying a valid table file with search_solr.
        """
        name = file_name or filename
        if not name:
            return "Error: file_name or filename parameter is required."
        return _inspect_columns(self.csv_dir, name)

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
    )
    return manager.get_tools()
