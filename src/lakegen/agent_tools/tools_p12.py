import json
from pathlib import Path
from typing import Callable, Literal
from pydantic import BaseModel, Field

from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping

from lakegen.core.types import SolrMetadata
from lakegen.agent_tools.tools_p2 import (
    MAX_INSPECTIONS_PER_FILE,
    _find_schema_matches,
    _inspect_columns,
    _rank_for_missing_requirements,
    _requirement_terms,
    _temporal_coverage_issue,
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
    requirement_coverage: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description=(
            "Map each essential question requirement to an object containing "
            "the exact selected table in `table` and supporting column names "
            "in `columns`, e.g. {'requested year': {'table': "
            "'permits.parquet', 'columns': ['issue_date']}}."
        ),
    )
    table_roles: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map every selected table filename to its distinct role in the "
            "answer, e.g. fact records, lookup, or yearly partition."
        ),
    )
    combination_strategy: Literal[
        "single_table",
        "join",
        "concat_partitions",
        "aggregate_separately",
        "lookup",
        "compare",
    ] = Field(
        default="single_table",
        description="How the coder should combine the selected tables.",
    )
    uncovered_requirements: list[str] = Field(
        default_factory=list,
        description=(
            "Essential requirements still not proven by the selected tables. "
            "Use an empty list when coverage is complete."
        ),
    )
    alternatives_rejected: dict[str, dict[str, object] | str] = Field(
        default_factory=dict,
        description=(
            "At most two inspected alternative filenames mapped to objects with "
            "`matched_requirements` and one concrete `missing_requirement`. A "
            "legacy plain missing-requirement string is also accepted."
        ),
    )


class P12State:
    """State tracker for the Phase 1 & 2 unified agent."""
    def __init__(self):
        self.all_candidates: list[str] = []
        self.solr_meta: SolrMetadata = {}
        self.used_keywords: list[str] = []
        self.keyword_history: list[list[str]] = []
        self.best_ranks: dict[str, int] = {}
        self.candidate_scores: dict[str, float] = {}
        self.search_cache: dict[tuple[str, ...], str] = {}
        self.search_attempts: list[dict[str, object]] = []
        self.semantic_failure: str | None = None
        self.inspection_cache: dict[str, str] = {}
        self.inspection_counts: dict[str, int] = {}
        self.visible_candidate_count = 0
        self.expansion_count = 0
        self.expansion_requirements: list[str] = []
        self.rejected_selections: set[tuple[str, ...]] = set()
        self.selection_plan: dict[str, object] = {}
        self.selection_advisories: list[str] = []

    def inspected_candidates(self) -> list[str]:
        """Return successfully inspected candidates in retrieval order."""
        return [
            candidate
            for candidate in self.all_candidates
            if (
                candidate.casefold() in self.inspection_cache
                and not self.inspection_cache[candidate.casefold()].startswith("Error:")
            )
        ]


class Phase12ToolsManager:
    """Manager for Phase 1 & 2 unified tools to avoid closures and improve testability."""

    INITIAL_CANDIDATES = 10
    EXPANSION_SIZE = 5
    MAX_EXPANSIONS = 1
    MAX_SEARCH_ATTEMPTS = 2
    INITIAL_SHORTLIST_SIZE = 3
    MAX_INSPECTED_CANDIDATES = 5
    
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
        return tuple(dict.fromkeys(
            keyword.casefold() for keyword in keywords if keyword.strip()
        ))

    def _search_tool_description(self) -> str:
        return (
            "Search for relevant tables using the retrieval strategy configured by "
            "the experiment. Provide 1-2 concise dataset concepts in the portal's "
            "native language. The tool applies the original question and the "
            "configured retrieval parameters automatically. One refinement with "
            "genuinely different concepts is allowed before inspecting any table; "
            "identical or later searches are blocked. Use the bounded metadata and "
            "schema previews to shortlist the strongest candidates, then verify "
            "them with inspect_columns before selecting tables."
        )

    def search_solr(self, concepts_str: str = "") -> str:
        """Search for relevant tables using one or two dataset concepts."""
        try:
            supplied_concepts = [
                item.strip() for item in concepts_str.split(" ") if item.strip()
            ]
            keywords = list(supplied_concepts)
            if self.retrieval_config.mode in (
                RetrievalMode.SEMANTIC,
                RetrievalMode.PNEUMA,
            ):
                keywords = []
            elif not keywords:
                return "No concepts provided. Search with one or two dataset concepts."
            if (
                self.retrieval_config.mode
                in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID)
                and self.state.semantic_failure is not None
            ):
                return (
                    "Configured retrieval skipped: representation generation already "
                    "failed for this request. " + self.state.semantic_failure
                )
            key = self._search_cache_key(keywords)
            if key in self.state.search_cache:
                return (
                    "Search skipped: identical concepts were already used. Do not "
                    "repeat this search.\n"
                    + self.state.search_cache[key]
                )
            if self.state.inspection_cache:
                return (
                    "Search refinement blocked: a candidate has already been "
                    "inspected. Use the existing evidence or one guided expansion."
                )
            if len(self.state.search_attempts) >= self.MAX_SEARCH_ATTEMPTS:
                return (
                    "Search limit reached (2 distinct attempts). Do not call "
                    "search_solr again; inspect, expand once if needed, then select."
                )
            self.state.used_keywords = supplied_concepts
            self.state.keyword_history.append(keywords)
            attempt = len(self.state.keyword_history)

            if self.retrieval_observer is None:
                retriever = get_table_retrieval_service(
                    self.solr_client,
                    self.retrieval_config,
                    *([self.csv_dir] if self.retrieval_config.mode == RetrievalMode.DUCKDB_AGENTIC else []),
                )
            else:
                retriever = get_table_retrieval_service(
                    self.solr_client,
                    self.retrieval_config,
                    *([self.csv_dir] if self.retrieval_config.mode == RetrievalMode.DUCKDB_AGENTIC else []),
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
                self.state.candidate_scores[matched] = (
                    self.state.candidate_scores.get(matched, 0.0)
                    + 1.0 / (60.0 + hit.rank)
                )
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

            # Fuse at most two distinct agent searches with reciprocal-rank
            # contributions. The first search alone preserves its original order.
            self.state.all_candidates.sort(key=lambda candidate: (
                -self.state.candidate_scores.get(candidate, 0.0),
                self.state.best_ranks.get(candidate, 10**9),
                candidate,
            ))
            self.state.all_candidates = self.state.all_candidates[
                : self.retrieval_config.top_k
            ]
            candidates = self.state.all_candidates
            self.state.visible_candidate_count = min(
                self.INITIAL_CANDIDATES,
                len(candidates),
            )
            visible_candidates = candidates[: self.state.visible_candidate_count]
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
                "Candidates in Solr order after local-file mapping:\n"
                + format_candidate_context(visible_candidates, self.state.solr_meta)
                + (
                    f"\n{len(candidates) - len(visible_candidates)} additional "
                    "ranked candidates are available through expand_candidates."
                    if len(candidates) > len(visible_candidates)
                    else ""
                )
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
        Normally inspect the 2-4 strongest candidates from the bounded metadata
        preview instead of inspecting every retrieved table.
        """
        name = file_name or filename
        if not name:
            return "Error: file_name or filename parameter is required."
        name = name.strip()
        key = name.casefold()
        visible_candidates = self.state.all_candidates[
            : self.state.visible_candidate_count
        ]
        if name not in visible_candidates:
            return (
                f"Error: {name} is not currently visible. Inspect only candidates "
                "already shown by search_solr or expand_candidates."
            )
        attempted_candidates = len(self.state.inspection_counts)
        if key not in self.state.inspection_cache:
            current_limit = (
                self.INITIAL_SHORTLIST_SIZE
                if self.state.expansion_count == 0
                else self.MAX_INSPECTED_CANDIDATES
            )
            if attempted_candidates >= current_limit:
                if (
                    self.state.expansion_count == 0
                    and self.state.visible_candidate_count < len(self.state.all_candidates)
                ):
                    return (
                        "Inspection blocked: the initial shortlist is limited to 3 "
                        "candidates. If coverage is incomplete, call expand_candidates "
                        "before inspecting another candidate."
                    )
                return (
                    "Inspection blocked: at most 5 distinct candidates may be "
                    "inspected for this request."
                )
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

    def expand_candidates(self, missing_requirements: str) -> str:
        """Reveal hidden candidates that best cover a known metadata gap.

        First inspect the strongest plausible visible candidates and identify the
        missing measure, dimension, filter, period, or join key. This tool does
        not run or re-rank retrieval; it only reveals the next ranked block.
        After expansion, inspect only candidates whose metadata could fill the
        identified gap. Do not call again when no ranked candidates remain.
        """
        if not self.state.inspected_candidates():
            return (
                "Expansion blocked: inspect at least one plausible visible candidate "
                "before requesting more results."
            )
        requirements = _requirement_terms(missing_requirements)
        if not requirements:
            return (
                "Expansion blocked: provide concrete missing requirements such as "
                "a measure, dimension, period, filter, or join key."
            )
        if self.state.expansion_count >= self.MAX_EXPANSIONS:
            return (
                "Expansion limit reached. Do not call expand_candidates again "
                "or search_solr again; select among the inspected candidates."
            )
        if self.state.visible_candidate_count >= len(self.state.all_candidates):
            return "No additional candidates are available."
        start = self.state.visible_candidate_count
        hidden = self.state.all_candidates[start:]
        newly_visible = _rank_for_missing_requirements(
            hidden, self.state.solr_meta, requirements, self.EXPANSION_SIZE
        )
        if not newly_visible:
            self.state.expansion_count += 1
            self.state.expansion_requirements = requirements
            return (
                "No hidden candidate has metadata matching the missing requirements. "
                "Do not call expand_candidates or search_solr again; select or "
                "reject using the inspected evidence."
            )
        selected = set(newly_visible)
        self.state.all_candidates[start:] = [
            *newly_visible,
            *(candidate for candidate in hidden if candidate not in selected),
        ]
        self.state.visible_candidate_count = start + len(newly_visible)
        self.state.expansion_count += 1
        self.state.expansion_requirements = requirements
        remaining = len(self.state.all_candidates) - self.state.visible_candidate_count
        next_step = (
            f"{remaining} ranked candidates remain hidden, but the single guided "
            "expansion has been used. Do not call expand_candidates again."
            if remaining
            else (
                "All ranked candidates are now visible. Do not call "
                "expand_candidates again."
            )
        )
        return (
            "Guided expansion for missing requirements: "
            + ", ".join(requirements)
            + f"\nRevealed {len(newly_visible)} best-matching hidden candidates "
            "(original retrieval ranks are preserved in metadata):\n"
            + format_candidate_context(newly_visible, self.state.solr_meta)
            + f"\n\n{next_step}"
        )

    def find_schema_matches(self, file_name_1: str, file_name_2: str) -> str:
        """
        Use Valentine to identify matching columns, then verify their practical
        joinability through overlap, coverage, uniqueness, cardinality, and
        estimated join expansion.
        """
        return _find_schema_matches(self.csv_dir, file_name_1, file_name_2)

    def confirm_unified_selection(
        self,
        reasoning: str,
        tables: list[str],
        requirement_coverage: dict[str, dict[str, object]] | None = None,
        table_roles: dict[str, str] | None = None,
        combination_strategy: str = "single_table",
        uncovered_requirements: list[str] | None = None,
        alternatives_rejected: dict[str, dict[str, object] | str] | None = None,
    ) -> str:
        """
        CRITICAL: Use this tool ONLY when you have identified the required files after searching solr and inspecting them.
        Calling this tool terminates execution and confirms the selection.
        """
        normalized_tables = list(dict.fromkeys(str(table).strip() for table in tables))
        if not normalized_tables:
            raise ValueError("Selection blocked: at least one inspected table is required.")
        visible_candidates = self.state.all_candidates[
            : self.state.visible_candidate_count
        ]
        unknown = [table for table in normalized_tables if table not in visible_candidates]
        if unknown:
            raise ValueError(
                "Selection blocked: only currently visible candidates may be "
                f"selected. Unknown table(s): {unknown}."
            )

        inspected = {table.casefold() for table in self.state.inspected_candidates()}
        uninspected = [
            table for table in normalized_tables if table.casefold() not in inspected
        ]
        if uninspected:
            raise ValueError(
                "Selection blocked: inspect_columns is mandatory for every selected "
                f"table. Inspect {uninspected}, verify requirement and temporal "
                "coverage, then confirm again."
            )

        selection_key = tuple(sorted(table.casefold() for table in normalized_tables))
        if selection_key in self.state.rejected_selections:
            raise ValueError(
                "Selection blocked: this exact table combination was already "
                "rejected by the full-context coder. Select at least one different "
                "table or reject the keywords if no alternative exists."
            )

        coverage_issue = _temporal_coverage_issue(
            self.question,
            normalized_tables,
            self.state.inspection_cache,
        )
        if coverage_issue:
            raise ValueError(
                "Selection blocked by temporal validation: " + coverage_issue + "."
            )

        requirement_coverage = requirement_coverage or {}
        table_roles = table_roles or {}
        uncovered_requirements = list(dict.fromkeys(
            str(item).strip() for item in (uncovered_requirements or [])
            if str(item).strip()
        ))
        normalized_alternatives: dict[str, dict[str, object]] = {}
        for raw_table, raw_evidence in (alternatives_rejected or {}).items():
            table = str(raw_table).strip()
            if not table:
                continue
            if isinstance(raw_evidence, dict):
                matched = list(dict.fromkeys(
                    str(item).strip()
                    for item in raw_evidence.get("matched_requirements", [])
                    if str(item).strip()
                ))
                missing = str(raw_evidence.get("missing_requirement", "")).strip()
            else:
                matched = []
                missing = str(raw_evidence).strip()
            normalized_alternatives[table] = {
                "matched_requirements": matched,
                "missing_requirement": missing,
            }
        alternatives_rejected = normalized_alternatives
        selected_set = set(normalized_tables)
        advisories: list[str] = []

        missing_roles = [table for table in normalized_tables if not table_roles.get(table)]
        if missing_roles:
            advisories.append(
                "Selected table(s) without an explicit role: " + ", ".join(missing_roles)
            )

        role_extras = [table for table in table_roles if table not in selected_set]
        if role_extras:
            advisories.append(
                "Table role(s) refer to unselected tables: " + ", ".join(role_extras)
            )

        covered_tables: set[str] = set()
        malformed_requirements: list[str] = []
        for requirement, evidence in requirement_coverage.items():
            if not isinstance(evidence, dict):
                malformed_requirements.append(requirement)
                continue
            table = str(evidence.get("table", "")).strip()
            columns = evidence.get("columns", [])
            if table in selected_set:
                covered_tables.add(table)
            if table not in selected_set or not isinstance(columns, list) or not columns:
                malformed_requirements.append(requirement)
        if not requirement_coverage:
            advisories.append("No explicit requirement coverage was supplied.")
        elif malformed_requirements:
            advisories.append(
                "Requirement coverage lacks selected-table/column evidence for: "
                + ", ".join(malformed_requirements)
            )

        uncovered_tables = [table for table in normalized_tables if table not in covered_tables]
        if uncovered_tables:
            advisories.append(
                "Selected table(s) cover no explicit requirement: "
                + ", ".join(uncovered_tables)
            )

        if len(normalized_tables) > 1 and combination_strategy == "single_table":
            advisories.append(
                "Multiple tables were selected but the combination strategy is single_table."
            )
        elif len(normalized_tables) == 1 and combination_strategy != "single_table":
            advisories.append(
                "One table was selected but the combination strategy is "
                f"{combination_strategy}."
            )
        if uncovered_requirements:
            advisories.append(
                "Selection was confirmed with requirements still marked uncovered: "
                + ", ".join(uncovered_requirements)
            )
        if len(alternatives_rejected) > 2:
            advisories.append(
                "More than two rejected alternatives were supplied; keep only the "
                "strongest inspected alternatives in future confirmations."
            )
        inspected_names = set(self.state.inspected_candidates())
        unsupported_alternatives = [
            table for table in alternatives_rejected
            if table not in inspected_names
        ]
        if unsupported_alternatives:
            advisories.append(
                "Rejected alternative(s) were not inspected: "
                + ", ".join(unsupported_alternatives)
            )
        vague_missing = {
            "", "less relevant", "not relevant", "not needed", "weaker match",
            "lower ranked", "redundant", "inferior",
        }
        for table, evidence in alternatives_rejected.items():
            matched = evidence.get("matched_requirements", [])
            missing = str(evidence.get("missing_requirement", "")).casefold()
            if len(matched) >= 2 and missing in vague_missing:
                advisories.append(
                    f"{table} matches multiple essential requirements but no "
                    "concrete missing requirement justifies excluding it. Reconsider "
                    "including or preferring this inspected alternative."
                )

        self.state.selection_plan = {
            "requirement_coverage": requirement_coverage,
            "table_roles": table_roles,
            "combination_strategy": combination_strategy,
            "uncovered_requirements": uncovered_requirements,
            "alternatives_rejected": alternatives_rejected,
        }
        self.state.selection_advisories = advisories

        final_tables = ", ".join(normalized_tables)

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning,
            "selection_plan": self.state.selection_plan,
            "advisories": advisories,
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    def get_tools(self) -> list[FunctionTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.search_solr,
                description=self._search_tool_description(),
            ),
            FunctionTool.from_defaults(fn=self.inspect_columns),
            FunctionTool.from_defaults(fn=self.expand_candidates),
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
