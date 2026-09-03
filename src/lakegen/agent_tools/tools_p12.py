import json
import re
from pathlib import Path
from typing import Callable, Literal, Mapping
from pydantic import BaseModel, Field, model_validator

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
from lakegen.core.table_io import read_table
from lakegen.retrieval import (
    EmbeddingGenerationError,
    RetrievalConfig,
    RetrievalRun,
    RetrievalMode,
)


class SemanticFilterBinding(BaseModel):
    requirement: str
    table: str
    column: str
    operator: Literal["equals", "contains", "in", "range", "not_null", "other"]
    value: str = ""
    evidence: str


class SemanticTemporalFilterBinding(SemanticFilterBinding):
    """A row-level time constraint (dataset-edition years do not belong here)."""


class SemanticJoinBinding(BaseModel):
    tables: list[str]
    keys: dict[str, str]
    how: Literal["inner", "left", "right", "outer"] = "inner"
    evidence: str


class SemanticDimensionBinding(BaseModel):
    output: str
    table: str
    column: str
    evidence: str


class SemanticMeasureBinding(BaseModel):
    output: str
    operation: Literal[
        "count_rows", "count_distinct", "sum", "mean", "min", "max",
        "ratio", "difference", "custom",
    ]
    table: str
    columns: list[str]
    distinct: bool = False
    evidence: str


class SemanticOrdering(BaseModel):
    output: str
    direction: Literal["ascending", "descending"]


class SemanticAnalysisPlan(BaseModel):
    filters: list[SemanticFilterBinding] = Field(default_factory=list)
    temporal_filters: list[SemanticTemporalFilterBinding] = Field(default_factory=list)
    dimensions: list[SemanticDimensionBinding] = Field(default_factory=list)
    measures: list[SemanticMeasureBinding]
    joins: list[SemanticJoinBinding | str] = Field(default_factory=list)
    ordering: list[SemanticOrdering] = Field(default_factory=list)
    limit: int | None = None
    output_columns: list[str]
    null_policy: str = "preserve nulls unless the requested operation requires exclusion"
    table_roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_complete_non_empty_plan(self) -> "SemanticAnalysisPlan":
        if not self.measures:
            raise ValueError("measures must contain at least one structured binding")
        if not self.output_columns:
            raise ValueError("output_columns must not be empty")
        if not self.table_roles:
            raise ValueError("table_roles must cover the selected runtime tables")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be a positive integer or null")
        return self


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
    requirements: dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Compact semantic requirements only: grouping, measures, filters, "
            "ordering, limit, and optional explicit joins with left_table, "
            "left_columns, right_table, right_columns, and how."
        ),
    )
    semantic_plan: dict[str, object] | None = Field(
        default=None,
        description=(
            "Non-oracle executable semantics derived only from the question and "
            "inspected table evidence. Bind every filter, dimension, and measure "
            "to exact selected-table columns; never use benchmark expectations."
        )
    )


class SubmitSemanticPlanDraftSchema(BaseModel):
    draft: dict[str, object] = Field(
        description=(
            "Compact draft with filters [column, operator, value], dimensions "
            "[output, column], measures [output, operation, columns], optional "
            "joins, ordering [output, direction], and limit."
        )
    )


def _normalize_semantic_plan(
    plan: dict[str, object], selected_tables: list[str]
) -> dict[str, object]:
    """Normalize harmless tool-input variants without inventing semantic evidence."""
    normalized = dict(plan)
    operation_aliases = {
        "count": "count_rows", "avg": "mean", "average": "mean",
    }
    operator_aliases = {
        "is_not_null": "not_null", "year_eq": "equals", "year_equals": "equals",
    }
    default_table = selected_tables[0] if len(selected_tables) == 1 else ""
    for key in ("filters", "temporal_filters", "dimensions", "measures"):
        items: list[object] = []
        for raw in normalized.get(key, []) if isinstance(normalized.get(key), list) else []:
            if not isinstance(raw, dict):
                items.append(raw)
                continue
            item = dict(raw)
            if default_table and not item.get("table"):
                item["table"] = default_table
            if key in {"filters", "temporal_filters"}:
                operator = str(item.get("operator") or item.get("type") or "").casefold()
                item["operator"] = operator_aliases.get(operator, operator)
                value = item.get("value", "")
                if not isinstance(value, str):
                    item["value"] = json.dumps(value, ensure_ascii=False)
            elif key == "measures":
                operation = str(
                    item.get("operation") or item.get("aggregation") or item.get("type") or ""
                ).casefold()
                item["operation"] = operation_aliases.get(operation, operation)
                if "columns" not in item and item.get("column"):
                    item["columns"] = [item.pop("column")]
            items.append(item)
        normalized[key] = items
    joins: list[object] = []
    for raw in normalized.get("joins", []) if isinstance(normalized.get("joins"), list) else []:
        if not isinstance(raw, dict):
            joins.append(raw)
            continue
        join = dict(raw)
        if not join.get("tables") and join.get("left_table") and join.get("right_table"):
            join["tables"] = [join.pop("left_table"), join.pop("right_table")]
        if not join.get("keys") and join.get("left_columns") and join.get("right_columns"):
            tables = list(join.get("tables") or [])
            left_columns = list(join.pop("left_columns") or [])
            right_columns = list(join.pop("right_columns") or [])
            if len(tables) == 2 and len(left_columns) == len(right_columns) == 1:
                join["keys"] = {
                    str(tables[0]): str(left_columns[0]),
                    str(tables[1]): str(right_columns[0]),
                }
        if not join.get("keys") and join.get("left_key") and join.get("right_key"):
            tables = list(join.get("tables") or [])
            if len(tables) == 2:
                join["keys"] = {
                    str(tables[0]): join.pop("left_key"),
                    str(tables[1]): join.pop("right_key"),
                }
        joins.append(join)
    normalized["joins"] = joins
    return normalized


def _draft_item(raw: object, names: tuple[str, ...]) -> dict[str, object]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        return {name: raw[index] for index, name in enumerate(names) if index < len(raw)}
    return {}


def compile_semantic_plan_draft(
    draft: dict[str, object], selected_tables: list[str],
    table_roles: dict[str, str], schema_by_table: dict[str, set[str]],
) -> dict[str, object]:
    """Compile an explicit model draft; never infer missing semantic choices."""
    default_table = selected_tables[0] if len(selected_tables) == 1 else ""

    def bind(raw: object, names: tuple[str, ...], kind: str) -> dict[str, object]:
        item = _draft_item(raw, names)
        table = str(item.get("table") or default_table)
        if not table:
            raise ValueError(f"{kind} binding must name a table for multi-table plans")
        column_names = item.get("columns")
        if column_names is None and item.get("column"):
            column_names = [item["column"]]
        columns = [str(value) for value in (column_names or [])]
        evidence_columns = columns or [str(item.get("column") or "")]
        evidence_columns = [column for column in evidence_columns if column]
        missing = [column for column in evidence_columns if column not in schema_by_table.get(table, set())]
        if table not in selected_tables or missing:
            raise ValueError(
                f"{kind} binding is not supported by inspected schema: "
                f"table={table!r}, missing_columns={missing}"
            )
        output = str(item.get("output") or "").strip()
        column = str(item.get("column") or "").strip()
        requirement = str(item.get("requirement") or output or column).strip()
        if not requirement:
            raise ValueError(f"{kind} binding needs an explicit output or requirement")
        item["table"] = table
        item["requirement"] = requirement
        item["evidence"] = (
            f"Inspected runtime schema for {table} contains: "
            + ", ".join(evidence_columns)
        )
        return item

    filters = [bind(raw, ("column", "operator", "value"), "filter")
               for raw in draft.get("filters", []) if raw is not None]
    temporal = [bind(raw, ("column", "operator", "value"), "temporal_filter")
                for raw in draft.get("temporal_filters", []) if raw is not None]
    dimensions = [bind(raw, ("output", "column"), "dimension")
                  for raw in draft.get("dimensions", []) if raw is not None]
    measures = [bind(raw, ("output", "operation", "columns"), "measure")
                for raw in draft.get("measures", []) if raw is not None]
    if not measures:
        raise ValueError("draft must contain at least one explicit measure")
    output_columns = [
        str(item.get("output")) for item in [*dimensions, *measures]
        if str(item.get("output") or "").strip()
    ]
    if not output_columns:
        raise ValueError("draft bindings must declare output names")
    normalized_join_plan = _normalize_semantic_plan(
        {"joins": list(draft.get("joins", []))}, selected_tables
    )
    joins: list[dict[str, object] | str] = []
    for raw in normalized_join_plan.get("joins", []):
        if isinstance(raw, str):
            joins.append(raw)
            continue
        if not isinstance(raw, dict):
            raise ValueError("join must be a structured object or explicit legacy string")
        join = dict(raw)
        tables = [str(table) for table in join.get("tables", [])]
        keys = {str(table): str(column) for table, column in dict(join.get("keys") or {}).items()}
        if len(tables) < 2 or any(table not in selected_tables for table in tables):
            raise ValueError("join tables must be explicitly selected")
        missing_keys = [
            f"{table}.{keys.get(table, '')}" for table in tables
            if not keys.get(table) or keys[table] not in schema_by_table.get(table, set())
        ]
        if missing_keys:
            raise ValueError("join keys missing from inspected schemas: " + ", ".join(missing_keys))
        join["tables"] = tables
        join["keys"] = keys
        join["evidence"] = "Inspected runtime schemas contain join keys: " + ", ".join(
            f"{table}.{keys[table]}" for table in tables
        )
        joins.append(join)
    plan = {
        "filters": filters,
        "temporal_filters": temporal,
        "dimensions": dimensions,
        "measures": measures,
        "joins": joins,
        "ordering": [
            _draft_item(raw, ("output", "direction"))
            for raw in draft.get("ordering", []) if raw is not None
        ],
        "limit": draft.get("limit"),
        "output_columns": output_columns,
        "null_policy": str(draft.get("null_policy") or (
            "preserve nulls unless the requested operation requires exclusion"
        )),
        "table_roles": dict(table_roles),
    }
    return _normalize_semantic_plan(plan, selected_tables)


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
        self.initial_stall_reason: str | None = None
        self.recovery_started = False
        self.recovery_stop_reason: str | None = None
        self.selection_plan_source = "none"
        self.confirmed_tables: list[str] = []
        self.selection_reasoning = ""
        self.selection_requirements: dict[str, object] = {}
        self.semantic_draft: dict[str, object] = {}
        self.semantic_planner_attempts = 0
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
        inspected: list[str] = []
        for candidate in self.all_candidates:
            filename: str | None = None
            if isinstance(candidate, str):
                filename = candidate.strip()
            elif isinstance(candidate, Mapping):
                filename = next((
                    str(candidate[key]).strip()
                    for key in ("file", "filename", "dataset")
                    if isinstance(candidate.get(key), str)
                    and str(candidate[key]).strip()
                ), None)
            if not filename:
                continue
            cached = self.inspection_cache.get(filename.casefold())
            if isinstance(cached, str) and not cached.startswith("Error:"):
                inspected.append(filename)
        return list(dict.fromkeys(inspected))


class Phase12ToolsManager:
    """Manager for Phase 1 & 2 unified tools to avoid closures and improve testability."""

    INITIAL_CANDIDATES = 10
    EXPANSION_SIZE = 5
    MAX_EXPANSIONS = 1
    MAX_SEARCH_ATTEMPTS = 1
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
            "configured retrieval parameters automatically. Exactly one initial "
            "search is allowed. Use the bounded metadata and "
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
                    "Search limit reached (1 initial attempt). Do not call "
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
        requirements: dict[str, object] | None = None,
        semantic_plan: dict[str, object] | SemanticAnalysisPlan | None = None,
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

        contract_first = semantic_plan is not None
        if isinstance(semantic_plan, SemanticAnalysisPlan):
            semantic_plan = semantic_plan.model_dump()
        if contract_first:
            if isinstance(semantic_plan, dict) and not semantic_plan.get("table_roles"):
                semantic_plan = {**semantic_plan, "table_roles": dict(table_roles)}
            if isinstance(semantic_plan, dict):
                semantic_plan = _normalize_semantic_plan(
                    semantic_plan, normalized_tables
                )
            try:
                semantic_plan = SemanticAnalysisPlan.model_validate(semantic_plan).model_dump()
            except Exception as exc:
                self.state.semantic_failure = str(exc)
                raise ValueError(
                    "Selection blocked: invalid semantic_plan: "
                    f"{exc}\nCorrect the listed fields and call "
                    "confirm_unified_selection again. Do not return the corrected "
                    "JSON only as assistant text."
                ) from exc

            schema_by_table: dict[str, set[str]] = {}
            for table in normalized_tables:
                try:
                    schema_by_table[table] = {
                        str(column) for column in read_table(
                            self.csv_dir / table, nrows=0
                        ).columns
                    }
                except Exception as exc:
                    raise ValueError(
                        f"Selection blocked: cannot validate schema for {table}: {exc}"
                    ) from exc
            invalid_bindings: list[str] = []
            bindings = [
                *semantic_plan["filters"],
                *semantic_plan["temporal_filters"],
                *semantic_plan["dimensions"],
                *semantic_plan["measures"],
            ]
            for binding in bindings:
                table = str(binding.get("table", ""))
                columns = binding.get("columns", [binding.get("column", "")])
                if table not in selected_set:
                    invalid_bindings.append(f"{table}: table is not selected")
                    continue
                for column in columns:
                    if str(column) not in schema_by_table.get(table, set()):
                        invalid_bindings.append(f"{table}.{column}: column not found")
                if not str(binding.get("evidence", "")).strip():
                    invalid_bindings.append(f"{table}: missing binding evidence")
            if invalid_bindings:
                raise ValueError(
                    "Selection blocked: semantic bindings are not supported by the "
                    "selected schemas: " + "; ".join(invalid_bindings)
                )

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

        selection_blockers: list[str] = []
        if missing_roles:
            selection_blockers.append("every selected table needs an explicit role")
        if not requirement_coverage:
            selection_blockers.append("requirement_coverage is required")
        if malformed_requirements:
            selection_blockers.append(
                "requirement coverage must bind selected tables and columns"
            )
        if uncovered_tables:
            selection_blockers.append(
                "every selected table must support at least one requirement"
            )
        if uncovered_requirements:
            selection_blockers.append(
                "uncovered_requirements must be resolved before confirmation"
            )
        if (
            len(normalized_tables) > 1
            and combination_strategy == "single_table"
        ) or (
            len(normalized_tables) == 1
            and combination_strategy != "single_table"
        ):
            selection_blockers.append(
                "combination_strategy must match the selected table count"
            )
        if contract_first and selection_blockers:
            raise ValueError(
                "Selection blocked by contract-first validation: "
                + "; ".join(selection_blockers)
                + ". Inspect/expand the existing ranked candidates and confirm again."
            )

        self.state.selection_plan = {
            "requirement_coverage": requirement_coverage,
            "table_roles": table_roles,
            "combination_strategy": combination_strategy,
            "uncovered_requirements": uncovered_requirements,
            "alternatives_rejected": alternatives_rejected,
            "requirements": dict(requirements or {}),
            "coder_brief": self._build_coder_brief(
                normalized_tables,
                requirement_coverage,
                dict(requirements or {}),
                combination_strategy,
                table_roles,
                semantic_plan if isinstance(semantic_plan, dict) else None,
            ),
            **({"semantic_plan": semantic_plan} if contract_first else {}),
        }
        self.state.semantic_failure = None
        self.state.selection_plan_source = (
            "confirm_unified_selection" if contract_first else "selection_only"
        )
        self.state.confirmed_tables = list(normalized_tables)
        self.state.selection_reasoning = reasoning
        self.state.selection_requirements = dict(requirements or {})
        self.state.selection_advisories = advisories

        final_tables = ", ".join(normalized_tables)

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning,
            "selection_plan": self.state.selection_plan,
            "advisories": advisories,
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    def _build_coder_brief(
        self,
        tables: list[str],
        coverage: dict[str, dict[str, object]],
        requirements: dict[str, object],
        combination_strategy: str,
        table_roles: dict[str, str],
        semantic_plan: dict[str, object] | None,
    ) -> dict[str, object]:
        """Normalize the agent's explicit choices without making new choices."""
        schemas: dict[str, list[str]] = {}
        for table in tables:
            try:
                schemas[table] = [str(column) for column in read_table(
                    self.csv_dir / table, nrows=0
                ).columns]
            except Exception:
                # Phase 3 remains the authoritative readability gate. Discovery
                # tests and remote adapters may expose inspection text only.
                schemas[table] = []
        selected_columns: dict[str, list[str]] = {table: [] for table in tables}
        normalization_errors: list[str] = []

        def canonical(value: str) -> str:
            return re.sub(r"[^a-z0-9]", "", value.casefold())

        def resolve(table: str, raw_column: object) -> str | None:
            raw = str(raw_column).strip()
            if not raw or table not in schemas:
                return None
            annotated = re.fullmatch(r"(.+?)\s*\(([^()]+\.(?:parquet|csv))\)\s*", raw)
            if annotated:
                raw = annotated.group(1).strip()
                annotated_table = annotated.group(2).strip()
                if annotated_table != table:
                    return None
            exact = [column for column in schemas[table] if column == raw]
            if len(exact) == 1:
                return exact[0]
            folded = [column for column in schemas[table] if column.casefold() == raw.casefold()]
            if len(folded) == 1:
                return folded[0]
            mechanical = [column for column in schemas[table] if canonical(column) == canonical(raw)]
            return mechanical[0] if len(mechanical) == 1 else None

        def resolve_unique(raw_column: object) -> tuple[str, str] | None:
            matches = [
                (table, column)
                for table in tables
                if (column := resolve(table, raw_column)) is not None
            ]
            return matches[0] if len(matches) == 1 else None

        annotated_join_columns: list[tuple[str, str]] = []
        for requirement, evidence in coverage.items():
            if not isinstance(evidence, dict):
                continue
            evidence_table = str(evidence.get("table", "")).strip()
            for raw_column in evidence.get("columns", []):
                annotation = re.fullmatch(
                    r"(.+?)\s*\(([^()]+\.(?:parquet|csv))\)\s*",
                    str(raw_column).strip(),
                )
                table = annotation.group(2).strip() if annotation else evidence_table
                column = resolve(table, raw_column)
                if column and table in selected_columns:
                    if column not in selected_columns[table]:
                        selected_columns[table].append(column)
                    if evidence_table not in tables or "join" in requirement.casefold():
                        annotated_join_columns.append((table, column))
                else:
                    normalization_errors.append(
                        f"{requirement}: {raw_column!s} is not one unambiguous column of {table or 'a selected table'}"
                    )

        joins = list(requirements.get("joins", [])) if isinstance(
            requirements.get("joins"), list
        ) else []
        if not joins and len({table for table, _ in annotated_join_columns}) == 2:
            left, right = annotated_join_columns[:2]
            if left[0] != right[0]:
                joins = [{
                    "left_table": left[0], "left_columns": [left[1]],
                    "right_table": right[0], "right_columns": [right[1]],
                    "how": "inner",
                }]
        dimensions: list[object] = []
        raw_grouping = requirements.get("grouping", [])
        if isinstance(raw_grouping, list):
            for grouping in raw_grouping:
                if not isinstance(grouping, str):
                    dimensions.append(grouping)
                    continue
                match = resolve_unique(grouping)
                dimensions.append(
                    {"table": match[0], "column": match[1], "output": grouping}
                    if match else grouping
                )

        brief: dict[str, object] = {
            "tables": list(tables),
            "selected_columns": selected_columns,
            "filters": list(requirements.get("filters", []))
            if isinstance(requirements.get("filters"), list) else [],
            "temporal_filters": [],
            "dimensions": dimensions,
            "measures": list(requirements.get("measures", []))
            if isinstance(requirements.get("measures"), list) else [],
            "result_type": str(requirements.get("result_type") or "auto"),
            "ordering": requirements.get("ordering"),
            "limit": requirements.get("limit"),
            "joins": joins,
            "output_columns": list(requirements.get("output_columns", []))
            if isinstance(requirements.get("output_columns"), list) else [],
            "null_policy": str(requirements.get("null_policy") or ""),
            "table_roles": dict(table_roles),
            "normalization_errors": normalization_errors,
        }
        if semantic_plan:
            brief["filters"] = list(semantic_plan.get("filters", []))
            brief["temporal_filters"] = list(semantic_plan.get("temporal_filters", []))
            brief["dimensions"] = list(semantic_plan.get("dimensions", []))
            brief["measures"] = list(semantic_plan.get("measures", []))
            brief["ordering"] = semantic_plan.get("ordering")
            brief["limit"] = semantic_plan.get("limit")
            brief["joins"] = list(semantic_plan.get("joins", []))
            brief["output_columns"] = list(semantic_plan.get("output_columns", []))
            brief["null_policy"] = str(semantic_plan.get("null_policy") or "")
            brief["table_roles"] = dict(semantic_plan.get("table_roles") or {})
            for group in ("dimensions", "measures", "filters", "temporal_filters"):
                for binding in semantic_plan.get(group, []):
                    if not isinstance(binding, dict):
                        continue
                    table = str(binding.get("table") or (tables[0] if len(tables) == 1 else ""))
                    for raw_column in binding.get("columns", [binding.get("column")]):
                        column = resolve(table, raw_column)
                        if column and column not in selected_columns.get(table, []):
                            selected_columns[table].append(column)
        if len(tables) > 1 and combination_strategy != "single_table":
            brief["combination_strategy"] = combination_strategy
        # Canonicalize harmless join-shape variants before the brief crosses
        # the Phase 2/3 boundary. Semantic strings are preserved verbatim.
        brief = _normalize_semantic_plan(brief, tables)
        # Temporary read compatibility for consumers of the old brief shape.
        # Phase 3 uses only the canonical fields above.
        brief["operations"] = list(brief["measures"])
        brief["task"] = {
            **dict(requirements),
            "grouping": ([
                str(item.get("output") or item.get("column") or "")
                for item in brief["dimensions"] if isinstance(item, dict)
            ] or list(requirements.get("grouping", []))),
        }
        return brief

    def submit_semantic_plan_draft(self, draft: dict[str, object]) -> str:
        """Compile and validate a compact benchmark-blind semantic draft."""
        self.state.semantic_planner_attempts += 1
        if self.state.semantic_planner_attempts > 2:
            raise ValueError("semantic planner correction limit reached (2)")
        tables = list(self.state.confirmed_tables)
        if not tables:
            raise ValueError("table selection must be confirmed before semantic planning")
        roles = dict(self.state.selection_plan.get("table_roles") or {})
        schema_by_table = {
            table: {str(column) for column in read_table(self.csv_dir / table, nrows=0).columns}
            for table in tables
        }
        try:
            compiled = compile_semantic_plan_draft(
                draft, tables, roles, schema_by_table
            )
            self.state.semantic_draft = dict(draft)
            return self.confirm_unified_selection(
                self.state.selection_reasoning,
                tables,
                requirement_coverage=dict(
                    self.state.selection_plan.get("requirement_coverage") or {}
                ),
                table_roles=roles,
                combination_strategy=str(
                    self.state.selection_plan.get("combination_strategy") or "single_table"
                ),
                uncovered_requirements=list(
                    self.state.selection_plan.get("uncovered_requirements") or []
                ),
                alternatives_rejected=dict(
                    self.state.selection_plan.get("alternatives_rejected") or {}
                ),
                requirements=dict(self.state.selection_requirements),
                semantic_plan=compiled,
            )
        except Exception as exc:
            self.state.semantic_failure = str(exc)
            raise ValueError(
                "Semantic draft validation failed. Correct only the reported fields "
                f"and call submit_semantic_plan_draft again. Error: {exc}"
            ) from exc

    def get_semantic_planner_tools(self) -> list[FunctionTool]:
        return [FunctionTool.from_defaults(
            fn=self.submit_semantic_plan_draft,
            fn_schema=SubmitSemanticPlanDraftSchema,
            return_direct=True,
        )]

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
