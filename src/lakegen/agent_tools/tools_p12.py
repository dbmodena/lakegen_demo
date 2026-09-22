import itertools
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence
from pydantic import BaseModel, Field, model_validator

from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping

from lakegen.core.types import SolrMetadata, StreamCallback
from lakegen.experiment_config import DiscoveryConfig
from lakegen.keyword_terms import keyword_terms, minimal_failing_subsets, split_keywords
from lakegen.agent_tools.tools_p2 import (
    MIN_BAN_JUSTIFICATION_CHARS,
    _check_join_union,
    _inspect_columns,
    _rank_for_missing_requirements,
    _requirement_terms,
    _temporal_coverage_issue,
)
from lakegen.agent_tools.requirement_ledger import (
    build_requirement_ledger,
    requirement_ledger_blockers,
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
        description=(
            "Map each essential question requirement to an object containing "
            "the exact selected table in `table` and supporting column names "
            "in `columns`, e.g. {'requested year': {'table': "
            "'permits.parquet', 'columns': ['issue_date']}}. `table` must be "
            "one of the `tables` selected in this same call. `columns` is "
            "REQUIRED and must be a list, even for a single column -- "
            "['issue_date'], never 'issue_date'. Every essential requirement "
            "needs its own entry, and every selected table must be the "
            "`table` of at least one entry."
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
        description=(
            "Compact semantic requirements only: grouping, measures, filters, "
            "result_type, ordering, limit. Every filter must name its table, "
            "column, operator and value; keep different periods separate. "
            "Optional explicit joins use left_table, "
            "left_columns, right_table, right_columns, and how."
        ),
    )
    semantic_plan: dict[str, object] | None = Field(
        default=None,
        description=(
            "Non-oracle executable semantics derived only from the question and "
            "inspected table evidence; never use benchmark expectations. A JSON "
            "object -- never a SQL string or prose -- with these keys: "
            "`measures` (REQUIRED, at least one) [{output, operation "
            "(count_rows|count_distinct|sum|mean|min|max|ratio|difference|"
            "custom), table, columns, evidence}]; `dimensions` [{output, table, "
            "column, evidence}]; `filters`/`temporal_filters` [{requirement, "
            "table, column, operator (equals|contains|in|range|not_null|other), "
            "value, evidence}]; `joins` [{tables, keys: {table: column}, how, "
            "evidence}]; `ordering` [{output, direction}]; `limit` (int or "
            "null); `null_policy` (str). Every binding's `table` must be one of "
            "the selected tables and every `column`/`columns` must be an exact "
            "inspected column name; `evidence` cites the inspected schema that "
            "justifies the binding. Omit `output_columns` and `table_roles` "
            "here -- they are filled in automatically from `dimensions`/"
            "`measures` outputs and from this call's own `table_roles` argument."
        )
    )


class RejectUnifiedSelectionSchema(BaseModel):
    reasoning: str = Field(description="Explain step-by-step why the current candidates are not good.")
    suggestion: str = Field(
        description="Suggest dataset concepts, not analytical operations or row-filter values."
    )
    ban_tables: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Already-inspected candidates you judge highly irrelevant -- they "
            "satisfy none of the essential requirements -- mapped to the "
            "concrete evidence proving that (e.g. 'Parent Department is only "
            "Crown Prosecution Service, never the requested department'), so "
            "they are excluded from later retrieval for this question. Every "
            "other inspected candidate is kept and carries over to the next "
            "attempt, even if it does not yet cover every requirement -- "
            "only actively-proven-irrelevant tables belong here. A table "
            "without a concrete justification is not banned."
        ),
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
        # A scalar lookup ("the area of X") is not an aggregate; the model
        # reaches for a passthrough name here often enough that it is worth
        # mapping onto the schema's own escape hatch rather than failing.
        "value": "custom", "identity": "custom", "lookup": "custom",
        "none": "custom", "select": "custom", "raw": "custom",
    }
    operator_aliases = {
        "is_not_null": "not_null", "year_eq": "equals", "year_equals": "equals",
    }
    direction_aliases = {
        "asc": "ascending", "desc": "descending",
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
    if isinstance(normalized.get("ordering"), list):
        # A coder_brief's own `ordering` may be an arbitrary pass-through value
        # (e.g. a free-form string), not this structured shape -- touch it only
        # when it is already a list, never coerce a non-list value away.
        ordering: list[object] = []
        for raw in normalized["ordering"]:
            if not isinstance(raw, dict):
                ordering.append(raw)
                continue
            item = dict(raw)
            direction = str(item.get("direction") or "").casefold()
            item["direction"] = direction_aliases.get(direction, direction)
            ordering.append(item)
        normalized["ordering"] = ordering
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


def _normalize_requirement_coverage(
    requirement_coverage: dict[str, object], selected_tables: list[str]
) -> dict[str, object]:
    """Normalize harmless requirement_coverage input variants.

    Never guesses a table when more than one is selected -- that would bind a
    requirement to evidence the model never actually gave -- but the two
    mistakes actually seen in practice (a single column string instead of a
    list, and an omitted table when only one is selected) are unambiguous.
    """
    default_table = selected_tables[0] if len(selected_tables) == 1 else ""
    normalized: dict[str, object] = {}
    for requirement, evidence in requirement_coverage.items():
        if not isinstance(evidence, dict):
            normalized[requirement] = evidence
            continue
        item = dict(evidence)
        table = str(item.get("table") or "").strip()
        item["table"] = table or default_table
        columns = item.get("columns")
        if columns is None and item.get("column"):
            columns = [item.pop("column")]
        elif isinstance(columns, str):
            columns = [columns]
        if columns is not None:
            item["columns"] = columns
        normalized[requirement] = item
    return normalized


def _derive_output_columns(plan: dict[str, object]) -> list[str]:
    """Name every result column once, from each dimension's/measure's `output`.

    Repeating that same list back as `output_columns` is a common, harmless
    omission in a model-supplied semantic_plan; only used where that field is
    genuinely optional input, never to invent a coder_brief's own explicit
    (and separately meaningful, possibly intentionally empty) output_columns.
    """
    return [
        str(item.get("output")) for item in [
            *plan.get("dimensions", []), *plan.get("measures", [])
        ]
        if isinstance(item, dict) and str(item.get("output") or "").strip()
    ]


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
        # Minimal antichain of AND keyword combinations that returned no local
        # candidates.  If {a} fails, keeping the older {a, b} adds no
        # information: every future AND query containing a is already doomed.
        self.failed_keyword_combinations: list[frozenset[str]] = []
        self.best_ranks: dict[str, int] = {}
        self.candidate_scores: dict[str, float] = {}
        self.search_cache: dict[tuple[str, ...], str] = {}
        self.search_attempts: list[dict[str, object]] = []
        # Retrieval representation (e.g. embedding) generation failure: blocks
        # re-running search_tables, since retrieval itself is what's broken.
        self.semantic_failure: str | None = None
        # semantic_plan / draft validation failure from confirm_unified_selection
        # or submit_semantic_plan_draft. Unrelated to retrieval, so it must never
        # gate search_tables -- only a bad plan, not the search, needs retrying.
        self.plan_failure: str | None = None
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
        self.rejection_keep_tables: list[str] = []
        self.rejection_skip_tables: list[str] = []
        # Question-scoped retrieval evidence persisted by the service after a
        # completed agent round. It is intentionally separate from the hard
        # zero-result AND banlist above.
        self.retrieval_memory_events: list[dict[str, object]] = []
        # Cross-round memory, seeded by the caller before this round starts
        # (mirrors DIVIDED's excluded_tables/carried_tables in service.py).
        # excluded_tables: proven-insufficient candidates from a prior round,
        # in this question, filtered out of fresh retrieval entirely.
        # carried_tables: candidates a prior round already verified satisfy
        # part of the question; re-surfaced here regardless of whether this
        # round's retrieval finds them again.
        self.excluded_tables: set[str] = set()
        self.carried_tables: list[str] = []
        self.carried_metadata: SolrMetadata = {}

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


# ============================================================================
# decompose_preview_search (DiscoveryConfig flag): split one question into its
# per-table distinguishing-detail search phrases, preview a few real AND-word
# candidates per phrase (with match counts and top matches) and let the model
# pick before any of them is actually run. Ported from
# experiments/retrieval_lab's decompose_tuned_preview_noacronym, the strongest
# keyword-search arm found there -- a real, if modest, improvement over a
# single free-form search at every scale tested (84/50/100/249 questions).
# ============================================================================

# Generic function words, stripped before a phrase's content words are counted
# against the index.
_SEARCH_STOPWORDS = set(
    "a an the of for in on at to and or by with from as is are was were be "
    "been that this these those what which who how many much per its their "
    "our your not no than then into over under between".split()
)
# Quantifier/aggregation words: they describe how to compute the answer, not
# what the dataset is, so they never occur in a table's title, description or
# publisher and only ever narrow or kill an AND query.
_AGGREGATION_STOPWORDS = set(
    "total average count counts correlation proportion percentage share rank "
    "distinct most largest smallest top each combined since increase decrease "
    "change difference number".split()
)

_DECOMPOSE_SYSTEM_PROMPT = """Some questions need SEVERAL tables from an open-data portal. Portals publish the same dataset many times: one file per year, month or snapshot, per edition, per agency or per area, all with almost identical descriptions. A question that compares, combines or spans such files needs each of them, and the files differ only in their distinguishing detail (a date, a year, an edition, an agency, a place) - or the question combines genuinely different datasets.
List the distinct tables the question needs. For each, write one standalone search request in plain words: the dataset's subject words plus THAT table's own distinguishing detail (its date, agency, edition or place). Write a SEPARATE entry for each distinct date or edition the question mentions, even when several appear in the same clause (e.g. "June 2022 and July 2023" needs two entries, one per month) -- never fold two into one request. Name the organisation exactly as the question names it; do not expand an acronym or guess a fuller official name yourself.
Never write the same request twice. If one table is enough, return a single request.
Return JSON: {"tables": [{"detail": "...", "request": "..."}]} with 1 to 4 entries."""

_CANDIDATE_PICK_SYSTEM_PROMPT = """You help a keyword search find the right table(s) in an open-data portal. Every candidate search below is an AND search (a table must contain every word) that the portal already ran; you see how many tables it matches and the first few matches. Pick the searches most likely to surface the table(s) THE QUESTION IS ABOUT: their top matches must name the question's subject, organisation, place and period. Prefer searches whose matches are on-topic over ones that are merely small. Return JSON: {"picks": [numbers]} with 1 to 3 numbers."""


def _extract_json_object(text: str) -> dict:
    """Parse one JSON object out of an LLM response, tolerating code fences and
    leading prose by scanning for the first balanced ``{...}``."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        loaded = None
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                loaded = candidate
                break
        if loaded is None:
            raise
    if not isinstance(loaded, dict):
        raise ValueError("response must be a JSON object")
    return loaded


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
        discovery_config: DiscoveryConfig | None = None,
        notice_callback: StreamCallback | None = None,
        llm: LLM | None = None,
    ):
        self.state = state
        self.solr_client = solr_client
        self.all_files = all_files
        self.csv_dir = Path(csv_dir)
        self.question = question
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.retrieval_observer = retrieval_observer
        self.discovery = discovery_config or DiscoveryConfig()
        self.notice_callback = notice_callback
        # Only used by decompose_preview_search (see DiscoveryConfig): the same
        # LLM instance driving the agent, reused for the internal decompose and
        # candidate-pick calls rather than constructing a second, separate one.
        self.llm = llm
        # decompose_preview_search only: per-word Solr match counts, memoized
        # for the life of this manager so sub-questions that share a word don't
        # re-query it.
        self._word_freq_cache: dict[str, int] = {}

    def _emit_notice(self, text: str) -> None:
        """Report to the operator, never to the agent.

        This is deliberately not part of any tool's return value: the agent must
        choose concepts on the merits of the question, not adapt to whichever
        retriever happens to be configured. Falls back to stdout when no channel
        is wired, so a standalone run still says what searched.
        """
        if self.notice_callback is not None:
            self.notice_callback(text)
        else:
            print(text, end="", flush=True)

    def _record_retrieval_memory_event(self, **event: object) -> None:
        """Collect bounded, factual retrieval feedback for persistence."""
        self.state.retrieval_memory_events.append(dict(event))

    def _search_cache_key(self, keywords: list[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            keyword.casefold() for keyword in keywords if keyword.strip()
        ))

    def _failed_keyword_subset(self, keywords: list[str]) -> frozenset[str] | None:
        """Return the smallest known failed subset contained in ``keywords``."""
        proposed = keyword_terms(keywords)
        matches = [failed for failed in self.state.failed_keyword_combinations
                   if failed <= proposed]
        return min(matches, key=lambda item: (len(item), sorted(item)), default=None)

    def _ban(self, terms: frozenset[str]) -> None:
        """Add one failed AND term set and prune its now-redundant supersets."""
        if not terms or any(known <= terms for known in self.state.failed_keyword_combinations):
            return
        self.state.failed_keyword_combinations = [
            known for known in self.state.failed_keyword_combinations
            if not terms < known
        ]
        self.state.failed_keyword_combinations.append(terms)
        self.state.failed_keyword_combinations.sort(
            key=lambda item: (len(item), sorted(item))
        )

    def _shrink_and_ban(
        self, keywords: list[str], count_fn: Callable[[list[str]], int] | None
    ) -> str | None:
        """Ban a genuinely zero-lexical-hit AND query as its smallest failing
        word subsets when ``count_fn`` can tell them apart, else ban the whole
        query. Returns what was learned, or ``None`` to signal the caller's
        plain fallback wording (no probe was possible or none was conclusive).
        """
        terms = keyword_terms(keywords)
        if count_fn is None or len(terms) < 2:
            self._ban(terms)
            return None
        try:
            failing, alone = minimal_failing_subsets(terms, count_fn)
        except Exception:
            self._ban(terms)
            return None
        alone_text = ", ".join(f"{word} {count}" for word, count in sorted(alone.items()))
        if failing:
            for subset in failing:
                self._ban(subset)
            sets_text = ", ".join("{" + ", ".join(sorted(subset)) + "}" for subset in failing)
            text = f"Smallest word sets that already match nothing: {sets_text}."
            if alone_text:
                text += f" Tables containing each word alone: {alone_text}."
            return text
        self._ban(terms)
        return f"Each word matches tables alone ({alone_text})." if alone_text else None

    def _search_tool_description(self) -> str:
        if self.retrieval_config.mode.value_keywords:
            # Cell-value retrieval matches row contents only, so the agent must be
            # asked for values the rows store; a dataset topic finds little.
            return (
                "Search for tables whose cells contain the supplied values. Pass "
                "`values` as a list of up to 8 values likely to appear verbatim in "
                "the cells of the needed tables, written the way the data would "
                "store them: category labels, place or entity names, codes, or "
                'years (for example ["Brooklyn", "suspension", "2016-17"]). Each '
                "value is matched whole, so a value with spaces stays one entry. "
                "Only these values are searched: not the question, and not titles "
                "or column names. Exactly one initial search is allowed. Use the "
                "bounded metadata and schema previews to shortlist the strongest "
                "candidates, then verify them with inspect_columns before "
                "selecting tables."
            )
        if self.retrieval_config.mode.verbatim_entities:
            # Entities are matched against table content as written, so the agent
            # names them the way the question does rather than describing a topic.
            return (
                "Search for relevant tables using the retrieval strategy configured "
                "by the experiment. Pass `entities` as a list of the specific, "
                "named, canonical entities the question mentions explicitly that "
                "would typically appear verbatim in a database: identifiers, "
                "symbols, codes, or names, each written exactly as it appears in "
                "the question with its original casing and punctuation, and listed "
                'once (for example ["East River", "P.S. 123"]). Do not pass general '
                "concepts or categories; pass an empty list when the question names "
                "none. The tool applies the original question and the configured "
                "retrieval parameters automatically. Exactly one initial search is "
                "allowed. Use the bounded metadata and schema previews to shortlist "
                "the strongest candidates, then verify them with inspect_columns "
                "before selecting tables."
            )
        return (
            "Search for relevant tables using the retrieval strategy configured by "
            "the experiment. Pass `concepts` as a list of 1-2 concise dataset "
            "concepts in the portal's native language. Keep each multi-word named "
            "entity in one list item. The tool applies the original question and "
            "the configured retrieval "
            "parameters automatically. Exactly one initial search is allowed. Use "
            "the bounded metadata and schema previews to shortlist the strongest "
            "candidates, then verify them with inspect_columns before selecting tables."
        )

    def search_tables(self, concepts_str: str = "") -> str:
        """Search for relevant tables using one or two dataset concepts."""
        return self._search(
            [item.strip() for item in concepts_str.split(" ") if item.strip()]
        )

    def search_keyword_concepts(self, concepts: list[str]) -> str:
        """Search with one or two complete concepts using strict AND.

        Concepts are split into the individual words Solr ANDs together
        (matching WordDelimiterGraphFilter) and deduplicated before the search
        runs, so the banlist keys on what Solr actually matched regardless of
        how the agent grouped or punctuated its concepts.
        """
        listed = concepts or []
        if (
            self.discovery.decompose_preview_search
            and self.llm is not None
            and self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID)
        ):
            return self._decompose_preview_search(listed)
        return self._search(split_keywords(listed))

    def search_table_values(self, values: list[str]) -> str:
        """Search for tables whose cells contain the listed values."""
        # Each value stays whole: "East River" is one search value, not two.
        return self._search([
            " ".join(str(value).split())
            for value in values or []
            if str(value).strip()
        ])

    def search_table_entities(self, entities: list[str]) -> str:
        """Search for tables using the entities the question names verbatim."""
        # Each entity stays whole. An empty list is still a search: the question
        # is ranked as usual, with no content search behind it.
        listed = [
            " ".join(str(entity).split())
            for entity in entities or []
            if str(entity).strip()
        ]
        return self._search(listed, entities=listed)

    def _search_limit_reached(self) -> bool:
        nonempty_attempts = sum(
            bool(attempt.get("current_candidates"))
            for attempt in self.state.search_attempts
        )
        lexical_retry_limit = (
            self.discovery.max_search_attempts
            + self.discovery.max_zero_result_retries
        )
        kw_hybrid_limit_reached = (
            nonempty_attempts >= self.discovery.max_search_attempts
            or len(self.state.search_attempts) >= lexical_retry_limit
        )
        if self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            return kw_hybrid_limit_reached
        return len(self.state.search_attempts) >= self.discovery.max_search_attempts

    def _run_one_retrieval(
        self, keywords: list[str], entities: list[str] | None = None
    ) -> dict:
        """Run one retrieval call, map it onto local files, fold it into the
        accumulated candidate pool (reciprocal-rank fusion with whatever is
        already there), and ban a zero-lexical-hit AND query as its smallest
        failing word subsets. Appends one entry to state.search_attempts.

        Does not check whether a search may run at all (cache, inspection
        lock, attempt limit) -- the caller decides that once, up front, since
        one caller (decompose_preview_search) may need this to run several
        times for what the agent experiences as a single search.
        """
        if self.retrieval_observer is None:
            retriever = get_table_retrieval_service(
                self.solr_client,
                self.retrieval_config,
                *([self.csv_dir] if self.retrieval_config.mode.requires_table_dir else []),
            )
        else:
            retriever = get_table_retrieval_service(
                self.solr_client,
                self.retrieval_config,
                *([self.csv_dir] if self.retrieval_config.mode.requires_table_dir else []),
                observer=self.retrieval_observer,
            )
        # Retrieved candidates must first be mapped and de-duplicated against
        # local files.  Request a wider ranked list here and apply the
        # workflow's final top_k only after that mapping below. Widen by
        # the current ban count too (mirrors DIVIDED's _solr_and_search),
        # so a banned hit shrinks the fetch instead of the final visible
        # pool -- otherwise every accumulated ban would silently starve
        # top_k with nothing backfilling it.
        fetch_k = (
            max(self.discovery.fetch_floor, self.retrieval_config.top_k)
            + len(self.state.excluded_tables)
        )
        retrieval_started = time.monotonic()
        hits = retriever.retrieve(
            question=self.question,
            keywords=keywords,
            top_k=fetch_k,
            lexical_fetch_k=fetch_k,
            q_op="AND",
            entities=entities,
        )
        search_mode = (
            "AND" if self.retrieval_config.mode == RetrievalMode.KEYWORD
            else self.retrieval_config.mode.value
        )

        searched = (
            "the question only"
            if self.retrieval_config.mode.ranks_question_only
            else f"concepts {list(keywords)}"
        )
        self._emit_notice(
            f"\n> \U0001f50e **Retrieval:** `{self.retrieval_config.mode.value}` "
            f"\u00b7 {searched} "
            f"\u00b7 {len(hits)} ranked hits in "
            f"{time.monotonic() - retrieval_started:.1f}s\n"
        )

        self.state.keyword_history.append(keywords)
        attempt = len(self.state.keyword_history)

        current_candidates: list[str] = []
        for hit in hits:
            doc = hit.document
            matched = match_local_csv(doc, self.all_files)
            if (
                matched is None
                or matched in current_candidates
                or matched.casefold() in self.state.excluded_tables
            ):
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
        # Carried candidates from a prior round already proved they
        # satisfy part of the question; re-surface them regardless of
        # whether this round's retrieval finds them again, even if that
        # means displacing the weakest fresh candidate.
        for table in self.state.carried_tables:
            if table.casefold() in self.state.excluded_tables:
                continue
            if table in self.state.all_candidates:
                self.state.all_candidates.remove(table)
            self.state.all_candidates.insert(0, table)
            if table not in self.state.solr_meta and table in self.state.carried_metadata:
                self.state.solr_meta[table] = self.state.carried_metadata[table]
        if len(self.state.all_candidates) > self.retrieval_config.top_k:
            self.state.all_candidates = self.state.all_candidates[
                : self.retrieval_config.top_k
            ]
        candidates = self.state.all_candidates
        self.state.visible_candidate_count = min(
            self.discovery.initial_candidates,
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

        # A strict AND lexical branch can fail to match anything while the
        # overall search still returns candidates in HYBRID mode, rescued by
        # the semantic branch -- that miss must still be shrunk and banned,
        # or a repeat of the same doomed words looks new every time.
        learned = None
        lexical_empty = False
        if self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            lexical_empty = (
                not hits if self.retrieval_config.mode == RetrievalMode.KEYWORD
                else getattr(retriever, "last_lexical_hit_count", None) == 0
            )
            if lexical_empty:
                count_fn = getattr(retriever, "lexical_match_count", None)
                learned = self._shrink_and_ban(keywords, count_fn)
            elif not current_candidates:
                # Solr matched something real for this AND query, just
                # nothing available locally -- ban the whole query as
                # attempted; there is nothing lexically zero to shrink.
                self._ban(keyword_terms(keywords))

        return {
            "attempt": attempt,
            "keywords": keywords,
            "searched": searched,
            "search_mode": search_mode,
            "current_candidates": current_candidates,
            "candidates": candidates,
            "visible_candidates": visible_candidates,
            "lexical_empty": lexical_empty,
            "learned": learned,
        }

    def _format_search_response(self, result: dict) -> str:
        attempt, searched = result["attempt"], result["searched"]
        current_candidates, candidates = result["current_candidates"], result["candidates"]
        visible_candidates = result["visible_candidates"]
        lexical_empty, learned = result["lexical_empty"], result["learned"]

        if not current_candidates:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            self._record_retrieval_memory_event(
                outcome="zero_results",
                terms=list(result["keywords"]),
            )
            return (
                f"Attempt: {attempt}\nSearched: {searched}\n"
                "No tables found. "
                + (f"{learned} " if learned else "")
                + "This AND keyword combination was added to "
                "the zero-result banlist. Search again with a genuinely "
                "different query.\n"
                + (
                    f"Zero-result banlist: {failed}"
                    if self.retrieval_config.mode in (
                        RetrievalMode.KEYWORD,
                        RetrievalMode.HYBRID,
                    )
                    else ""
                )
            )

        prefix = ""
        if lexical_empty:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            prefix = (
                "No table contains all of these words."
                + (f" {learned}" if learned else "")
                + "\n"
                f"Zero-result banlist: {failed}\n"
                "The candidates below do not match every word (found through "
                "the configured semantic branch instead):\n\n"
            )

        return (
            f"Attempt: {attempt}\nSearched: {searched}\n\n"
            + prefix
            + "Candidates in retrieval order after local-file mapping:\n"
            + format_candidate_context(visible_candidates, self.state.solr_meta)
            + (
                f"\n{len(candidates) - len(visible_candidates)} additional "
                "ranked candidates are available through expand_candidates."
                if len(candidates) > len(visible_candidates)
                else ""
            )
        )

    def _search(
        self, supplied_concepts: list[str], entities: list[str] | None = None
    ) -> str:
        """Run the configured retrieval once for the supplied search terms."""
        try:
            keywords = list(supplied_concepts)
            if (
                self.retrieval_config.mode.ranks_question_only
                or self.retrieval_config.mode.verbatim_entities
            ):
                keywords = []
            elif not keywords:
                if self.retrieval_config.mode.value_keywords:
                    return (
                        "No values provided. Search with a list of values likely to "
                        "appear in the cells of the needed tables."
                    )
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
            # Keyed on what the agent supplied, as in every mode. Question-only
            # modes drop the concepts before retrieval, and keying on that empty
            # list told the agent that a different search repeated the first.
            key = self._search_cache_key([*supplied_concepts, *(entities or [])])
            if self.retrieval_config.mode in (
                RetrievalMode.KEYWORD,
                RetrievalMode.HYBRID,
            ):
                failed_subset = self._failed_keyword_subset(keywords)
                if failed_subset is not None:
                    blocked = ", ".join(sorted(failed_subset))
                    return (
                        "Search rejected before retrieval: this AND query contains "
                        f"the known zero-result keyword subset {{{blocked}}}. "
                        "Formulate a different query that does not contain that subset."
                    )
            if key in self.state.search_cache:
                return (
                    "Search skipped: identical concepts were already used. Do not "
                    "repeat this search.\n"
                    + self.state.search_cache[key]
                )
            # inspection_counts, not inspection_cache: a carried-forward
            # table's cache entry is seeded onto inspection_cache before this
            # round's first search, so checking inspection_cache here would
            # block search_tables outright the moment anything was carried
            # over. inspection_counts only grows from an inspect_columns call
            # actually made this round, so it still blocks a genuine
            # search-after-judging within the round.
            if self.state.inspection_counts and not self.discovery.search_after_inspection:
                return (
                    "Search refinement blocked: a candidate has already been "
                    "inspected. Use the existing evidence or one guided expansion."
                )
            if self._search_limit_reached():
                return (
                    f"Search limit reached ({self.discovery.max_search_attempts} "
                    "attempt(s)). Do not call search_tables again; inspect, expand "
                    "if needed, then select."
                )
            self.state.used_keywords = supplied_concepts
            result = self._run_one_retrieval(keywords, entities)
            response = self._format_search_response(result)
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
            return f"Error during table retrieval: {exc}."

    # ---------------------------------------------------- decompose_preview_search

    def _chat_json(self, system: str, user: str, *, stage: str) -> dict:
        """One structured LLM call, with a single syntax-repair retry."""
        response = self.llm.chat([
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ])
        raw = str(response.message.content or "").strip()
        try:
            return _extract_json_object(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            repair = self.llm.chat([
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=(
                    user + f"\n\nYour {stage} response was not one valid JSON "
                    "object. Return ONLY the same intended object as valid "
                    "JSON, no prose, no code fences."
                )),
            ])
            return _extract_json_object(str(repair.message.content or "").strip())

    def _decompose_question(self) -> list[str]:
        """Split the ORIGINAL question into 1-4 standalone per-table search
        requests. Falls back to the question itself on any failure."""
        try:
            obj = self._chat_json(
                _DECOMPOSE_SYSTEM_PROMPT, f"QUESTION: {self.question}", stage="decompose"
            )
        except Exception:
            return [self.question]
        subs: list[str] = []
        for table in obj.get("tables") or []:
            request = str((table or {}).get("request") or "").strip()
            if request and request not in subs:
                subs.append(request)
        return subs[:4] or [self.question]

    def _word_frequency(self, word: str) -> int:
        key = word.casefold()
        if key not in self._word_freq_cache:
            response = self.solr_client.select([word], q_op="AND", rows=0)
            self._word_freq_cache[key] = int(
                response.get("response", {}).get("numFound", 0)
            )
        return self._word_freq_cache[key]

    def _and_count(self, words: Sequence[str]) -> int:
        response = self.solr_client.select(list(words), q_op="AND", rows=0)
        return int(response.get("response", {}).get("numFound", 0))

    def _question_search_words(self, text: str, cap: int = 8) -> list[str]:
        """Content words of `text` that occur somewhere in the index, rarest first."""
        seen, out = set(), []
        for word in re.findall(r"[\w'\u2019-]+", text):
            key = word.casefold().strip("'\u2019-")
            if (
                key
                and key not in _SEARCH_STOPWORDS
                and key not in _AGGREGATION_STOPWORDS
                and key not in seen
            ):
                seen.add(key)
                out.append(word.strip("'\u2019-"))
        present = [word for word in out if self._word_frequency(word) > 0]
        present.sort(key=self._word_frequency)
        return present[:cap]

    def _candidate_word_subsets(
        self, words: list[str], *, lo: int = 1, hi: int = 60, cap: int = 8, workers: int = 6
    ) -> list[tuple[list[str], int]]:
        """2-word AND combinations whose match count fits the pool, rarest
        first. Capped to pairs and a small word universe (unlike the lab
        version's 2-3 word / 13-word sweep) since each combination costs one
        live Solr round trip and this runs inline in an interactive tool call.
        """
        combos = list(itertools.combinations(words, 2))
        if not combos:
            return []

        def probe(combo: tuple[str, str]) -> tuple[tuple[str, str], int]:
            return combo, self._and_count(combo)

        out: list[tuple[list[str], int]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for combo, matched in executor.map(probe, combos):
                if lo <= matched <= hi:
                    out.append((list(combo), matched))
        # Same relative order as ranking by summed IDF: for same-size subsets,
        # log(N) per word is a constant, so ranking by ascending summed
        # log(frequency) is equivalent to descending summed IDF, without
        # needing the corpus size N at all.
        out.sort(key=lambda item: sum(
            math.log(max(1, self._word_frequency(word))) for word in item[0]
        ))
        return out[:cap]

    def _preview_candidate_words(self, words: list[str]) -> str:
        response = self.solr_client.select(
            words, q_op="AND", rows=3, fl="title,publisher,description"
        )
        docs = response.get("response", {}).get("docs", [])
        items = []
        for doc in docs:
            description = re.sub(
                r"\s+", " ", re.sub(r"<[^>]+>", " ", doc.get("description") or "")
            ).strip()[:70]
            items.append(f"{(doc.get('publisher') or '')[:22]} | {doc.get('title') or ''} | {description}")
        return " ;; ".join(items)

    def _pick_candidate_word_sets(self, sub_question: str) -> list[list[str]]:
        """1-3 AND-word sets to actually search for one sub-question, chosen
        by the model from a preview of real candidates (match count + top
        matches), not guessed blind."""
        words = self._question_search_words(sub_question)
        if len(words) < 2:
            return [words] if words else []
        subsets = self._candidate_word_subsets(words)
        if not subsets:
            return [words[:3]]
        try:
            with ThreadPoolExecutor(max_workers=6) as executor:
                previews = list(executor.map(
                    lambda item: self._preview_candidate_words(item[0]), subsets
                ))
            listing = "\n".join(
                f"{i}. {json.dumps(words_)}  matches {n}: {preview}"
                for i, ((words_, n), preview) in enumerate(zip(subsets, previews), 1)
            )
            obj = self._chat_json(
                _CANDIDATE_PICK_SYSTEM_PROMPT,
                f"QUESTION: {sub_question}\n\nCANDIDATE SEARCHES:\n{listing}",
                stage="candidate pick",
            )
        except Exception:
            return [subsets[0][0]]
        picks = [
            int(pick) for pick in (obj.get("picks") or [])
            if str(pick).isdigit() and 1 <= int(pick) <= len(subsets)
        ][:3]
        chosen = [subsets[pick - 1][0] for pick in picks]
        return chosen or [subsets[0][0]]

    def _format_decompose_response(self, sub_questions: list[str], results: list[dict]) -> str:
        candidates = self.state.all_candidates
        visible_candidates = candidates[: self.state.visible_candidate_count]
        attempts = [result["attempt"] for result in results]
        searched = "; ".join(f"table {i}: {sub}" for i, sub in enumerate(sub_questions, 1))
        any_candidates = any(result["current_candidates"] for result in results)
        all_lexical_empty = all(result["lexical_empty"] for result in results)
        learned_notes = " ".join(
            result["learned"] for result in results if result["learned"]
        )

        if not any_candidates:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            self._record_retrieval_memory_event(
                outcome="zero_results",
                terms=[word for result in results for word in result["keywords"]],
            )
            return (
                f"Attempt: {attempts[0]}-{attempts[-1]}\n"
                f"Searched (split by table): {searched}\n"
                "No tables found for any of the split searches. "
                + (f"{learned_notes} " if learned_notes else "")
                + "These AND combinations were added to the zero-result "
                "banlist. Search again with a genuinely different query.\n"
                + (
                    f"Zero-result banlist: {failed}"
                    if self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID)
                    else ""
                )
            )

        prefix = ""
        if all_lexical_empty:
            failed = [sorted(item) for item in self.state.failed_keyword_combinations]
            prefix = (
                "No table contains all of the words in any split search."
                + (f" {learned_notes}" if learned_notes else "")
                + "\n"
                f"Zero-result banlist: {failed}\n"
                "The candidates below do not match every word (found through "
                "the configured semantic branch instead):\n\n"
            )

        return (
            f"Attempt: {attempts[0]}-{attempts[-1]}\n"
            f"Searched (split by table): {searched}\n\n"
            + prefix
            + "Candidates in retrieval order after local-file mapping:\n"
            + format_candidate_context(visible_candidates, self.state.solr_meta)
            + (
                f"\n{len(candidates) - len(visible_candidates)} additional "
                "ranked candidates are available through expand_candidates."
                if len(candidates) > len(visible_candidates)
                else ""
            )
        )

    def _decompose_preview_search(self, concepts: list[str]) -> str:
        """decompose_tuned_preview_noacronym, ported from
        experiments/retrieval_lab/arms_retrieval.py: split the ORIGINAL
        question into its per-table distinguishing-detail search phrases,
        then for each, preview a few real AND-word candidates (with match
        counts and top matches) and let the model pick before any of them
        actually runs. The agent's own `concepts` still gate caching and the
        zero-result banlist exactly as a plain search would, so the tool's
        contract to the agent is unchanged; only what actually gets searched
        gets smarter.
        """
        try:
            key = self._search_cache_key(concepts)
            if self.retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
                failed_subset = self._failed_keyword_subset(split_keywords(concepts))
                if failed_subset is not None:
                    blocked = ", ".join(sorted(failed_subset))
                    return (
                        "Search rejected before retrieval: this AND query contains "
                        f"the known zero-result keyword subset {{{blocked}}}. "
                        "Formulate a different query that does not contain that subset."
                    )
            if key in self.state.search_cache:
                return (
                    "Search skipped: identical concepts were already used. Do not "
                    "repeat this search.\n" + self.state.search_cache[key]
                )
            if self.state.inspection_counts and not self.discovery.search_after_inspection:
                return (
                    "Search refinement blocked: a candidate has already been "
                    "inspected. Use the existing evidence or one guided expansion."
                )
            if self._search_limit_reached():
                return (
                    f"Search limit reached ({self.discovery.max_search_attempts} "
                    "attempt(s)). Do not call search_tables again; inspect, expand "
                    "if needed, then select."
                )

            self.state.used_keywords = concepts
            sub_questions = self._decompose_question()

            results = []
            for sub_question in sub_questions:
                for words in self._pick_candidate_word_sets(sub_question):
                    if words:
                        results.append(self._run_one_retrieval(words))

            if not results:
                # Decompose/pick produced nothing usable: fall back to the
                # plain single-shot search over the agent's own concepts.
                return self._search(split_keywords(concepts))

            response = self._format_decompose_response(sub_questions, results)
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
            return f"Error during table retrieval: {exc}."

    def inspect_columns(
        self,
        file_name: str | None = None,
        filename: str | None = None,
        candidate_number: int | None = None,
    ) -> str:
        """
        Returns a compact profile for one table in the active dataset.
        Shows row count, bounded min/max coverage for temporal columns, column
        types, and sample values for low-cardinality categorical columns. At
        most two requests per file are useful; repeated requests use a cache.
        If the question has a date or time range, compare it with the reported
        temporal coverage before selecting the table.
        Use this only after identifying a valid table file with search_tables.
        Normally inspect the 2-4 strongest candidates from the bounded metadata
        preview instead of inspecting every retrieved table.

        Prefer candidate_number -- the "Candidate N" label search_tables and
        expand_candidates already print above each entry -- over file_name or
        filename. Generated filenames are long and easy to mistype; a small
        integer has nothing to transcribe incorrectly.
        """
        visible_candidates = self.state.all_candidates[
            : self.state.visible_candidate_count
        ]
        if candidate_number is not None:
            if not (1 <= candidate_number <= len(visible_candidates)):
                return (
                    f"Error: candidate_number must be between 1 and "
                    f"{len(visible_candidates)} (the currently visible candidates)."
                )
            name = visible_candidates[candidate_number - 1]
        else:
            name = file_name or filename
            if not name:
                return (
                    "Error: candidate_number (preferred) or file_name/filename "
                    "is required."
                )
            name = name.strip()
        key = name.casefold()
        if name not in visible_candidates:
            return (
                f"Error: {name} is not currently visible. Inspect only candidates "
                "already shown by search_tables or expand_candidates."
            )
        attempted_candidates = len(self.state.inspection_counts)
        if key not in self.state.inspection_cache:
            current_limit = (
                self.discovery.initial_shortlist_size
                if self.state.expansion_count == 0
                else self.discovery.max_inspected_candidates
            )
            if attempted_candidates >= current_limit:
                if (
                    self.state.expansion_count == 0
                    and self.state.visible_candidate_count < len(self.state.all_candidates)
                ):
                    return (
                        "Inspection blocked: the initial shortlist is limited to "
                        f"{self.discovery.initial_shortlist_size} "
                        "candidates. If coverage is incomplete, call expand_candidates "
                        "before inspecting another candidate."
                    )
                return (
                    "Inspection blocked: at most "
                    f"{self.discovery.max_inspected_candidates} distinct candidates "
                    "may be inspected for this request."
                )
        count = self.state.inspection_counts.get(key, 0) + 1
        self.state.inspection_counts[key] = count
        if count > self.discovery.max_inspections_per_file:
            return (
                f"Inspection skipped: {name} has already been inspected "
                f"{self.discovery.max_inspections_per_file} times. Use the cached schema and "
                "continue with confirm_unified_selection."
            )
        if key not in self.state.inspection_cache:
            self.state.inspection_cache[key] = _inspect_columns(self.csv_dir, name)
            return self.state.inspection_cache[key]
        if count == 1:
            # First time this schema is actually shown this round -- whether
            # freshly cached above or carried over from a prior round -- so
            # the full profile earns its place in context.
            return (
                f"Cached inspection (attempt {count}/"
                f"{self.discovery.max_inspections_per_file}):\n"
                + self.state.inspection_cache[key]
            )
        # Already shown once this round: point back instead of re-sending a
        # profile the agent has already seen, to keep context lean.
        candidate_label = f"Candidate {visible_candidates.index(name) + 1}"
        return (
            f"Cached inspection (attempt {count}/"
            f"{self.discovery.max_inspections_per_file}): already shown above "
            f"for {candidate_label}. Reuse that schema; do not request it again."
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
        if self.state.expansion_count >= self.discovery.max_expansions:
            return (
                "Expansion limit reached. Do not call expand_candidates again "
                "or search_tables again; select among the inspected candidates."
            )
        if self.state.visible_candidate_count >= len(self.state.all_candidates):
            return "No additional candidates are available."
        start = self.state.visible_candidate_count
        hidden = self.state.all_candidates[start:]
        newly_visible = _rank_for_missing_requirements(
            hidden, self.state.solr_meta, requirements, self.discovery.expansion_size
        )
        if not newly_visible:
            self.state.expansion_count += 1
            self.state.expansion_requirements = requirements
            return (
                "No hidden candidate has metadata matching the missing requirements. "
                "Do not call expand_candidates or search_tables again; select or "
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
            + format_candidate_context(
                newly_visible, self.state.solr_meta, start_rank=start + 1
            )
            + f"\n\n{next_step}"
        )

    def check_join_union(self, file_name_1: str, file_name_2: str) -> str:
        """
        Check whether two tables join or union with each other. Reports the join
        key columns when they join, the aligned columns when they union, or that
        they neither join nor union.
        """
        return _check_join_union(self.csv_dir, file_name_1, file_name_2)

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

        requirement_coverage = _normalize_requirement_coverage(
            requirement_coverage or {}, normalized_tables
        )
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
                if not semantic_plan.get("output_columns"):
                    semantic_plan["output_columns"] = _derive_output_columns(semantic_plan)
            try:
                semantic_plan = SemanticAnalysisPlan.model_validate(semantic_plan).model_dump()
            except Exception as exc:
                self.state.plan_failure = str(exc)
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

        normalized_requirements = dict(requirements or {})
        coder_brief = self._build_coder_brief(
            normalized_tables,
            requirement_coverage,
            normalized_requirements,
            combination_strategy,
            table_roles,
            semantic_plan if isinstance(semantic_plan, dict) else None,
        )
        requirement_ledger = self._build_requirement_ledger(
            requirement_coverage,
            normalized_requirements,
            uncovered_requirements,
            semantic_plan if isinstance(semantic_plan, dict) else None,
        )
        ledger_blockers = requirement_ledger_blockers(
            requirement_ledger, normalized_tables
        )
        if ledger_blockers:
            raise ValueError(
                "Selection blocked: fundamental data requirements lack concrete "
                "selected-table/column evidence: " + ", ".join(ledger_blockers) + ". "
                "Inspect or expand candidates and bind them in requirement_coverage. "
                "Keep calculations in the ledger as computational."
            )
        self.state.selection_plan = {
            "requirement_coverage": requirement_coverage,
            "table_roles": table_roles,
            "combination_strategy": combination_strategy,
            "uncovered_requirements": uncovered_requirements,
            "alternatives_rejected": alternatives_rejected,
            "requirements": normalized_requirements,
            "requirement_ledger": requirement_ledger,
            "coder_brief": coder_brief,
            **({"semantic_plan": semantic_plan} if contract_first else {}),
        }
        self.state.plan_failure = None
        self.state.selection_plan_source = (
            "confirm_unified_selection" if contract_first else "selection_only"
        )
        self.state.confirmed_tables = list(normalized_tables)
        self.state.selection_reasoning = reasoning
        self.state.selection_requirements = normalized_requirements
        self.state.selection_advisories = advisories

        final_tables = ", ".join(normalized_tables)

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning,
            "selection_plan": self.state.selection_plan,
            "advisories": advisories,
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    def reject_unified_selection(
        self, reasoning: str, suggestion: str, ban_tables: dict[str, str] | None = None
    ) -> str:
        """
        Use this tool when the candidates cannot yet fully cover the question's
        essential requirements. Map every already-inspected candidate you
        judge highly irrelevant to the concrete evidence proving that in
        ban_tables, so it is excluded from later retrieval for this question.
        Every other inspected candidate is kept and carries over to the next
        attempt, even if it does not cover every requirement.
        Calling this tool means you have finished this attempt.
        """
        inspected = self.state.inspected_candidates()
        by_fold = {table.casefold(): table for table in inspected}
        unjustified = [
            table for table, justification in (ban_tables or {}).items()
            if table.casefold() in by_fold
            and len(str(justification).strip()) < MIN_BAN_JUSTIFICATION_CHARS
        ]
        if unjustified:
            raise ValueError(
                "Selection blocked: ban_tables must map each highly-irrelevant "
                "candidate to the concrete evidence proving it satisfies "
                "nothing, not a bare name. Missing or too-short justification "
                f"for: {', '.join(unjustified)}. Give it a real justification "
                "or drop it from ban_tables."
            )
        banned = [
            by_fold[table.casefold()]
            for table in (ban_tables or {})
            if table.casefold() in by_fold
        ]
        self.state.rejection_skip_tables = banned
        self.state.rejection_keep_tables = [
            table for table in inspected if table not in banned
        ]
        self._record_retrieval_memory_event(
            outcome="insufficient_coverage",
            terms=list(self.state.used_keywords),
            tables=list(inspected),
            reason=reasoning,
            suggestion=suggestion,
            evidence=dict(ban_tables or {}),
        )
        return f"REJECT_KEYWORDS: {reasoning}\nSuggestion: {suggestion}"

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
            # Prefer a globally unique exact spelling before trying harmless
            # aliases. Trip_distance and trip_distance can belong to different
            # yearly tables and must not become an ambiguous folded match.
            exact = [(table, str(raw_column)) for table, names in schemas.items()
                     if raw_column in names]
            if exact:
                return exact[0] if len(exact) == 1 else None
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
        for kind in ("filters", "temporal_filters", "dimensions", "measures"):
            for binding in brief[kind]:
                if not isinstance(binding, dict):
                    continue
                columns = binding.get("columns") or [binding.get("column")]
                columns = [column for column in columns if column]
                table = str(binding.get("table") or "")
                if not table and columns:
                    matches = [resolve_unique(column) for column in columns]
                    if all(matches) and len({match[0] for match in matches}) == 1:
                        table = matches[0][0]
                        binding["table"] = table
                if table:
                    resolved = [resolve(table, column) for column in columns]
                    if columns and all(resolved):
                        if "column" in binding:
                            binding["column"] = resolved[0]
                        if "columns" in binding:
                            binding["columns"] = resolved
                        for column in resolved:
                            if column not in selected_columns[table]:
                                selected_columns[table].append(column)
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

    def _build_requirement_ledger(
        self,
        coverage: dict[str, dict[str, object]],
        requirements: dict[str, object],
        uncovered: list[str],
        semantic_plan: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        """Create one compact mode-independent, non-prescriptive checklist."""
        return build_requirement_ledger(
            self.question, coverage, requirements, uncovered, semantic_plan
        )

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
            self.state.plan_failure = str(exc)
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

    def _search_tool(self) -> FunctionTool:
        if self.retrieval_config.mode.verbatim_entities:
            # Entities are matched whole, so the agent passes a real list, as for
            # cell values. The tool keeps its name for prompts and activity logs.
            return FunctionTool.from_defaults(
                fn=self.search_table_entities,
                name="search_tables",
                description=self._search_tool_description(),
            )
        if self.retrieval_config.mode.value_keywords:
            # Values are searched whole, so the agent passes a real list; splitting
            # a string on spaces would turn "East River" into two searches. The
            # tool keeps its name, so prompts and activity logs are unchanged.
            return FunctionTool.from_defaults(
                fn=self.search_table_values,
                name="search_tables",
                description=self._search_tool_description(),
            )
        return FunctionTool.from_defaults(
            fn=self.search_keyword_concepts,
            name="search_tables",
            description=self._search_tool_description(),
        )

    def is_tool_available(self, tool_name: str) -> bool:
        """Report whether a tool is still worth offering to the model.

        Every tool already refuses gracefully once spent, but that refusal
        still costs a full model turn and one of the run's capped tool calls.
        This mirrors each tool's own exhaustion checks -- never a precondition
        the tool just hasn't met *yet* -- so the pruning agent can drop a
        genuinely spent tool from a turn's offering instead of paying for a
        refusal that only repeats what the last one already said. A tool
        blocked only by something else that can still change (an expansion
        that hasn't run, a carried schema not yet served) stays offered.
        """
        if tool_name == "search_tables":
            mode = self.retrieval_config.mode
            if (
                mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID)
                and self.state.semantic_failure is not None
            ):
                return False
            # inspection_counts, not inspection_cache: see _search's own note --
            # a carried-forward schema must not look like an in-round inspection.
            if self.state.inspection_counts and not self.discovery.search_after_inspection:
                return False
            nonempty_attempts = sum(
                bool(attempt.get("current_candidates"))
                for attempt in self.state.search_attempts
            )
            lexical_retry_limit = (
                self.discovery.max_search_attempts + self.discovery.max_zero_result_retries
            )
            search_limit_reached = (
                nonempty_attempts >= self.discovery.max_search_attempts
                or len(self.state.search_attempts) >= lexical_retry_limit
            )
            if mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
                return not search_limit_reached
            return len(self.state.search_attempts) < self.discovery.max_search_attempts

        if tool_name == "expand_candidates":
            # Only the spendable resource -- the expansion budget -- withdraws
            # this tool. "Nothing to inspect yet" and "nothing hidden right now"
            # are the tool's own, still-changeable refusals, not exhaustion.
            return self.state.expansion_count < self.discovery.max_expansions

        if tool_name == "inspect_columns":
            visible_keys = {
                candidate.casefold()
                for candidate in self.state.all_candidates[: self.state.visible_candidate_count]
            }
            attempted_candidates = len(self.state.inspection_counts)
            current_limit = (
                self.discovery.initial_shortlist_size
                if self.state.expansion_count == 0
                else self.discovery.max_inspected_candidates
            )
            if attempted_candidates < current_limit:
                return True
            # A carried schema not yet served this round bypasses the limit in
            # inspect_columns itself, so it must not look spent here either.
            has_unserved_carried = any(
                key in self.state.inspection_cache and key not in self.state.inspection_counts
                for key in visible_keys
            )
            if has_unserved_carried:
                return True
            return (
                self.state.expansion_count == 0
                and self.is_tool_available("expand_candidates")
                and self.state.visible_candidate_count < len(self.state.all_candidates)
            )

        return True

    def get_tools(self) -> list[FunctionTool]:
        return [
            self._search_tool(),
            FunctionTool.from_defaults(fn=self.inspect_columns),
            FunctionTool.from_defaults(fn=self.expand_candidates),
            FunctionTool.from_defaults(fn=self.check_join_union),
            FunctionTool.from_defaults(fn=self.confirm_unified_selection, fn_schema=ConfirmUnifiedSelectionSchema, return_direct=True),
            FunctionTool.from_defaults(fn=self.reject_unified_selection, fn_schema=RejectUnifiedSelectionSchema, return_direct=True),
        ]


def make_p12_tools(
    state: P12State,
    solr_client: LocalSolrClient,
    all_files: list[str],
    csv_dir: Path,
    question: str = "",
    retrieval_config: RetrievalConfig | None = None,
    retrieval_observer: Callable[[RetrievalRun], None] | None = None,
    discovery_config: DiscoveryConfig | None = None,
    notice_callback: StreamCallback | None = None,
    llm: LLM | None = None,
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
        discovery_config=discovery_config,
        notice_callback=notice_callback,
        llm=llm,
    )
    return manager.get_tools()
