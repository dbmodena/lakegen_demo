import json
import re
from functools import lru_cache
import pandas as pd
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

from lakegen.core.table_io import iter_table_chunks, read_table, table_row_count
from lakegen.core.types import SolrMetadata
from lakegen.phases.utils import format_candidate_context
from lakegen.agent_tools.requirement_ledger import (
    build_requirement_ledger,
    requirement_ledger_blockers,
)
from lakegen.agent_tools.schema_matching import (
    SM_MACRO_AVG_THRESHOLD,
    SM_MICRO_AVG_THRESHOLD,
    join_evidence,
    union_evidence,
    verify_pair_schema,
)

# ==========================================
# TOOLS
# ==========================================
MAX_TOOL_OUTPUT_CHARS = 4000
MAX_SCHEMA_SAMPLE_ROWS = 500
MAX_SCHEMA_COLUMNS = 80
MAX_UNIQUE_VALUES = 8
MAX_PREVIEW_COLUMNS = 20
MAX_TEMPORAL_PROFILE_COLUMNS = 4
PROFILE_CHUNK_ROWS = 100_000
MAX_TEMPORAL_PROFILE_ROWS = 500_000
MAX_INSPECTIONS_PER_FILE = 2

_TEMPORAL_COLUMN_PATTERN = re.compile(
    r"(^|_)(date|datetime|timestamp|time|year)($|_)",
    re.IGNORECASE,
)
_QUESTION_YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_COVERAGE_RANGE_PATTERN = re.compile(
    r"^-[^:]+:\s*((?:19|20)\d{2})(?:-\d{2}-\d{2})?\s+to\s+"
    r"((?:19|20)\d{2})(?:-\d{2}-\d{2})?",
    re.MULTILINE,
)


def _temporal_coverage_issue(
    question: str,
    tables: list[str],
    inspection_cache: dict[str, str],
) -> str | None:
    """Return an issue only when measured coverage proves insufficiency."""
    requested = sorted(
        {int(value) for value in _QUESTION_YEAR_PATTERN.findall(question)}
    )
    if not requested:
        return None

    ranges: list[tuple[int, int]] = []
    for table in tables:
        inspection = inspection_cache.get(table.casefold(), "")
        ranges.extend(
            (int(start), int(end))
            for start, end in _COVERAGE_RANGE_PATTERN.findall(inspection)
        )
    # Snapshot tables can encode their fiscal year only in authoritative
    # metadata. Absence of a measurable time column is therefore inconclusive.
    if not ranges:
        return None

    missing = [
        year
        for year in requested
        if not any(start <= year <= end for start, end in ranges)
    ]
    if not missing:
        return None
    measured = ", ".join(f"{start}-{end}" for start, end in ranges)
    return (
        f"requested year(s) {missing} are outside the inspected temporal "
        f"coverage ({measured})"
    )


class ConfirmSelectionSchema(BaseModel):
    reasoning: str = Field(description="MANDATORY. Write a brief explanation IN ENGLISH explaining why these specific tables were selected and how they answer the question. Do NOT use quotes, apostrophes, or special characters.")
    tables: list[str] = Field(description="A list of the exact file names needed (e.g., ['2016.parquet']). Do not omit any table you need!")
    requirement_coverage: dict[str, dict[str, object]] = Field(default_factory=dict)
    table_roles: dict[str, str] = Field(default_factory=dict)
    combination_strategy: Literal[
        "single_table", "join", "concat_partitions", "aggregate_separately",
        "lookup", "compare",
    ] = "single_table"
    uncovered_requirements: list[str] = Field(default_factory=list)
    requirements: dict[str, object] = Field(default_factory=dict)
    semantic_plan: dict[str, object] | None = None

class RejectSelectionSchema(BaseModel):
    reasoning: str = Field(description="Explain step-by-step why the current tables are not good.")
    suggestion: str = Field(
        description="Suggest dataset concepts, not analytical operations or row-filter values."
    )
    keep_tables: list[str] = Field(
        default_factory=list,
        description=(
            "Already-inspected candidates that satisfy at least one essential "
            "requirement and should carry over to the next attempt, even though "
            "the full set is incomplete."
        ),
    )


class RejectSelectionValueSearchSchema(BaseModel):
    reasoning: str = Field(description="Explain step-by-step why the current tables are not good.")
    suggestion: str = Field(
        description=(
            "Suggest values likely to appear in the rows of the missing table, "
            "not analytical operations or dataset topics."
        )
    )
    keep_tables: list[str] = Field(
        default_factory=list,
        description=(
            "Already-inspected candidates that satisfy at least one essential "
            "requirement and should carry over to the next attempt, even though "
            "the full set is incomplete."
        ),
    )


def _compact_tool_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = str(text).strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _compact_value(value, max_chars: int = 40) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def _table_path(table_dir: Path, file_name: str) -> Path:
    return Path(table_dir) / file_name.strip()


def _temporal_profile(path: Path, columns: list[str]) -> tuple[str, list[str]]:
    """Return row count and bounded coverage for likely temporal columns."""
    temporal_columns = [
        col for col in columns if _TEMPORAL_COLUMN_PATTERN.search(str(col))
    ][:MAX_TEMPORAL_PROFILE_COLUMNS]
    exact_rows = table_row_count(path)
    if not temporal_columns and exact_rows is not None:
        return f"{exact_rows:,}", []

    usecols = temporal_columns or columns[:1]
    profiled_rows = 0
    stats = {
        col: {"valid": 0, "min": None, "max": None}
        for col in temporal_columns
    }

    for chunk in iter_table_chunks(
        path,
        columns=usecols,
        chunk_rows=PROFILE_CHUNK_ROWS,
    ):
        remaining = MAX_TEMPORAL_PROFILE_ROWS - profiled_rows
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk.head(remaining)
        profiled_rows += len(chunk)
        for col in temporal_columns:
            series = chunk[col]
            col_stats = stats[col]

            if re.search(r"(^|_)year($|_)", str(col), re.IGNORECASE):
                parsed = pd.to_numeric(series, errors="coerce")
                parsed = parsed[(parsed >= 1000) & (parsed <= 3000)]
            else:
                parsed = pd.to_datetime(
                    series,
                    errors="coerce",
                    utc=True,
                    format="mixed",
                )

            parsed = parsed.dropna()
            if parsed.empty:
                continue
            col_stats["valid"] += len(parsed)
            chunk_min = parsed.min()
            chunk_max = parsed.max()
            if col_stats["min"] is None or chunk_min < col_stats["min"]:
                col_stats["min"] = chunk_min
            if col_stats["max"] is None or chunk_max > col_stats["max"]:
                col_stats["max"] = chunk_max
        if profiled_rows >= MAX_TEMPORAL_PROFILE_ROWS:
            break

    coverage_lines = []
    for col, col_stats in stats.items():
        if not col_stats["valid"]:
            continue
        unavailable_pct = (
            (profiled_rows - col_stats["valid"]) / profiled_rows * 100
            if profiled_rows
            else 0.0
        )
        min_value = col_stats["min"]
        max_value = col_stats["max"]
        if isinstance(min_value, pd.Timestamp):
            min_value = min_value.strftime("%Y-%m-%d")
            max_value = max_value.strftime("%Y-%m-%d")
        coverage_lines.append(
            f"- {col}: {min_value} to {max_value} "
            f"(missing/unparseable {unavailable_pct:.1f}%)"
        )

    if exact_rows is not None:
        row_label = f"{exact_rows:,}"
    elif profiled_rows >= MAX_TEMPORAL_PROFILE_ROWS:
        row_label = f"at least {profiled_rows:,}"
    else:
        row_label = f"{profiled_rows:,}"
    if temporal_columns and (
        exact_rows is None or profiled_rows < exact_rows
    ):
        coverage_lines.insert(
            0, f"- sampled first {profiled_rows:,} rows (bounded profile)"
        )
    return row_label, coverage_lines


@lru_cache(maxsize=256)
def _inspect_columns_cached(
    table_dir: str,
    file_name: str,
    _file_size: int,
    _modified_ns: int,
) -> str:
    """Inspect one immutable file version and reuse the result across workflows."""
    path = _table_path(Path(table_dir), file_name)
    if not path.exists():
        return f"Error: File missing in active dataset: {file_name}"

    try:
        df = read_table(path, nrows=MAX_SCHEMA_SAMPLE_ROWS)
        schema_info = []
        columns = list(df.columns)
        row_label, temporal_coverage = _temporal_profile(path, columns)

        for col in columns[:MAX_SCHEMA_COLUMNS]:
            dtype = str(df[col].dtype)
            if (
                pd.api.types.is_string_dtype(df[col].dtype)
                or isinstance(df[col].dtype, pd.CategoricalDtype)
            ):
                unique_vals = df[col].dropna().astype(str).unique().tolist()
                if 0 < len(unique_vals) <= MAX_UNIQUE_VALUES:
                    values = [_compact_value(value) for value in unique_vals]
                    schema_info.append(f"- {col} (Category sample): {values}")
                    continue
            schema_info.append(f"- {col} ({dtype})")

        if len(columns) > MAX_SCHEMA_COLUMNS:
            schema_info.append(
                f"- ... {len(columns) - MAX_SCHEMA_COLUMNS} more columns omitted"
            )

        header_lines = [f"Schema for {file_name}:", f"Rows: {row_label}"]
        if temporal_coverage:
            header_lines.extend(["Temporal coverage:", *temporal_coverage])
        header_lines.append(
            f"Columns (types and categories sampled from first {MAX_SCHEMA_SAMPLE_ROWS} rows):"
        )
        output = "\n".join(header_lines + schema_info)
        return _compact_tool_output(output)
    except Exception as e:
        return f"Error: {str(e)}"


def _inspect_columns(table_dir: Path, file_name: str) -> str:
    path = _table_path(table_dir, file_name)
    if not path.exists():
        return f"Error: File missing in active dataset: {file_name}"
    try:
        stat = path.stat()
    except OSError as exc:
        return f"Error: {exc}"
    return _inspect_columns_cached(
        str(Path(table_dir).resolve()),
        file_name.strip(),
        stat.st_size,
        stat.st_mtime_ns,
    )


def _preview_data(table_dir: Path, file_name: str, n_rows: int = 3) -> str:
    path = _table_path(table_dir, file_name)
    if not path.exists():
        return f"Error: File missing in active dataset: {file_name}"

    try:
        n_rows = int(n_rows)
    except (TypeError, ValueError):
        n_rows = 3
    n_rows = max(1, min(n_rows, 5))
    try:
        df = read_table(path, nrows=n_rows)
        omitted = ""
        if len(df.columns) > MAX_PREVIEW_COLUMNS:
            omitted = f"\n... {len(df.columns) - MAX_PREVIEW_COLUMNS} more columns omitted"
            df = df.iloc[:, :MAX_PREVIEW_COLUMNS]
        output = f"Preview of {file_name}:\n{df.to_string(index=False)}{omitted}"
        return _compact_tool_output(output, max_chars=3000)
    except Exception as e:
        return f"Error: {str(e)}"


def _format_join_union(
    file_name_1: str,
    file_name_2: str,
    q_columns: list[str],
    evidence: dict[str, object],
    table_shapes: tuple[tuple[int, int], tuple[int, int]],
) -> str:
    """State whether two tables join or union with each other, never mere schema overlap."""
    left, right = f"'{file_name_1}'", f"'{file_name_2}'"
    average, best = evidence["sm_macro_avg"], evidence["sm_micro_avg"]
    sizes = "; ".join(
        f"{name} {rows:,} rows x {columns} columns"
        for name, (rows, columns) in zip((left, right), table_shapes)
    )
    lines = [
        f"Join/union check between {left} and {right} "
        f"(Valentine over all rows and columns: {sizes})."
    ]

    # The average never exceeds the best score, so OrQa's pair gate passes
    # exactly when the best column pair supports a join.
    join = join_evidence(evidence)
    if not join["supported"]:
        lines.append(
            f"NO RELATIONSHIP: {left} neither joins nor unions with {right} "
            f"(best match score {best:.3f} < {SM_MICRO_AVG_THRESHOLD})."
        )
        return "\n".join(lines)

    def key_pair(q_col: str, r_col: str) -> str:
        return f"{q_col} ({file_name_1}) = {r_col} ({file_name_2})"

    q_key, r_key = join["key"]
    lines.append(
        f"JOIN: {left} joins {right} on {key_pair(q_key, r_key)} "
        f"(match score {join['score']:.3f} >= {SM_MICRO_AVG_THRESHOLD})."
    )
    if join["alternatives"]:
        lines.append(
            "  Alternative join keys: "
            + "; ".join(
                f"{key_pair(q_col, r_col)} ({score:.3f})"
                for q_col, r_col, score in join["alternatives"]
            )
        )

    union = union_evidence(evidence, q_columns)
    if union["supported"]:
        aligned = ", ".join(
            f"{q_col} -> {r_col} ({score:.3f})"
            for q_col, r_col, score in zip(
                union["q_columns"], union["r_columns"], union["column_scores"]
            )
        )
        lines.append(
            f"UNION: {left} unions with {right}, aligning {left} -> {right} columns: "
            f"{aligned}; {len(set(union['q_columns']))} of {len(q_columns)} columns of "
            f"{left} aligned (average match score {average:.3f} >= {SM_MACRO_AVG_THRESHOLD})."
        )
    else:
        lines.append(
            f"UNION: {left} does not union with {right} "
            f"(average match score {average:.3f} < {SM_MACRO_AVG_THRESHOLD})."
        )
    return "\n".join(lines)


def _check_join_union(table_dir: Path, file_name_1: str, file_name_2: str) -> str:
    """Decide with OrQa's Valentine criteria whether two tables join or union."""
    try:
        frames = [
            read_table(_table_path(table_dir, file_name))
            for file_name in (file_name_1, file_name_2)
        ]
        evidence = verify_pair_schema(*frames)
    except Exception as e:
        return f"Error checking join/union between '{file_name_1}' and '{file_name_2}': {e}. Try different tables."

    return _compact_tool_output(
        _format_join_union(
            file_name_1,
            file_name_2,
            list(frames[0].columns),
            evidence,
            (frames[0].shape, frames[1].shape),
        )
    )


_EXPANSION_STOPWORDS = {
    "a", "an", "and", "by", "for", "from", "in", "of", "or", "the", "to",
    "con", "da", "del", "della", "di", "e", "il", "in", "la", "per", "un",
    "una",
}


def _requirement_terms(value: str) -> list[str]:
    """Extract a small, stable set of explicit coverage requirements."""

    terms = re.findall(r"[a-z0-9]+", str(value).casefold())
    return list(dict.fromkeys(
        term for term in terms
        if len(term) > 1 and term not in _EXPANSION_STOPWORDS
    ))[:12]


def _rank_for_missing_requirements(
    candidates: list[str],
    metadata: SolrMetadata,
    requirements: list[str],
    limit: int,
) -> list[str]:
    """Rank hidden candidates using only already-retrieved metadata."""

    scored: list[tuple[int, int, str]] = []
    for original_index, candidate in enumerate(candidates):
        searchable = json.dumps(
            {"file": candidate, "metadata": metadata.get(candidate, {})},
            ensure_ascii=False,
            default=str,
        ).casefold()
        score = sum(1 for requirement in requirements if requirement in searchable)
        if score:
            scored.append((-score, original_index, candidate))
    scored.sort()
    return [candidate for _score, _index, candidate in scored[:limit]]


class Phase2JudgeToolsManager:
    """Manager for Phase 2 judge tools to avoid closures and improve testability."""
    
    INITIAL_CANDIDATES = 10
    EXPANSION_SIZE = 5
    MAX_EXPANSIONS = 1
    INITIAL_SHORTLIST_SIZE = 3
    MAX_INSPECTED_CANDIDATES = 5

    def __init__(
        self,
        candidates: list[str],
        csv_dir: Path,
        question: str = "",
        metadata: SolrMetadata | None = None,
        value_search: bool = False,
    ):
        self.candidates = candidates
        self.csv_dir = Path(csv_dir)
        self.question = question
        self.metadata = metadata or {}
        self.value_search = value_search
        self.rejection_keep_tables: list[str] = []
        self.rejection_skip_tables: list[str] = []
        self.visible_candidate_count = min(self.INITIAL_CANDIDATES, len(candidates))
        self.expansion_count = 0
        self.expansion_requirements: list[str] = []
        self._inspection_cache: dict[str, str] = {}
        self._inspection_counts: dict[str, int] = {}
        self.selection_plan: dict[str, object] = {}

    def visible_candidates(self) -> list[str]:
        return self.candidates[: self.visible_candidate_count]

    def inspected_candidates(self) -> list[str]:
        return [
            candidate
            for candidate in self.candidates
            if (
                candidate.casefold() in self._inspection_cache
                and not self._inspection_cache[candidate.casefold()].startswith("Error:")
            )
        ]

    def inspect_columns(self, file_name: str) -> str:
        """
        Returns a compact profile for one table in the active dataset.
        Shows row count, bounded min/max coverage for temporal columns, column
        types, and sample values for low-cardinality categorical columns. A
        repeated request may reuse the cached result.
        Use this to understand what data a table contains.
        """
        name = file_name.strip()
        key = name.casefold()
        if name not in self.visible_candidates():
            return (
                f"Error: {name} is not currently visible. Inspect only candidates "
                "already shown by retrieval or expand_candidates."
            )
        attempted_candidates = len(self._inspection_counts)
        if key not in self._inspection_cache:
            current_limit = (
                self.INITIAL_SHORTLIST_SIZE
                if self.expansion_count == 0
                else self.MAX_INSPECTED_CANDIDATES
            )
            if attempted_candidates >= current_limit:
                if self.expansion_count == 0 and self.visible_candidate_count < len(self.candidates):
                    return (
                        "Inspection blocked: the initial shortlist is limited to 3 "
                        "candidates. If coverage is incomplete, call expand_candidates "
                        "before inspecting another candidate."
                    )
                return (
                    "Inspection blocked: at most 5 distinct candidates may be "
                    "inspected for this request."
                )
        count = self._inspection_counts.get(key, 0) + 1
        self._inspection_counts[key] = count
        if count > MAX_INSPECTIONS_PER_FILE:
            return (
                f"Inspection skipped: {name} has already been inspected "
                f"{MAX_INSPECTIONS_PER_FILE} times. Use the cached schema and "
                "continue with table selection."
            )
        if key not in self._inspection_cache:
            self._inspection_cache[key] = _inspect_columns(self.csv_dir, name)
            return self._inspection_cache[key]
        return (
            f"Cached inspection (attempt {count}/{MAX_INSPECTIONS_PER_FILE}):\n"
            + self._inspection_cache[key]
        )

    def expand_candidates(self, missing_requirements: str) -> str:
        """Reveal up to five hidden candidates that best cover a known gap.

        First inspect the strongest plausible visible candidates and identify the
        missing measure, dimension, filter, period, or join key. This tool does
        not run or re-rank retrieval; it only reveals the next ranked block.
        After expansion, inspect only candidates whose metadata could fill the
        identified gap. Do not call again when no ranked candidates remain.
        """
        if not self.inspected_candidates():
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
        if self.expansion_count >= self.MAX_EXPANSIONS:
            return (
                "Expansion limit reached. Do not call expand_candidates again; "
                "select among the inspected candidates."
            )
        if self.visible_candidate_count >= len(self.candidates):
            return "No additional candidates are available."
        start = self.visible_candidate_count
        hidden = self.candidates[start:]
        newly_visible = _rank_for_missing_requirements(
            hidden, self.metadata, requirements, self.EXPANSION_SIZE
        )
        if not newly_visible:
            self.expansion_count += 1
            self.expansion_requirements = requirements
            return (
                "No hidden candidate has metadata matching the missing requirements. "
                "Do not call expand_candidates again; select or reject using the "
                "inspected evidence."
            )
        selected = set(newly_visible)
        self.candidates[start:] = [
            *newly_visible,
            *(candidate for candidate in hidden if candidate not in selected),
        ]
        self.visible_candidate_count = start + len(newly_visible)
        self.expansion_count += 1
        self.expansion_requirements = requirements
        remaining = len(self.candidates) - self.visible_candidate_count
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
            + format_candidate_context(newly_visible, self.metadata)
            + f"\n\n{next_step}"
        )

    def check_join_union(self, file_name_1: str, file_name_2: str) -> str:
        """
        Check whether two tables join or union with each other. Reports the join
        key columns when they join, the aligned columns when they union, or that
        they neither join nor union.
        """
        return _check_join_union(self.csv_dir, file_name_1, file_name_2)

    def confirm_table_selection(
        self,
        reasoning: str,
        tables: list[str],
        requirement_coverage: dict[str, dict[str, object]] | None = None,
        table_roles: dict[str, str] | None = None,
        combination_strategy: str = "single_table",
        uncovered_requirements: list[str] | None = None,
        requirements: dict[str, object] | None = None,
        semantic_plan: dict[str, object] | None = None,
    ) -> str:
        """
        CRITICAL: Use this tool ONLY when you have identified the required files.
        Calling this tool terminates execution and confirms the selection.
        """
        normalized_tables = list(dict.fromkeys(str(table).strip() for table in tables))
        if not normalized_tables:
            raise ValueError("Selection blocked: at least one inspected table is required.")
        unknown = [table for table in normalized_tables if table not in self.visible_candidates()]
        if unknown:
            raise ValueError(
                "Selection blocked: only currently visible candidates may be selected. "
                f"Unknown table(s): {unknown}."
            )
        inspected = {table.casefold() for table in self.inspected_candidates()}
        uninspected = [
            table for table in normalized_tables if table.casefold() not in inspected
        ]
        if uninspected:
            raise ValueError(
                "Selection blocked: inspect_columns is mandatory for every selected "
                f"table. Inspect {uninspected}, verify complete requirement and "
                "temporal coverage, then confirm again."
            )

        coverage_issue = _temporal_coverage_issue(
            self.question,
            normalized_tables,
            self._inspection_cache,
        )
        if coverage_issue:
            raise ValueError(
                "Selection blocked by temporal validation: " + coverage_issue + "."
            )

        final_tables = ", ".join(normalized_tables)

        coverage = dict(requirement_coverage or {})
        normalized_requirements = dict(requirements or {})
        uncovered = list(uncovered_requirements or [])
        ledger = build_requirement_ledger(
            self.question, coverage, normalized_requirements, uncovered,
            semantic_plan,
        )
        blockers = requirement_ledger_blockers(ledger, normalized_tables)
        if blockers:
            raise ValueError(
                "Selection blocked: fundamental data requirements lack concrete "
                "selected-table/column evidence: " + ", ".join(blockers) + ". "
                "Inspect or expand candidates and bind them in requirement_coverage. "
                "Keep calculations in the ledger as computational."
            )
        self.selection_plan = {
            "requirement_coverage": coverage,
            "table_roles": dict(table_roles or {}),
            "combination_strategy": combination_strategy,
            "uncovered_requirements": uncovered,
            "requirements": normalized_requirements,
            "requirement_ledger": ledger,
            **({"semantic_plan": semantic_plan} if semantic_plan else {}),
        }

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning,
            "selection_plan": self.selection_plan,
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    def reject_selection(
        self, reasoning: str, suggestion: str, keep_tables: list[str] | None = None
    ) -> str:
        """
        Use this tool when the candidates cannot yet fully cover the question's
        essential requirements. List every already-inspected candidate that
        satisfies at least one essential requirement in keep_tables so that
        verified progress is not discarded; every other inspected candidate is
        treated as ruled out and excluded from later retrieval for this question.
        Calling this tool means you have finished this attempt.
        """
        inspected = self.inspected_candidates()
        by_fold = {table.casefold(): table for table in inspected}
        kept = [
            by_fold[table.casefold()]
            for table in (keep_tables or [])
            if table.casefold() in by_fold
        ]
        self.rejection_keep_tables = kept
        self.rejection_skip_tables = [table for table in inspected if table not in kept]
        return f"REJECT_KEYWORDS: {reasoning}\nSuggestion: {suggestion}"

    def get_tools(self) -> list[FunctionTool]:
        reject_schema = (
            RejectSelectionValueSearchSchema if self.value_search else RejectSelectionSchema
        )
        return [
            FunctionTool.from_defaults(fn=self.inspect_columns),
            FunctionTool.from_defaults(fn=self.expand_candidates),
            FunctionTool.from_defaults(fn=self.check_join_union),
            FunctionTool.from_defaults(fn=self.confirm_table_selection, fn_schema=ConfirmSelectionSchema, return_direct=True),
            FunctionTool.from_defaults(fn=self.reject_selection, fn_schema=reject_schema, return_direct=True),
        ]


def make_p2_judge_tools(
    candidates: list[str],
    csv_dir: Path,
) -> list:
    """
    Build tools for the Phase 2 *judge-only* agent.
    Does NOT include search_tables — retrieval is done programmatically
    before the agent runs, and candidates are provided in the prompt.

    Tools: inspect_columns, check_join_union (Valentine),
           confirm_table_selection.
    """
    manager = Phase2JudgeToolsManager(candidates, csv_dir)
    return manager.get_tools()
