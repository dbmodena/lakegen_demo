import json
import re
from functools import lru_cache
import pandas as pd
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

from lakegen.core_2.table_io import (
    TableProfile,
    iter_table_chunks,
    read_profile,
    read_table,
    read_table_sample,
    table_row_count,
)
from lakegen.core.types import SolrMetadata
from lakegen.phases.utils import format_candidate_context
from lakegen.agent_tools.requirement_ledger import (
    build_requirement_ledger,
    requirement_ledger_blockers,
)
from lakegen.agent_tools_2.join_keys import find_join_keys, format_join_section
from lakegen.agent_tools_2.schema_matching import verify_pair_schema
from lakegen.agent_tools_2.union_mapping import format_union_section, union_by_names

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
MIN_BAN_JUSTIFICATION_CHARS = 10

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
    ban_tables: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Already-inspected candidates you judge highly irrelevant -- they "
            "satisfy none of the essential requirements -- mapped to the "
            "concrete evidence proving that (e.g. 'Account Name values are "
            "all Leeds City Council, none match Transport for Greater "
            "Manchester'), so they are excluded from later retrieval for this "
            "question. Every other inspected candidate is kept and carries "
            "over to the next attempt, even if it does not yet cover every "
            "requirement -- only actively-proven-irrelevant tables belong "
            "here. A table without a concrete justification is not banned."
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
    ban_tables: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Already-inspected candidates you judge highly irrelevant -- they "
            "satisfy none of the essential requirements -- mapped to the "
            "concrete evidence proving that (e.g. 'Account Name values are "
            "all Leeds City Council, none match Transport for Greater "
            "Manchester'), so they are excluded from later retrieval for this "
            "question. Every other inspected candidate is kept and carries "
            "over to the next attempt, even if it does not yet cover every "
            "requirement -- only actively-proven-irrelevant tables belong "
            "here. A table without a concrete justification is not banned."
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


# Rows measured per table by check_join_union. Shared-value counts and column mappings only need a
# bounded sample; COMA is given a much smaller one (schema_matching.COMA_SAMPLE_ROWS). Measured on 73
# real 200k-1M-row UK pairs: 20,000 rows take about 0.3 s per pair (p90 1 s) and found 41 of 45 joins;
# 200,000 rows take about 1 s (p90 3 s) and found all 45. The joins the smaller sample misses are
# low-overlap ones (0.2%-15% of rows match).
PAIR_SAMPLE_ROWS = 20_000


def _sampling_note(name: str, frame: pd.DataFrame, total: int | None) -> str:
    if total is not None and total > len(frame):
        return f"'{name}' was sampled ({len(frame):,} of {total:,} rows)"
    if total is None and len(frame) >= PAIR_SAMPLE_ROWS:
        return f"'{name}' was cut to its first {len(frame):,} rows"
    return ""


def _format_join_union(
    file_name_1: str,
    file_name_2: str,
    q: pd.DataFrame,
    r: pd.DataFrame,
    totals: tuple[int | None, int | None],
    matches: list[tuple[str, str, float]],
    profiles: tuple[TableProfile | None, TableProfile | None] = (None, None),
) -> str:
    """Report whether two tables join and whether they union, from measured values and coverage.

    ``profiles`` are the whole-table facts of each side when ``q`` / ``r`` are samples; without them a
    frame is taken to be the whole table.
    """
    sizes = "; ".join(
        f"'{name}' {len(frame):,} rows x {frame.shape[1]} columns"
        for name, frame in ((file_name_1, q), (file_name_2, r))
    )
    lines = [
        f"Join/union check between '{file_name_1}' and '{file_name_2}' "
        f"(Valentine column matching; values measured on {sizes})."
    ]
    sampled = [
        note
        for note in (
            _sampling_note(file_name_1, q, totals[0]),
            _sampling_note(file_name_2, r, totals[1]),
        )
        if note
    ]
    if sampled:
        lines.append(
            "Note: " + "; ".join(sampled) + " -- shared-value counts are lower bounds and "
            "uniqueness is an upper bound."
        )
    lines.append(
        format_join_section(
            file_name_1, file_name_2, find_join_keys(q, r, matches, profiles[0], profiles[1]), q, r
        )
    )
    lines.append(format_union_section(file_name_1, file_name_2, union_by_names(q, r, matches)))
    return "\n".join(lines)


def _read_pair_side(
    table_dir: Path, file_name: str
) -> tuple[pd.DataFrame, int | None, TableProfile | None]:
    """One table of a pair as (frame, total rows, profile).

    The profile (parquet footer statistics) is read only when the frame is a sample of a larger
    table: a random sample of a row-number column (1..N) is no longer consecutive, so only the
    whole-table range tells the check it is not a join key. A whole table needs no profile.
    """
    path = _table_path(table_dir, file_name)
    frame, total = read_table_sample(path, PAIR_SAMPLE_ROWS)
    sampled = total is not None and total > len(frame)
    return frame, total, read_profile(path) if sampled else None


def _check_join_union(table_dir: Path, file_name_1: str, file_name_2: str) -> str:
    """Measure whether two tables join (shared key values) or union (mappable columns)."""
    try:
        (q, q_total, q_profile), (r, r_total, r_profile) = (
            _read_pair_side(table_dir, file_name) for file_name in (file_name_1, file_name_2)
        )
        matches = verify_pair_schema(q, r)["matches"]
        report = _format_join_union(
            file_name_1, file_name_2, q, r, (q_total, r_total), matches, (q_profile, r_profile)
        )
    except Exception as e:
        return f"Error checking join/union between '{file_name_1}' and '{file_name_2}': {e}. Try different tables."

    return _compact_tool_output(report)


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
    MAX_INSPECTED_CANDIDATES = 6

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

    def inspect_columns(
        self, file_name: str | None = None, candidate_number: int | None = None
    ) -> str:
        """
        Returns a compact profile for one table in the active dataset.
        Shows row count, bounded min/max coverage for temporal columns, column
        types, and sample values for low-cardinality categorical columns. A
        repeated request may reuse the cached result.
        Use this to understand what data a table contains.

        Prefer candidate_number -- the "Candidate N" label the candidate list
        and expand_candidates already print above each entry -- over
        file_name. Generated filenames are long and easy to mistype; a small
        integer has nothing to transcribe incorrectly.
        """
        visible = self.visible_candidates()
        if candidate_number is not None:
            if not (1 <= candidate_number <= len(visible)):
                return (
                    f"Error: candidate_number must be between 1 and "
                    f"{len(visible)} (the currently visible candidates)."
                )
            name = visible[candidate_number - 1]
        else:
            if not file_name:
                return (
                    "Error: candidate_number (preferred) or file_name is required."
                )
            name = file_name.strip()
        key = name.casefold()
        if name not in visible:
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
            + format_candidate_context(
                newly_visible, self.metadata, start_rank=start + 1
            )
            + f"\n\n{next_step}"
        )

    def check_join_union(self, file_name_1: str, file_name_2: str) -> str:
        """
        Check whether two tables join or union by measuring their values. Reports
        ranked join-key candidates with shared-value counts, cardinality and join size
        (or that no name-similar key shares values), and a union verdict (UNION,
        SUBSET UNION, PARTIAL UNION or NO UNION) with the column mapping and any type
        clashes.
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
        inspected = self.inspected_candidates()
        by_fold = {table.casefold(): table for table in inspected}
        # A bare name with no justification is how a table could get banned
        # on a whim rather than proven evidence. Require the model to commit
        # to a concrete reason before it is excluded from the rest of the
        # question.
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
        self.rejection_skip_tables = banned
        self.rejection_keep_tables = [table for table in inspected if table not in banned]
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
