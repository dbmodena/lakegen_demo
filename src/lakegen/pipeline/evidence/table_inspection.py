"""Table inspection and candidate ranking shared by the discovery agents.

The helpers of the divided judge, which the MemGPT selector replaced; the
MemGPT selector and the coder use them: a
bounded schema profile with measured temporal coverage, a data preview, the
join/union check, and a ranking of hidden candidates for a missing requirement.
"""

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from lakegen.datastructures.types import SolrMetadata
from lakegen.pipeline.evidence.join_union.pair_check import check_join_union
from lakegen.utils.table_io import (
    iter_table_chunks,
    read_table,
    table_row_count,
)

logger = logging.getLogger(__name__)

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


def _profile_temporal_chunks(
    path: Path, usecols: list[str], temporal_columns: list[str]
) -> tuple[int, dict[str, dict[str, object]]]:
    """Read ``path`` in chunks, up to ``MAX_TEMPORAL_PROFILE_ROWS`` rows, and bound each temporal column.

    A column named like a year is read as numbers between 1000 and 3000, any
    other as UTC timestamps; values that do not parse are left out.

    Args:
        path: The table file.
        usecols: The columns to read.
        temporal_columns: The columns to profile, all within ``usecols``.

    Returns:
        The rows read, and per temporal column its count of parsed values, minimum and maximum.
    """
    profiled_rows = 0
    stats = {col: {"valid": 0, "min": None, "max": None} for col in temporal_columns}

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
    return profiled_rows, stats


def _coverage_lines(stats: dict[str, dict[str, object]], profiled_rows: int) -> list[str]:
    """One "- col: min to max (missing/unparseable N%)" line per column with parsed values.

    Args:
        stats: Per column, its count of parsed values, minimum and maximum.
        profiled_rows: The rows read.

    Returns:
        The lines, dates written as YYYY-MM-DD.
    """
    coverage_lines = []
    for col, col_stats in stats.items():
        if not col_stats["valid"]:
            continue
        unavailable_pct = (
            (profiled_rows - col_stats["valid"]) / profiled_rows * 100 if profiled_rows else 0.0
        )
        min_value = col_stats["min"]
        max_value = col_stats["max"]
        if isinstance(min_value, pd.Timestamp):
            min_value = min_value.strftime("%Y-%m-%d")
            max_value = max_value.strftime("%Y-%m-%d")
        coverage_lines.append(
            f"- {col}: {min_value} to {max_value} (missing/unparseable {unavailable_pct:.1f}%)"
        )
    return coverage_lines


def _temporal_profile(path: Path, columns: list[str]) -> tuple[str, list[str]]:
    """Return row count and bounded coverage for likely temporal columns.

    Args:
        path: The table file.
        columns: The table's columns.

    Returns:
        The row count as shown ("1,234" or "at least 500,000"), and the coverage lines.
    """
    temporal_columns = [col for col in columns if _TEMPORAL_COLUMN_PATTERN.search(str(col))][
        :MAX_TEMPORAL_PROFILE_COLUMNS
    ]
    exact_rows = table_row_count(path)
    if not temporal_columns and exact_rows is not None:
        return f"{exact_rows:,}", []

    usecols = temporal_columns or columns[:1]
    profiled_rows, stats = _profile_temporal_chunks(path, usecols, temporal_columns)
    coverage_lines = _coverage_lines(stats, profiled_rows)

    if exact_rows is not None:
        row_label = f"{exact_rows:,}"
    elif profiled_rows >= MAX_TEMPORAL_PROFILE_ROWS:
        row_label = f"at least {profiled_rows:,}"
    else:
        row_label = f"{profiled_rows:,}"
    if temporal_columns and (exact_rows is None or profiled_rows < exact_rows):
        coverage_lines.insert(0, f"- sampled first {profiled_rows:,} rows (bounded profile)")
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
            if pd.api.types.is_string_dtype(df[col].dtype) or isinstance(
                df[col].dtype, pd.CategoricalDtype
            ):
                unique_vals = df[col].dropna().astype(str).unique().tolist()
                if 0 < len(unique_vals) <= MAX_UNIQUE_VALUES:
                    values = [_compact_value(value) for value in unique_vals]
                    schema_info.append(f"- {col} (Category sample): {values}")
                    continue
            schema_info.append(f"- {col} ({dtype})")

        if len(columns) > MAX_SCHEMA_COLUMNS:
            schema_info.append(f"- ... {len(columns) - MAX_SCHEMA_COLUMNS} more columns omitted")

        header_lines = [f"Schema for {file_name}:", f"Rows: {row_label}"]
        if temporal_coverage:
            header_lines.extend(["Temporal coverage:", *temporal_coverage])
        header_lines.append(
            f"Columns (types and categories sampled from first {MAX_SCHEMA_SAMPLE_ROWS} rows):"
        )
        output = "\n".join(header_lines + schema_info)
        return _compact_tool_output(output)
    except Exception as e:
        logger.debug("inspect_columns failed on %s", path, exc_info=True)
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


def _check_join_union(table_dir: Path, file_name_1: str, file_name_2: str) -> str:
    """Measure whether two tables join (shared key values) or union (mappable columns)."""
    return _compact_tool_output(check_join_union(table_dir, file_name_1, file_name_2))


_EXPANSION_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "or",
    "the",
    "to",
    "con",
    "da",
    "del",
    "della",
    "di",
    "e",
    "il",
    "la",
    "per",
    "un",
    "una",
}


def _requirement_terms(value: str) -> list[str]:
    """Extract a small, stable set of explicit coverage requirements."""
    terms = re.findall(r"[a-z0-9]+", str(value).casefold())
    return list(
        dict.fromkeys(term for term in terms if len(term) > 1 and term not in _EXPANSION_STOPWORDS)
    )[:12]


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
