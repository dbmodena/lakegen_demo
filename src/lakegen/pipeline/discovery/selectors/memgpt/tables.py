"""What the MemGPT selector reads about tables: a profile, and how two tables combine.

The profile is the shared one (``tools.shared.judge._inspect_columns``) with one addition. That
profile measures temporal coverage only for columns named like ``date``, ``time``, ``year`` or
``timestamp``, so a text column named ``dateobserved``, ``recvtime``, ``fecha`` or
``data_rilevazione`` is listed as ``(str)`` with no range, and the question's year is then "not in
any inspected value or date range": a correct partition looks as if it lacked the year. Text
columns whose values parse as dates are measured over the whole column (at most ``MAX_ROWS`` rows)
and written in the lines the ledger and the cards read ("- col: start to end (...)").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from lakegen.pipeline.evidence.join_union import check_join_union
from lakegen.pipeline.evidence.table_inspection import _compact_tool_output, _inspect_columns
from lakegen.utils.table_io import iter_table_chunks, read_table

logger = logging.getLogger(__name__)

PROFILE_CHARS = 4000
DATE_NAME = re.compile(
    r"(date|datum|time|timestamp|observed|recv|fecha|hora|data|anno|ann[ée]e|year|jahr|period|mes\b|month|"
    r"d[ií]a\b|jour|giorno|quarter|trimestre|semestre|week|semaine|settimana|created|updated|modified)",
    re.IGNORECASE,
)
# A value counts as a date only when it looks like one: ISO or slashed dates, optionally with a time.
DATE_SHAPE = re.compile(
    r"^\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{4}|\d{4}[-/]\d{1,2}(?![\d.]))"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?\s*$"
)
SANE_YEARS = (1800, 2100)
SAMPLE_ROWS = 300
MAX_ROWS = 500_000
CHUNK_ROWS = 100_000
MAX_COLUMNS = 3
_RANGE_LINE = re.compile(r"^- (.+?): \S+ to \S+", re.MULTILINE)
_TEXT_COLUMN = re.compile(r"^- (.+?) \((?:str|object|string|Category sample)", re.MULTILINE)
_KEY_FILES = re.compile(r" \([^()]*\.(?:parquet|csv)\)")


def table_profile(table_dir: Path, name: str) -> str:
    """The shared profile of a table with the dates its text columns hold, bounded for the context."""
    raw = _inspect_columns(table_dir, name)
    return _compact_tool_output(
        with_text_date_coverage(table_dir / name.strip(), raw), PROFILE_CHARS
    )


def relationship(table_dir: Path, first: str, second: str) -> str:
    """How two tables join and union, in one line."""
    return summarize_relationship(check_join_union(table_dir, first, second))


def summarize_relationship(report: str) -> str:
    """One line from a check_join_union report: the join verdict with its best key and size, the union verdict."""
    if report.startswith("Error"):
        return report.splitlines()[0][:200]
    lines = report.splitlines()
    join = next((line for line in lines if re.match(r"(NO )?JOIN", line)), "")
    key = next((line.strip() for line in lines if re.match(r"\s+1\. ", line)), "")
    union = next((line for line in lines if re.match(r"(NO |SUBSET |PARTIAL )?UNION", line)), "")
    parts: list[str] = []
    if join.startswith("NO JOIN"):
        parts.append("NO JOIN")
    elif join.startswith("JOIN"):
        label = (
            "JOIN (many-to-many: aggregate one side first)" if "many-to-many" in join else "JOIN"
        )
        match = re.match(r"1\. (.+?) -- ([^,]+), (.+?);.*?inner join = (.+?) rows", key)
        if match:
            key_text = _KEY_FILES.sub("", match.group(1))
            parts.append(
                f"{label} on {key_text} ({match.group(2)}, {match.group(3)}, inner join {match.group(4)} rows)"
            )
        else:
            parts.append(label + (": " + _KEY_FILES.sub("", key)[:160] if key else ""))
    if union:
        verdict = union.split(":", 1)[0]
        columns = re.search(r"(\d+) columns? map", union)
        parts.append(verdict + (f" ({columns.group(1)} columns map 1-1)" if columns else ""))
    return "; ".join(parts) or (lines[0][:200] if lines else "")


# ---------------------------------------------------------------- dates held as text


def _parses_as_dates(values: pd.Series) -> bool:
    text = values.dropna().astype(str).str.strip()
    text = text[text != ""]
    if len(text) < 5:
        return False
    if text.str.match(DATE_SHAPE).mean() < 0.8:
        return False
    parsed = pd.to_datetime(text, errors="coerce", utc=True, format="mixed")
    years = parsed.dropna().dt.year
    return (
        parsed.notna().mean() >= 0.8
        and bool(len(years))
        and years.between(*SANE_YEARS).mean() >= 0.95
    )


def date_text_columns(path: Path, profile: str) -> list[str]:
    """Text columns of the profile that hold dates and have no measured range yet."""
    covered = set(_RANGE_LINE.findall(profile))
    candidates = [c for c in _TEXT_COLUMN.findall(profile) if c not in covered]
    if not candidates:
        return []
    try:
        sample = read_table(path, nrows=SAMPLE_ROWS, columns=candidates)
    except Exception:
        logger.debug("No text-date sample of %s", path, exc_info=True)
        return []
    found = []
    for column in candidates:
        if column not in sample.columns:
            continue
        try:
            if _parses_as_dates(sample[column]):
                found.append(column)
        except Exception:
            logger.debug("Column %s of %s not read as dates", column, path, exc_info=True)
            continue
    # name hints first, so the most plausible columns are measured within the limit
    found.sort(key=lambda c: 0 if DATE_NAME.search(c) else 1)
    return found[:MAX_COLUMNS]


def _measure(stats: dict[str, dict], chunk: pd.DataFrame) -> None:
    for column, s in stats.items():
        parsed = pd.to_datetime(
            chunk[column].astype(str), errors="coerce", utc=True, format="mixed"
        ).dropna()
        if parsed.empty:
            continue
        s["valid"] += len(parsed)
        lo, hi = parsed.min(), parsed.max()
        s["min"] = lo if s["min"] is None or lo < s["min"] else s["min"]
        s["max"] = hi if s["max"] is None or hi > s["max"] else s["max"]


def text_date_coverage(path: Path, profile: str) -> list[str]:
    """Coverage lines ("- col: start to end (...)") for the text columns that hold dates."""
    columns = date_text_columns(path, profile)
    if not columns:
        return []
    stats = {c: {"valid": 0, "min": None, "max": None} for c in columns}
    rows = 0
    try:
        for chunk in iter_table_chunks(path, columns=columns, chunk_rows=CHUNK_ROWS):
            remaining = MAX_ROWS - rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)
            rows += len(chunk)
            _measure(stats, chunk)
    except Exception:
        logger.debug("Text-date coverage of %s not measured", path, exc_info=True)
        return []
    lines = []
    for column, s in stats.items():
        if not s["valid"] or s["min"] is None:
            continue
        missing = (rows - s["valid"]) / rows * 100 if rows else 0.0
        note = "text dates parsed from the values" + (
            f", first {rows:,} rows" if rows >= MAX_ROWS else ""
        )
        lines.append(
            f"- {column}: {s['min'].strftime('%Y-%m-%d')} to {s['max'].strftime('%Y-%m-%d')} ({note}; missing/unparseable {missing:.1f}%)"
        )
    return lines


def with_text_date_coverage(path: Path, profile: str) -> str:
    """The profile with the text dates' lines under its Temporal coverage header (added when absent)."""
    if profile.startswith("Error"):
        return profile
    lines = text_date_coverage(path, profile)
    if not lines:
        return profile
    block = "\n".join(lines)
    if "\nTemporal coverage:\n" in profile:
        head, rest = profile.split("\nTemporal coverage:\n", 1)
        return f"{head}\nTemporal coverage:\n{block}\n{rest}"
    marker = "\nColumns ("
    if marker in profile:
        head, rest = profile.split(marker, 1)
        return f"{head}\nTemporal coverage:\n{block}{marker}{rest}"
    return profile + "\nTemporal coverage:\n" + block
