"""Data facts about the selected tables, for the planner and the plan judge.

A plan is only as good as what its author knew about the data. Metadata says
what a table is supposed to hold (a description reading "invoices of GBP 500
or over"); this says what it does hold: which values a column takes, whether a
number is stored as text, and where its real minimum lies. A plan can then
cite evidence, and the plan judge can tell a constraint that is truly
unnecessary from one that only looks it.

Facts are computed from the frames the reviewed coder already loads, on a
bounded prefix so a large table costs no more than a small one, and every
column line states what it was computed from.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from lakegen.agent_tools.tools_p2 import _TEMPORAL_COLUMN_PATTERN, _parse_datetimes

MAX_PROFILED_ROWS = 200_000
MAX_COLUMNS_PER_TABLE = 40
MAX_LISTED_VALUES = 12
MAX_EXAMPLES = 3

_NUMBER_TEXT = re.compile(r"^\(?-?[£$€]?\s*\d[\d,]*(?:\.\d+)?\)?$")


def _to_number(text: pd.Series) -> pd.Series:
    """Numbers written as text: thousands separators, a currency sign, and
    accounting negatives such as (683.94)."""
    cleaned = text.astype(str).str.strip()
    negative = cleaned.str.startswith("(") & cleaned.str.endswith(")")
    cleaned = cleaned.str.replace(r"[()£$€\s]", "", regex=True)
    cleaned = cleaned.str.replace(",", "", regex=False)
    numbers = pd.to_numeric(cleaned, errors="coerce")
    return numbers.where(~negative, -numbers.abs())


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _column_fact(name: str, series: pd.Series) -> str:
    non_null = series.dropna()
    null_share = 1 - len(non_null) / len(series) if len(series) else 0.0
    nulls = f", {null_share:.0%} missing" if null_share >= 0.01 else ""
    if non_null.empty:
        return f"- {name}: no values"

    if pd.api.types.is_bool_dtype(series.dtype):
        return f"- {name} (boolean{nulls})"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return (
            f"- {name} (number{nulls}): min {_fmt(non_null.min())}, "
            f"max {_fmt(non_null.max())}"
        )
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return f"- {name} (date{nulls}): {non_null.min():%Y-%m-%d} to {non_null.max():%Y-%m-%d}"

    text = non_null.astype(str).str.strip()
    text = text[text != ""]
    if text.empty:
        return f"- {name} (text{nulls}): blank"

    parsed = _to_number(text)
    share = parsed.notna().mean()
    if share >= 0.5 and text.str.match(_NUMBER_TEXT).mean() >= 0.5:
        valid = parsed.dropna()
        brackets = int((text.str.startswith("(") & text.str.endswith(")")).sum())
        unparsed = int(parsed.isna().sum())
        examples = ", ".join(repr(v) for v in text[parsed.notna()].drop_duplicates().head(MAX_EXAMPLES))
        notes = []
        if brackets:
            notes.append(f"{brackets} in parentheses = negative")
        if unparsed:
            notes.append(f"{unparsed} not numeric")
        return (
            f"- {name} (TEXT holding numbers{nulls}; e.g. {examples}): as numbers "
            f"min {_fmt(float(valid.min()))}, max {_fmt(float(valid.max()))}"
            + (f"; {'; '.join(notes)}" if notes else "")
        )

    if _TEMPORAL_COLUMN_PATTERN.search(str(name)):
        dates = _parse_datetimes(text).dropna()
        if len(dates) >= 0.5 * len(text):
            return (
                f"- {name} (TEXT holding dates{nulls}; e.g. {text.iloc[0]!r}): "
                f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"
            )

    counts = text.value_counts()
    if len(counts) <= MAX_LISTED_VALUES:
        listed = ", ".join(f"{v!r} ({c:,})" for v, c in counts.items())
        return f"- {name} (text{nulls}): {len(counts)} distinct: {listed}"
    examples = ", ".join(repr(v) for v in counts.head(MAX_EXAMPLES).index)
    return f"- {name} (text{nulls}): {len(counts):,} distinct, most frequent {examples}"


def build_table_facts(
    frames: dict[str, pd.DataFrame], solr_meta: dict[str, Any] | None = None
) -> str:
    """One block per table: what it is said to be, then what it holds."""
    solr_meta = solr_meta or {}
    blocks: list[str] = []
    for table, frame in frames.items():
        meta = solr_meta.get(table) or solr_meta.get(str(table).rsplit(".", 1)[0]) or {}
        title = str(meta.get("title") or "").strip()
        description = " ".join(str(meta.get("description") or "").split())[:240]
        sample = frame.head(MAX_PROFILED_ROWS)
        head = [f"Table: {table}"]
        if title:
            head.append(f"  Title: {title}")
        if description:
            head.append(f"  Description (what the publisher says, not verified): {description}")
        scope = (
            f"first {len(sample):,} of {len(frame):,} rows"
            if len(sample) < len(frame) else f"all {len(frame):,} rows"
        )
        head.append(f"  Rows: {len(frame):,}. Column facts below are computed from {scope}.")
        columns = list(sample.columns)
        lines = [_column_fact(str(c), sample[c]) for c in columns[:MAX_COLUMNS_PER_TABLE]]
        if len(columns) > MAX_COLUMNS_PER_TABLE:
            lines.append(f"- ... {len(columns) - MAX_COLUMNS_PER_TABLE} more columns not profiled")
        blocks.append("\n".join(head + ["  Columns:"] + [f"  {line}" for line in lines]))
    return "\n\n".join(blocks)
