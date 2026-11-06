"""What the whole table says: the additions to the profile that keep a card from stating false negatives.

The shared profile (``tables.table_profile``) is read from the first 500 rows: a text column with more
than 8 distinct values shows only its type, a numeric column only its type, and a year held as an
integer (``Any``, ``exercice_comptable``, ``anno``) is never measured. The cards built from it said
"words found nowhere: debra, coupar" for the table whose ``Traveller Name`` holds Debra Coupar, and
"2021: not in any inspected value or date range" for the table whose ``Any`` is 2021, and the selector
rejected or added a table on them. ``evidence_profile`` writes, from the whole table (at most
``MAX_ROWS`` rows):

  years      an integer column whose values are all years, as its complete list of values (so the
             ledger, the cards and the period items read them as values)
  ranges     the range of a numeric column; the number of distinct values and the most frequent ones
             of a text column
  steps      how far apart the dates of a date column are (every 15 minutes, daily, monthly)
  dates      text columns of m/d/yy dates, which the shared profile does not read
  search     the question's and the requirements' words found in the text values, with the column
             and an example (``SEARCH_HEADER``); the cards read them as values (``searched``)

Validated in experiments/profile_lab (2026-10-22, 400 questions).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from lakegen.pipeline.discovery.selectors.memgpt.tables import PROFILE_CHARS, table_profile
from lakegen.pipeline.discovery.selectors.memgpt.text import fold, mentions
from lakegen.pipeline.evidence.table_inspection import _compact_tool_output
from lakegen.utils.table_io import read_table

logger = logging.getLogger(__name__)

MAX_ROWS = 200_000
MAX_UNIQUES = 50_000
MAX_YEARS_LISTED = 30
FREQUENT = 4
MAX_WORDS = 30
"""Words searched in the values: the question's, then the requirements'."""
MAX_FOUND = 10
SEARCH_HEADER = "Values matching the question (whole table): "
OTHER_HEADER = "Other columns (geometry, GIS ids, audit; not profiled): "
"""One line for the columns that never hold what a question asks: geometry blobs, the GIS system's ids, who edited
the row and when. Their sampled values were 11% of the profile text (coverage_lab, decision2) and a date range of
``last_edited_date`` made a period look covered."""
_OTHER = re.compile(
    r"^(?:geo_?shape|geo_?point(?:_2d)?|geopoint|the_geom|geom(?:etry)?|wkt|wkb|geojson|shape|georeferenced?(?: column)?|"
    r"objectid|globalid|fid|ogc_fid|created_(?:user|date)|last_edited_(?:user|date)|creator|editor|edit_?date)$",
    re.I,
)
_COLUMN_LINE = re.compile(r"^- (.+?) \((\w+)\)$")
_COVERAGE_LINE = re.compile(r"^- (.+?): (\S+) to (\S+)(.*)$")
_SHORT_DATE = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2}\s*$")
_SEARCH = re.compile(r"(\S+) in (.+?) \[e\.g\. [^\]]*\]")
STEPS = (
    (pd.Timedelta(minutes=1), "every minute or less"),
    (pd.Timedelta(minutes=5), "every 5 minutes"),
    (pd.Timedelta(minutes=10), "every 10 minutes"),
    (pd.Timedelta(minutes=15), "every 15 minutes"),
    (pd.Timedelta(minutes=30), "every 30 minutes"),
    (pd.Timedelta(hours=1), "hourly"),
    (pd.Timedelta(days=1), "daily"),
    (pd.Timedelta(days=7), "weekly"),
    (pd.Timedelta(days=30.4), "monthly"),
    (pd.Timedelta(days=91.3), "quarterly"),
    (pd.Timedelta(days=365.25), "yearly"),
)


@lru_cache(maxsize=8)
def _frame(path: str, mtime: int) -> pd.DataFrame | None:
    try:
        return read_table(Path(path), nrows=MAX_ROWS)
    except Exception:
        logger.debug("No evidence frame for %s", path, exc_info=True)
        return None


def frame(path: Path) -> pd.DataFrame | None:
    """The table's first ``MAX_ROWS`` rows (the last few tables read are kept)."""
    try:
        return _frame(str(path), path.stat().st_mtime_ns)
    except OSError:
        return None


def _is_text(series: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_datetime64_any_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def year_values(series: pd.Series) -> list[int] | None:
    """The years an integer column holds, when every value is one."""
    if not (
        pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_float_dtype(series.dtype)
    ):
        return None
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2 or not ((values == values.round()) & values.between(1900, 2100)).all():
        return None
    return sorted({int(v) for v in values.unique()})


def _number(value: float) -> str:
    """A number written so that a count like 2019 does not read as a year."""
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"


def step(values: pd.Series) -> str:
    """How far apart a column's dates are: the typical gap between consecutive distinct dates."""
    parsed = pd.to_datetime(
        values.dropna().astype(str).unique()[:5000], errors="coerce", utc=True, format="mixed"
    )
    parsed = pd.Series(parsed).dropna().sort_values().drop_duplicates()
    if len(parsed) < 3:
        return ""
    gap = parsed.diff().dropna().median()
    for bound, label in STEPS:
        if gap <= bound * 1.1:
            return label
    return f"every {gap.days} days"


def short_date_coverage(series: pd.Series) -> str | None:
    """The range of a text column of m/d/yy dates (the shared profile reads only four-digit years)."""
    text = series.dropna().astype(str)
    if len(text) < 5 or text.head(500).str.match(_SHORT_DATE).mean() < 0.8:
        return None
    parsed = pd.to_datetime(text, format="%m/%d/%y", errors="coerce").dropna()
    if parsed.empty:
        return None
    return f"{parsed.min():%Y-%m-%d} to {parsed.max():%Y-%m-%d}"


def search(df: pd.DataFrame, words: list[str], profile: str) -> dict[str, tuple[str, str]]:
    """The words the profile does not show, found in the text values: word -> (column, example value)."""
    shown = fold(profile)
    wanted = [w for w in words if not mentions(shown, fold(w))]
    found: dict[str, tuple[str, str]] = {}
    for column in df.columns:
        if not wanted or not _is_text(df[column]):
            continue
        uniques = df[column].dropna().astype(str).unique()[:MAX_UNIQUES]
        folded = [fold(v) for v in uniques]
        for word in list(wanted):
            stem = fold(word)
            hit = next((v for v, f in zip(uniques, folded, strict=True) if mentions(f, stem)), None)
            if hit is not None:
                found[word] = (
                    str(column),
                    " ".join(str(hit).split())[:50].replace("[", "(").replace("]", ")"),
                )
                wanted.remove(word)
    return found


def other_column(name: str) -> bool:
    """A geometry, GIS id or audit column: summarised on the ``OTHER_HEADER`` line instead of profiled."""
    return bool(_OTHER.match(name.strip()))


def column_name(line: str) -> str:
    """The column a profile line describes ("- name (...)" or "- name: ..."), else ""."""
    match = re.match(r"^- (.+?)(?: \(|: )", line)
    return match.group(1) if match else ""


def column_line(line: str, df: pd.DataFrame) -> tuple[str, str | None]:
    """A profile's column line written from the whole column, and the coverage line it adds (years, m/d/yy dates)."""
    match = _COLUMN_LINE.match(line)
    if not match or match.group(1) not in df.columns:
        return line, None
    column, dtype = match.groups()
    series = df[column].dropna()
    years = year_values(series)
    if years and len(years) <= MAX_YEARS_LISTED:
        return f"- {column} (Category sample): {[str(y) for y in years]}", None
    if years:
        return line, f"- {column}: {years[0]} to {years[-1]} (years held as integers)"
    if series.empty:
        return line, None
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
        return f"- {column} ({dtype}; {_number(series.min())} to {_number(series.max())})", None
    if not _is_text(series):
        return line, None
    dates = short_date_coverage(series)
    counts = series.astype(str).value_counts()
    coverage = f"- {column}: {dates} (m/d/yy dates parsed from the values)" if dates else None
    if len(counts) >= len(series):  # an id or a free text: its most frequent values say nothing
        return f"- {column} ({dtype}; {len(counts):,} distinct, one per row)", coverage
    frequent = ", ".join(repr(" ".join(v.split())[:40]) for v in counts.index[:FREQUENT])
    return f"- {column} ({dtype}; {len(counts):,} distinct; most frequent: {frequent})", coverage


def head_lines(head: str, df: pd.DataFrame, coverage: list[str]) -> list[str]:
    """The profile's lines above the columns, each date range with its step, and the added coverage lines."""
    out = []
    for line in head.splitlines():
        match = _COVERAGE_LINE.match(line)
        if match and other_column(match.group(1)):
            continue  # when a row was edited is not the table's period
        label = step(df[match.group(1)]) if match and match.group(1) in df.columns else ""
        if label:
            line = line[:-1] + f"; {label})" if line.endswith(")") else f"{line} ({label})"
        out.append(line)
    if coverage:
        if "Temporal coverage:" not in out:
            out.append("Temporal coverage:")
        out.extend(coverage)
    return out


def evidence_profile(table_dir: Path, name: str, words: list[str]) -> str:
    """The shared profile with years, ranges, steps and the question's words in the values, from the whole table."""
    base = table_profile(table_dir, name)
    df = frame(Path(table_dir) / name.strip())
    if base.startswith("Error") or df is None:
        return base
    head, _, columns_part = base.partition("\nColumns (")
    lines = columns_part.splitlines()
    header = (lines[0] if lines else "") + " Ranges, years and counts: whole table."
    written = [column_line(line, df) for line in lines[1:]]
    other = [column_name(line) for line, _ in written if other_column(column_name(line))]
    written = [(line, extra) for line, extra in written if column_name(line) not in other]
    out = head_lines(head, df, [extra for _, extra in written if extra])
    found = search(df, words, base)
    if found:
        out.append(
            SEARCH_HEADER
            + "; ".join(f"{w} in {c} [e.g. {v}]" for w, (c, v) in list(found.items())[:MAX_FOUND])
        )
    text = "\n".join(out) + "\nColumns (" + header + "".join(f"\n{line}" for line, _ in written)
    if other:
        text += f"\n{OTHER_HEADER}{', '.join(other)}"
    return _compact_tool_output(text, PROFILE_CHARS)


def searched(profile: str) -> dict[str, str]:
    """Word -> column, from the profile's search line (empty for a profile without one)."""
    line = next((ln for ln in profile.splitlines() if ln.startswith(SEARCH_HEADER)), "")
    return {w: c for w, c in _SEARCH.findall(line[len(SEARCH_HEADER) :])}
