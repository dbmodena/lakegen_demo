"""Measured join-key evidence for a pair of tables.

Valentine's schema-only COMA proposes column pairs by NAME similarity, and identical names score
1.0 everywhere in open data (`Date`, `Name`, `Parent Department`), so a name match alone says
almost nothing about whether two tables join. Each candidate pair is therefore measured against
the actual values, and the report states facts instead of asserting one key:

  * do the columns share any values (after strip/casefold, int-vs-float and numeric-text
    normalisation)?
  * is the key single-valued on both sides (a cross product), or just 1..N row numbers on both
    sides (overlap guaranteed by construction)?
  * the cardinality (1:1, 1:N, N:1, N:N) and the exact inner-join row count, so an N:N join is
    flagged "aggregate first" instead of being silently accepted;
  * composite keys, because real open-data joins often need several columns.

Candidates are ranked, not decided: values cannot choose between equivalent identifiers
(`BBL` / `BIN` / `Boro-Block-Lot`) or the column a question groups by, so the caller sees the
top few with their facts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

from lakegen.agent_tools.column_values import (
    column_kind,
    format_datetime,
    format_number,
    format_text,
    map_uniques,
    mostly_numeric,
)
from lakegen.agent_tools.schema_matching import SCHEMA_MATCH_THRESHOLD

if TYPE_CHECKING:
    from lakegen.core.table_io import TableProfile

# distinct/non-null rows at or above this count a side as "unique" for the cardinality label
UNIQUE_RATIO = 0.99
# a match is "thin" when neither side has this share of its key rows finding a partner
MIN_MATCH_RATE = 0.5
# this many shared values or fewer, with a thin match, is indistinguishable from chance
COINCIDENCE_SHARED = 2

MAX_KEY_COLUMNS = 4
MIN_SINGLE_CANDIDATES = 10
MAX_SINGLE_CANDIDATES = 60
CELL_BUDGET = 30_000_000       # ~ rows scanned across all measured candidates
COMPOSITE_POOL = 6             # name-similar pairs considered for composite keys
COMPOSITE_MAX_ROWS = 400_000   # skip the composite search on very large table pairs

# Only JOIN is "supported". Rejection needs hard evidence (no shared values, a constant key,
# row-number ids) or coincidence-level evidence (a thin match backed by very few values). A thin
# overlap backed by many shared values stays a JOIN -- benchmark joins include `how="left"`
# merges and "appears in both" questions where a small overlap is the point -- and is reported as
# a fact via `KeyEvidence.thin`.
JOIN = "JOIN"
WEAK_OVERLAP = "WEAK_OVERLAP"
SURROGATE_ID = "SURROGATE_ID"
CONSTANT_KEY = "CONSTANT_KEY"
NO_OVERLAP = "NO_OVERLAP"
_TIER = {JOIN: 0, WEAK_OVERLAP: 1, SURROGATE_ID: 2, CONSTANT_KEY: 3, NO_OVERLAP: 4}


@dataclass
class KeyEvidence:
    left_cols: tuple[str, ...]
    right_cols: tuple[str, ...]
    name_score: float
    verdict: str
    cardinality: str = ""
    left_rows: int = 0            # non-null key rows
    right_rows: int = 0
    left_distinct: int = 0
    right_distinct: int = 0
    shared_distinct: int = 0
    left_match_rate: float = 0.0  # share of left key rows whose key exists on the right
    right_match_rate: float = 0.0
    left_unique_ratio: float = 0.0
    right_unique_ratio: float = 0.0
    inner_rows: int = 0
    measure_like: bool = False    # a key column holds fractional numbers (amounts, coordinates)
    notes: list[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return self.verdict == JOIN

    @property
    def thin(self) -> bool:
        """Few of the rows on EITHER side find a partner."""
        best = max(self.left_match_rate, self.right_match_rate)
        return self.shared_distinct > 0 and best < MIN_MATCH_RATE

    @property
    def unique_side(self) -> bool:
        return max(self.left_unique_ratio, self.right_unique_ratio) >= UNIQUE_RATIO


def _normalise_pair(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series, bool]:
    """Bring two key columns to a common text form.

    Returns (a_norm, b_norm, dtype_compatible); the flag is False when the sides could not be
    brought to a comparable kind.
    """
    kind_a, kind_b = column_kind(a), column_kind(b)
    if kind_a == kind_b == "dt":
        return map_uniques(a, format_datetime), map_uniques(b, format_datetime), True
    # numbers stored as text ('2,012,938', '007') are numbers: compare them as such
    numeric_a = kind_a == "num" or (kind_a == "str" and mostly_numeric(a))
    numeric_b = kind_b == "num" or (kind_b == "str" and mostly_numeric(b))
    if numeric_a and numeric_b:
        return map_uniques(a, format_number), map_uniques(b, format_number), True
    compatible = kind_a == kind_b == "str"
    return map_uniques(a, format_text), map_uniques(b, format_text), compatible


def _composite(series: Sequence[pd.Series]) -> pd.Series:
    """Row-wise composite key; a row with any missing part has no key."""
    key = series[0].astype("string")
    for extra in series[1:]:
        key = key + "\x1f" + extra.astype("string")
    return key.dropna()


def _looks_like_row_numbers(series: pd.Series, profile: TableProfile | None = None) -> bool:
    """Consecutive integers (1..N / 0..N-1): overlap between two such columns is guaranteed by
    construction and says nothing about a relationship.

    A random sample of such a column is NOT consecutive (a 2,000-row sample of 1..458,000 has gaps
    everywhere), so on a sampled table the check would miss it and report the row numbers as a
    1:1 join key. When ``profile`` says the frame is a sample of a larger table, the table's own
    footer statistics decide instead: the column starts at 0 or 1 and spans exactly the table's
    non-null row count, and the sample must hold no repeated value.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna()
    if len(values) < 3 or not (values % 1 == 0).all():
        return False
    distinct = values.nunique()
    if distinct != len(values):
        return False
    whole = profile.ranges.get(str(series.name)) if profile is not None and profile.rows > len(series) else None
    if whole is not None:
        low, high, nulls = whole
        return low in (0, 1) and high % 1 == 0 and (high - low + 1) == profile.rows - nulls
    low, high = values.min(), values.max()
    return low in (0, 1) and (high - low + 1) == distinct


def _has_fractions(series: pd.Series) -> bool:
    """A float column with non-integral values is a measurement, not an identifier."""
    if not pd.api.types.is_float_dtype(series):
        return False
    values = series.head(100_000).dropna().head(10_000)
    return bool(len(values)) and bool((values % 1 != 0).any())


def _percent(share: float) -> str:
    return f"{share * 100:.0f}%"


def measure_key(
    q: pd.DataFrame,
    r: pd.DataFrame,
    left_cols: Sequence[str],
    right_cols: Sequence[str],
    name_score: float,
    left_profile: TableProfile | None = None,
    right_profile: TableProfile | None = None,
) -> KeyEvidence:
    """Measure one (possibly composite) candidate key between ``q`` and ``r``.

    The profiles are the whole-table facts of each side when ``q`` / ``r`` are samples (see
    `_looks_like_row_numbers`); without them a frame is taken to be the whole table.
    """
    ev = KeyEvidence(tuple(left_cols), tuple(right_cols), round(name_score, 3), NO_OVERLAP)
    left_parts, right_parts, compatible = [], [], True
    try:
        for left_col, right_col in zip(left_cols, right_cols):
            left_norm, right_norm, ok = _normalise_pair(q[left_col], r[right_col])
            left_parts.append(left_norm)
            right_parts.append(right_norm)
            compatible = compatible and ok
    except TypeError:  # unhashable cells: lists, dicts and other nested values cannot be compared
        ev.notes.append("column holds nested values (lists or objects), which cannot be compared")
        return ev
    left_key, right_key = _composite(left_parts), _composite(right_parts)
    ev.measure_like = any(
        _has_fractions(q[lc]) and _has_fractions(r[rc]) for lc, rc in zip(left_cols, right_cols)
    )
    if ev.measure_like:
        ev.notes.append(
            "key column holds fractional numbers (looks like a measurement, not an identifier)"
        )
    ev.left_rows, ev.right_rows = len(left_key), len(right_key)
    if not compatible:
        ev.notes.append("dtype mismatch (compared as text)")
    if not len(left_key) or not len(right_key):
        ev.notes.append("key column is entirely null on one side")
        return ev

    left_counts, right_counts = left_key.value_counts(), right_key.value_counts()
    ev.left_distinct, ev.right_distinct = len(left_counts), len(right_counts)
    ev.left_unique_ratio = round(ev.left_distinct / ev.left_rows, 4)
    ev.right_unique_ratio = round(ev.right_distinct / ev.right_rows, 4)
    shared = left_counts.index.intersection(right_counts.index)
    ev.shared_distinct = len(shared)
    if not len(shared):
        return ev

    left_hits, right_hits = left_counts[shared], right_counts[shared]
    ev.left_match_rate = round(int(left_hits.sum()) / ev.left_rows, 4)
    ev.right_match_rate = round(int(right_hits.sum()) / ev.right_rows, 4)
    ev.inner_rows = int(
        (left_hits.to_numpy(dtype="int64") * right_hits.to_numpy(dtype="int64")).sum()
    )

    left_unique = ev.left_unique_ratio >= UNIQUE_RATIO
    right_unique = ev.right_unique_ratio >= UNIQUE_RATIO
    ev.cardinality = {
        (True, True): "1:1", (True, False): "1:N", (False, True): "N:1", (False, False): "N:N",
    }[(left_unique, right_unique)]

    if ev.left_distinct == 1 and ev.right_distinct == 1:
        ev.verdict = CONSTANT_KEY
        ev.notes.append(
            "key is single-valued on both sides -- the join is a cross product "
            "(every row paired with every row)"
        )
    elif (
        len(left_cols) == 1
        and _looks_like_row_numbers(q[left_cols[0]], left_profile)
        and _looks_like_row_numbers(r[right_cols[0]], right_profile)
    ):
        ev.verdict = SURROGATE_ID
        ev.notes.append(
            "both sides are consecutive row numbers -- overlap is guaranteed by construction "
            "and is not evidence of a relationship"
        )
    elif ev.thin and ev.shared_distinct <= COINCIDENCE_SHARED:
        ev.verdict = WEAK_OVERLAP
        best = max(ev.left_match_rate, ev.right_match_rate)
        ev.notes.append(
            f"only {ev.shared_distinct} shared key value(s) and {_percent(best)} of rows match "
            "-- indistinguishable from coincidence"
        )
    else:
        ev.verdict = JOIN
        if min(ev.left_distinct, ev.right_distinct) == 1:
            ev.notes.append(
                "key is single-valued on one side -- the join acts as a filter on the other "
                "side's matching rows, not a link between entities"
            )
        if ev.thin:
            ev.notes.append(
                f"thin overlap: {ev.shared_distinct:,} shared key values, few rows match; "
                "expected for a subset/intersection question, suspicious otherwise"
            )
        if ev.cardinality == "N:N":
            ev.notes.append(
                f"N:N -- inner join yields {ev.inner_rows:,} rows from {ev.left_rows:,} x "
                f"{ev.right_rows:,}; aggregate one side to the key first"
            )
    return ev


def _key_ness(ev: KeyEvidence) -> float:
    """How key-like the columns are, continuously: distinct/rows on the better side plus on the
    worse side (a 1:1 key scores ~2, a 1:N key ~1+, a categorical column ~0). No hard 'unique'
    cliff: an id that is 97% unique (a few repeated posts) is still an id."""
    high = max(ev.left_unique_ratio, ev.right_unique_ratio)
    low = min(ev.left_unique_ratio, ev.right_unique_ratio)
    return round(high + low, 2)


def _rank(ev: KeyEvidence) -> tuple:
    return (
        _TIER[ev.verdict],
        ev.measure_like,                 # measurements are almost never join keys
        min(ev.left_distinct, ev.right_distinct) == 1,  # a filter on one side, not a link between entities
        -_key_ness(ev),                  # identifies entities on one side, ideally both
        -round(max(ev.left_match_rate, ev.right_match_rate), 2),  # value evidence ...
        -ev.name_score,                  # ... before name similarity (identical names are everywhere)
        -ev.shared_distinct,
        len(ev.left_cols),
    )


def _has_key_evidence(ev: KeyEvidence) -> bool:
    return ev.verdict == JOIN and ev.unique_side and ev.shared_distinct >= 2 and not ev.measure_like


def _composite_search(
    q: pd.DataFrame, r: pd.DataFrame, pairs: Sequence[tuple[str, str, float]]
) -> list[KeyEvidence]:
    """Smallest composite keys (2..MAX_KEY_COLUMNS columns) over the name-similar pairs that are
    unique on a side AND share values.

    A subset with no shared values is pruned together with every superset (adding columns can only
    remove matches). There is no early stop: a 3-column key unique on ONE side can hide the
    4-column key that is unique on BOTH (benchmark merges use the latter; the former fans rows
    out). A composite is only proposed when it IS a key: crossing categorical columns raises the
    distinct ratio without adding a relationship, so a merely "more unique" composite must not
    displace its single-column parts.
    """
    pool = list(pairs[:COMPOSITE_POOL])
    dead: list[frozenset[int]] = []
    found: list[KeyEvidence] = []
    for size in range(2, min(MAX_KEY_COLUMNS, len(pool)) + 1):
        for combo in combinations(range(len(pool)), size):
            left_cols = [pool[i][0] for i in combo]
            right_cols = [pool[i][1] for i in combo]
            if len(set(left_cols)) < size or len(set(right_cols)) < size:
                continue
            members = frozenset(combo)
            if any(pruned <= members for pruned in dead):
                continue
            ev = measure_key(q, r, left_cols, right_cols, min(pool[i][2] for i in combo))
            if ev.shared_distinct == 0:
                dead.append(members)
                continue
            found.append(ev)
    return [ev for ev in found if _has_key_evidence(ev)]


def find_join_keys(
    q: pd.DataFrame,
    r: pd.DataFrame,
    coma_matches: Sequence[tuple[str, str, float]],
    left_profile: TableProfile | None = None,
    right_profile: TableProfile | None = None,
) -> list[KeyEvidence]:
    """Measure COMA's name-similar pairs (and composites when needed) and rank them, best first."""
    rows = len(q) + len(r)
    cap = max(MIN_SINGLE_CANDIDATES, min(MAX_SINGLE_CANDIDATES, CELL_BUDGET // max(rows, 1)))
    pairs = [m for m in coma_matches if m[2] >= SCHEMA_MATCH_THRESHOLD][:cap]
    singles = [
        measure_key(q, r, [a], [b], score, left_profile, right_profile) for a, b, score in pairs
    ]
    if not singles:
        return []
    results = list(singles)
    # composites are only worth their cost when no single column is already a real key
    if (
        not any(_has_key_evidence(ev) for ev in singles)
        and len(pairs) >= 2
        and rows <= COMPOSITE_MAX_ROWS
    ):
        results += _composite_search(q, r, pairs)
    return sorted(results, key=_rank)


# ------------------------------------------------------------------ report

MAX_REPORTED_KEYS = 3
SAMPLE_VALUES = 3
SAMPLE_VALUE_CHARS = 24


def _sample_values(series: pd.Series) -> str:
    values = series.dropna().astype(str).drop_duplicates().head(SAMPLE_VALUES)
    return ", ".join(
        "'" + (v if len(v) <= SAMPLE_VALUE_CHARS else v[: SAMPLE_VALUE_CHARS - 3] + "...") + "'"
        for v in values
    )


def _key_text(ev: KeyEvidence, left: str, right: str) -> str:
    return (
        f"{' + '.join(ev.left_cols)} ({left}) = {' + '.join(ev.right_cols)} ({right})"
    )


def _facts(ev: KeyEvidence) -> str:
    # left/right, not the file names: real names are ~75 characters and the tool output is budgeted
    return (
        f"{ev.cardinality}, {ev.shared_distinct:,} shared key values; "
        f"{_percent(ev.left_match_rate)} of the left table's rows and "
        f"{_percent(ev.right_match_rate)} of the right table's rows find a partner; "
        f"inner join = {ev.inner_rows:,} rows"
    )


def format_join_section(
    left: str,
    right: str,
    ranked: Sequence[KeyEvidence],
    q: pd.DataFrame,
    r: pd.DataFrame,
) -> str:
    """The join half of the `check_join_union` report: ranked candidates with measured facts."""
    if not ranked:
        return (
            f"NO JOIN: no column pair between '{left}' and '{right}' has similar names "
            f"(score >= {SCHEMA_MATCH_THRESHOLD}), so no values were compared. Only "
            "name-similar columns are tested; a join on differently named columns is not detected."
        )
    supported = [ev for ev in ranked if ev.supported][:MAX_REPORTED_KEYS]
    if supported:
        if any(ev.unique_side for ev in supported):
            lines = [f"JOIN: '{left}' joins '{right}'; candidate keys, best first:"]
        else:
            lines = [
                "JOIN (many-to-many only): the tables share values on these columns, but none "
                "identifies rows on either side; aggregate one side to the key before merging:"
            ]
        for number, ev in enumerate(supported, 1):
            lines.append(f"  {number}. {_key_text(ev, left, right)} -- {_facts(ev)}")
            lines.extend(f"     note: {note}" for note in ev.notes)
        return "\n".join(lines)

    lines = [
        f"NO JOIN: no name-similar key between '{left}' and '{right}' is backed by shared values "
        "(an inner join on these returns no useful rows; an outer join or a union needs no "
        "overlap). Only name-similar columns are tested."
    ]
    for ev in ranked[:MAX_REPORTED_KEYS]:
        lines.append(f"  - {_key_text(ev, left, right)}: {'; '.join(ev.notes) or 'no shared values'}")
        if ev.verdict == NO_OVERLAP and ev.shared_distinct == 0:
            left_sample = _sample_values(q[ev.left_cols[0]])
            right_sample = _sample_values(r[ev.right_cols[0]])
            lines.append(f"    values: {left} {left_sample} | {right} {right_sample}")
    return "\n".join(lines)
