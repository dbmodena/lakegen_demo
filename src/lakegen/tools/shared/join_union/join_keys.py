"""Measured join-key evidence for a pair of tables.

Valentine's COMA proposes column pairs, and identical names score 1.0 everywhere in open data (`Date`,
`Name`, `Parent Department`), so a match alone says almost nothing about whether two tables join. Each
candidate pair is therefore measured against the actual values, and the evidence states facts instead
of asserting one key:

  * do the columns share any values (after strip/casefold, int-vs-float and numeric-text
    normalisation)?
  * is the key single-valued on both sides (a cross product), or just 1..N row numbers on both
    sides (overlap guaranteed by construction)?
  * the cardinality (1:1, 1:N, N:1, N:N) and the inner, left and outer join row counts, scaled to the
    whole tables when the frames measured are samples, with the size class of the merge (EMPTY, SAFE,
    FANOUT, or EXPLODES for the merges the sandbox stops), so an N:N join is flagged "aggregate first"
    instead of being silently accepted;
  * what a plain `pd.merge` on the columns as stored would do (return nothing, raise, or match the
    missing keys with each other).

`key_search` ranks the candidates and finds composite keys, `full_scan` counts the best ones over every
row, `join_report` writes them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from lakegen.domain.merge_limits import (
    EXPLODES,
    FANOUT,
    MERGE_BLOWUP_FACTOR,
    classify_merge,
)
from lakegen.tools.shared.join_union.matching.column_values import (
    column_kind,
    format_datetime,
    format_number,
    format_text,
    has_fractions,
    looks_like_row_numbers,
    map_uniques,
    mostly_numeric,
)

if TYPE_CHECKING:
    from lakegen.integrations.table_io import TableProfile

# distinct/non-null rows at or above this count a side as "unique" for the cardinality label
UNIQUE_RATIO = 0.99
# a match is "thin" when neither side has this share of its key rows finding a partner
MIN_MATCH_RATE = 0.5
# this many shared values or fewer, with a thin match, is indistinguishable from chance
COINCIDENCE_SHARED = 2


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
VERDICT_ORDER = {JOIN: 0, WEAK_OVERLAP: 1, SURROGATE_ID: 2, CONSTANT_KEY: 3, NO_OVERLAP: 4}


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
    inner_rows: int = 0           # inner join rows among the rows measured
    measure_like: bool = False    # a key column holds fractional numbers (amounts, coordinates)
    notes: list[str] = field(default_factory=list)
    # Sizes over the WHOLE tables. When the frames measured are samples they are scaled up from the
    # sample (an unbiased estimate of the join size); ``exact`` is True when every row was counted.
    left_total: int = 0           # rows of the whole left table
    right_total: int = 0
    inner_total: int = 0          # rows of ``left.merge(right, on=key)``
    left_join_total: int = 0      # rows of ``how="left"``
    outer_join_total: int = 0     # rows of ``how="outer"``
    exact: bool = True
    by_values: bool = False       # the two columns are named differently; Valentine matched their values
    grain: bool = False           # all the shared dimension columns together (not a ranked candidate)
    by_cue: bool = False          # proposed by the column names (a foreign key), then confirmed on the values
    # What ``left.merge(right, left_on=..., right_on=...)`` does on the columns as they are stored,
    # without the normalising this module applies (strip, casefold, numbers read from text).
    raw_checked: bool = False
    raw_inner_total: int | None = None   # None: pandas refuses (text against numbers)

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

    @property
    def grouped_rows(self) -> int:
        """Rows of the join once each table is aggregated to one row per key value."""
        return self.shared_distinct

    @property
    def size_class(self) -> str:
        """EMPTY, SAFE, FANOUT or EXPLODES for the inner join of the whole tables on this key."""
        return classify_merge(self.inner_total, self.left_total, self.right_total)

    @property
    def fanout(self) -> float:
        """Inner join rows per row of the larger table."""
        return self.inner_total / max(self.left_total, self.right_total, 1)


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


def percent(share: float) -> str:
    return f"{share * 100:.0f}%"


def measure_key(
    q: pd.DataFrame,
    r: pd.DataFrame,
    left_cols: Sequence[str],
    right_cols: Sequence[str],
    name_score: float,
    left_profile: TableProfile | None = None,
    right_profile: TableProfile | None = None,
    left_total: int | None = None,
    right_total: int | None = None,
) -> KeyEvidence:
    """Measure one (possibly composite) candidate key between ``q`` and ``r``.

    When ``q`` / ``r`` are samples of larger tables, ``left_total`` / ``right_total`` are the rows of
    the whole tables and the profiles their footer facts (see `looks_like_row_numbers`): the join
    sizes are then scaled up from the sample. Without them a frame is taken to be the whole table.
    """
    ev = KeyEvidence(tuple(left_cols), tuple(right_cols), round(name_score, 3), NO_OVERLAP)
    n_left, n_right = len(q), len(r)
    ev.left_total, ev.right_total = max(left_total or 0, n_left), max(right_total or 0, n_right)
    ev.exact = ev.left_total == n_left and ev.right_total == n_right
    ev.left_join_total, ev.outer_join_total = ev.left_total, ev.left_total + ev.right_total
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
        has_fractions(q[lc]) and has_fractions(r[rc]) for lc, rc in zip(left_cols, right_cols)
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
    # float64: a key shared by millions of rows on both sides overflows an int64 product
    ev.inner_rows = int(
        round((left_hits.to_numpy(dtype="float64") * right_hits.to_numpy(dtype="float64")).sum())
    )
    scale_left, scale_right = ev.left_total / max(n_left, 1), ev.right_total / max(n_right, 1)
    ev.inner_total = round(ev.inner_rows * scale_left * scale_right)
    ev.left_join_total = round(
        ev.inner_total + ev.left_total - float(left_hits.sum()) * scale_left
    )
    ev.outer_join_total = round(
        ev.left_join_total + ev.right_total - float(right_hits.sum()) * scale_right
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
        and looks_like_row_numbers(q[left_cols[0]], left_profile)
        and looks_like_row_numbers(r[right_cols[0]], right_profile)
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
            f"only {ev.shared_distinct} shared key value(s) and {percent(best)} of rows match "
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
        _size_notes(ev)
    return ev


def rows_text(ev: KeyEvidence, count: int) -> str:
    """A row count, marked ``~`` when it was scaled up from a sample."""
    return f"{count:,}" if ev.exact else f"~{count:,}"


def _size_notes(ev: KeyEvidence) -> None:
    """What merging the whole tables on this key would do, when it is more than a lookup."""
    size = ev.size_class
    if size == EXPLODES:
        ev.notes.append(
            f"EXPLODES: {ev.fanout:,.0f}x the larger table, and the sandbox stops a merge above "
            f"{MERGE_BLOWUP_FACTOR}x. Aggregate both tables to one row per key first: that join has "
            f"{ev.grouped_rows:,} rows."
        )
    elif ev.cardinality == "N:N":
        ratio = f" ({ev.fanout:.1f}x the larger table)" if ev.fanout >= 1.0 else ""
        ev.notes.append(f"N:N -- the key repeats on both sides{ratio}; aggregate one side to the key first")
    elif size == FANOUT:
        ev.notes.append(
            f"fan-out: {ev.fanout:.1f}x the larger table; the rows of the one side repeat"
        )


def raw_merge_rows(
    q: pd.DataFrame, r: pd.DataFrame, left_cols: Sequence[str], right_cols: Sequence[str]
) -> int | None:
    """Rows of the inner merge of ``q`` and ``r`` on these columns exactly as stored; None if pandas refuses.

    `pd.merge` compares text exactly (case and spaces count), refuses to merge text with numbers, and
    matches a missing key with every other missing key. A key that matches once normalised can
    therefore still return nothing, raise, or return far more rows in the generated code.
    """
    try:  # one row each is enough for pandas to compare the dtypes
        q[list(left_cols)].head(1).merge(r[list(right_cols)].head(1), left_on=list(left_cols), right_on=list(right_cols))
    except (ValueError, TypeError):
        return None
    left, right = q[list(left_cols)], r[list(right_cols)]
    if left.empty or right.empty:
        return 0
    try:
        left_counts = _key_counts(left)
        right_counts = _key_counts(right)
        shared = left_counts.index.intersection(right_counts.index)
    except TypeError:  # unhashable cells
        return None
    return int(round((left_counts[shared].to_numpy("float64") * right_counts[shared].to_numpy("float64")).sum()))


def _key_counts(keys: pd.DataFrame) -> pd.Series:
    """Rows per key value, missing values counted as a value of their own (as `pd.merge` does)."""
    return keys.iloc[:, 0].value_counts(dropna=False) if keys.shape[1] == 1 else keys.value_counts(dropna=False)


def _plain_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def add_raw_merge_facts(ev: KeyEvidence, q: pd.DataFrame, r: pd.DataFrame) -> None:
    """Record what a plain merge on this key returns, scaled to the whole tables like the other sizes."""
    ev.raw_checked = True
    columns = list(zip(ev.left_cols, ev.right_cols))
    if all(
        _plain_number(q[lc]) and _plain_number(r[rc]) and not q[lc].isna().any() and not r[rc].isna().any()
        for lc, rc in columns
    ):
        ev.raw_inner_total = ev.inner_total  # numbers compare as numbers, normalised or not
        return
    rows = raw_merge_rows(q, r, ev.left_cols, ev.right_cols)
    if rows is None:
        ev.raw_inner_total = None
        return
    scale = (ev.left_total / max(len(q), 1)) * (ev.right_total / max(len(r), 1))
    ev.raw_inner_total = round(rows * scale)
