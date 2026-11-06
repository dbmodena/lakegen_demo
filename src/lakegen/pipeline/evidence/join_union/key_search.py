"""Which pairs of columns are the join keys of two tables, ranked.

The candidate pairs come from Valentine (`schema_matching`); each is measured (`join_keys.measure_key`) and
the measured keys are ranked: the ones that identify rows first, the measurements and the single-valued
columns last, and a key the sandbox would stop sinks below the ones it lets through. When no single
column is a key, composite keys are searched; when the tables share two or more columns that only group
rows (a year, a district), those are also measured together as the grain they are joined on.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING

import pandas as pd

from lakegen.pipeline.evidence.join_union.join_keys import (
    JOIN,
    VERDICT_ORDER,
    KeyEvidence,
    add_raw_merge_facts,
    measure_key,
)
from lakegen.pipeline.evidence.join_union.matching.column_values import (
    looks_like_row_numbers,
    looks_numeric,
)
from lakegen.pipeline.evidence.join_union.matching.schema_matching import SCHEMA_MATCH_THRESHOLD
from lakegen.utils.merge_limits import EXPLODES

if TYPE_CHECKING:
    from lakegen.utils.table_io import TableProfile

# a pair matched on values alone (named differently) must share this many values, and this share of
# the rows of one side must find a partner
VALUE_MATCH_MIN_SHARED = 10
VALUE_MATCH_MIN_RATE = 0.8


MAX_KEY_COLUMNS = 4
MIN_SINGLE_CANDIDATES = 10
MAX_SINGLE_CANDIDATES = 60
CELL_BUDGET = 30_000_000  # ~ rows scanned across all measured candidates
COMPOSITE_POOL = 6  # name-similar pairs considered for composite keys
COMPOSITE_MAX_ROWS = 400_000  # skip the composite search on very large table pairs

# the report numbers this many keys, and records the plain-merge outcome of them
MAX_REPORTED_KEYS = 3


def _key_ness(ev: KeyEvidence) -> float:
    """How key-like the columns are, continuously.

    Distinct/rows on the better side plus on the worse side (a 1:1 key scores ~2, a 1:N key ~1+, a
    categorical column ~0). No hard 'unique' cliff: an id that is 97% unique (a few repeated posts)
    is still an id.

    Args:
        ev: The measured key.

    Returns:
        The sum of the two unique ratios, rounded to two decimals.
    """
    high = max(ev.left_unique_ratio, ev.right_unique_ratio)
    low = min(ev.left_unique_ratio, ev.right_unique_ratio)
    return round(high + low, 2)


def rank_keys(keys: Sequence[KeyEvidence]) -> list[KeyEvidence]:
    """The candidate keys, best first."""
    return sorted(keys, key=_rank)


def _rank(ev: KeyEvidence) -> tuple:
    return (
        ev.grain,  # the grain is shown on its own line, after the keys
        VERDICT_ORDER[ev.verdict],
        ev.positional,  # a join by position is the last resort among the supported keys
        ev.measure_like,  # measurements are almost never join keys
        min(ev.left_distinct, ev.right_distinct)
        == 1,  # a filter on one side, not a link between entities
        ev.size_class == EXPLODES,  # a merge the sandbox stops sinks below the ones it lets through
        -_key_ness(ev),  # identifies entities on one side, ideally both
        -round(max(ev.left_match_rate, ev.right_match_rate), 2),  # value evidence ...
        -ev.name_score,  # ... before name similarity (identical names are everywhere)
        -ev.shared_distinct,
        len(ev.left_cols),
    )


def _value_match_holds(
    ev: KeyEvidence,
    left: pd.Series,
    right: pd.Series,
    left_profile: TableProfile | None,
    right_profile: TableProfile | None,
) -> bool:
    """A pair matched on values alone is kept only when the values really are one domain.

    Valentine's value similarity is a lead, not a proof (two columns of small whole numbers resemble
    each other). The measured overlap decides: many shared values, most rows of one side finding a
    partner, neither column a row number or a measurement, and a number column an identifier.

    Args:
        ev: The measured key.
        left: The left key column.
        right: The right key column.
        left_profile: The footer facts of the left table, when it was sampled.
        right_profile: The footer facts of the right table, when it was sampled.

    Returns:
        Whether the match holds.
    """
    numeric = looks_numeric(left) and looks_numeric(right)
    return (
        ev.shared_distinct >= VALUE_MATCH_MIN_SHARED
        and max(ev.left_match_rate, ev.right_match_rate) >= VALUE_MATCH_MIN_RATE
        and not ev.measure_like
        and (ev.unique_side or not numeric)  # counts overlap in value; only an identifier is unique
        and not looks_like_row_numbers(left, left_profile)
        and not looks_like_row_numbers(right, right_profile)
    )


def _has_key_evidence(ev: KeyEvidence) -> bool:
    return ev.verdict == JOIN and ev.unique_side and ev.shared_distinct >= 2 and not ev.measure_like


def _composite_search(
    q: pd.DataFrame,
    r: pd.DataFrame,
    pairs: Sequence[tuple[str, str, float]],
    left_total: int | None = None,
    right_total: int | None = None,
) -> list[KeyEvidence]:
    """Smallest composite keys (2..MAX_KEY_COLUMNS columns) over the name-similar pairs.

    Only keys that are unique on a side AND share values are kept. A subset with no shared values is pruned together with every superset (adding columns can only
    remove matches). There is no early stop: a 3-column key unique on ONE side can hide the
    4-column key that is unique on BOTH (benchmark merges use the latter; the former fans rows
    out). A composite is only proposed when it IS a key: crossing categorical columns raises the
    distinct ratio without adding a relationship, so a merely "more unique" composite must not
    displace its single-column parts.

    Args:
        q: The left table, or a sample of it.
        r: The right table, or a sample of it.
        pairs: The name-similar (left, right, score) column pairs, best first.
        left_total: The rows of the whole left table, when ``q`` is a sample.
        right_total: The rows of the whole right table, when ``r`` is a sample.

    Returns:
        The composite keys with key evidence.
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
            ev = measure_key(
                q,
                r,
                left_cols,
                right_cols,
                min(pool[i][2] for i in combo),
                left_total=left_total,
                right_total=right_total,
            )
            if ev.shared_distinct == 0:
                dead.append(members)
                continue
            found.append(ev)
    return [ev for ev in found if _has_key_evidence(ev)]


# a foreign key proposed by the names is kept when most rows of one side find a partner in the other
FOREIGN_KEY_MIN_MATCH = 0.9
FOREIGN_KEY_MIN_SHARED = 2


def foreign_key_holds(ev: KeyEvidence) -> bool:
    """Whether the values confirm a foreign key the names suggested."""
    return (
        ev.verdict == JOIN
        and max(ev.left_match_rate, ev.right_match_rate) >= FOREIGN_KEY_MIN_MATCH
        and ev.shared_distinct >= FOREIGN_KEY_MIN_SHARED
        and not ev.measure_like
    )


def settle_foreign_keys(ranked: Sequence[KeyEvidence]) -> list[KeyEvidence]:
    """Drop the keys the names proposed that the values do not confirm (call after the whole-table count)."""
    return [ev for ev in ranked if not ev.by_cue or foreign_key_holds(ev)]


def find_join_keys(
    q: pd.DataFrame,
    r: pd.DataFrame,
    coma_matches: Sequence[tuple[str, str, float]],
    left_profile: TableProfile | None = None,
    right_profile: TableProfile | None = None,
    left_total: int | None = None,
    right_total: int | None = None,
    cue_pairs: Sequence[tuple[str, str]] = (),
) -> list[KeyEvidence]:
    """Measure COMA's name-similar pairs (and composites when needed) and rank them, best first.

    ``left_total`` / ``right_total`` are the rows of the whole tables when ``q`` / ``r`` are samples.
    ``cue_pairs`` are column pairs whose names suggest a foreign key; they are measured too and flagged
    ``by_cue``, and ``settle_foreign_keys`` keeps them only once the values confirm them.

    Args:
        q: The left table, or a sample of it.
        r: The right table, or a sample of it.
        coma_matches: Valentine's (left, right, score) column matches.
        left_profile: The footer facts of the left table, when ``q`` is a sample.
        right_profile: The footer facts of the right table, when ``r`` is a sample.
        left_total: The rows of the whole left table, when ``q`` is a sample.
        right_total: The rows of the whole right table, when ``r`` is a sample.
        cue_pairs: Column pairs whose names suggest a foreign key.

    Returns:
        The measured keys, best first; empty when no pair reaches the threshold.
    """
    rows = len(q) + len(r)
    cap = max(MIN_SINGLE_CANDIDATES, min(MAX_SINGLE_CANDIDATES, CELL_BUDGET // max(rows, 1)))
    candidates = [m for m in coma_matches if m[2] >= SCHEMA_MATCH_THRESHOLD][:cap]
    named = {(m[0], m[1]) for m in candidates}
    cue_only = [
        (a, b) for a, b in cue_pairs if (a, b) not in named and a in q.columns and b in r.columns
    ]
    pairs: list[tuple[str, str, float]] = []
    singles: list[KeyEvidence] = []
    for left, right in cue_only:
        ev = measure_key(
            q,
            r,
            [left],
            [right],
            SCHEMA_MATCH_THRESHOLD,
            left_profile,
            right_profile,
            left_total,
            right_total,
        )
        ev.by_cue = True
        singles.append(
            ev
        )  # not a composite or grain candidate: the sample may have missed its values
    for match in candidates:
        left, right, score = match[0], match[1], match[2]
        ev = measure_key(
            q, r, [left], [right], score, left_profile, right_profile, left_total, right_total
        )
        if getattr(match, "by_values", False):
            if not _value_match_holds(ev, q[left], r[right], left_profile, right_profile):
                continue
            ev.by_values = True
        pairs.append((left, right, score))
        singles.append(ev)
    if not singles:
        return []
    results = list(singles)
    singles = [ev for ev in singles if not ev.by_cue]
    # composites are only worth their cost when no single column is already a real key
    if (
        not any(_has_key_evidence(ev) for ev in singles)
        and len(pairs) >= 2
        and rows <= COMPOSITE_MAX_ROWS
    ):
        results += _composite_search(q, r, pairs, left_total, right_total)
    grain = _grain_key(
        q,
        r,
        [ev for ev in results if not ev.by_cue],
        left_profile,
        right_profile,
        left_total,
        right_total,
    )
    if grain is not None:
        results.append(grain)
    ranked = rank_keys(results)
    add_raw_facts_to_shown(ranked, q, r)
    return ranked


def add_raw_facts_to_shown(ranked: Sequence[KeyEvidence], q: pd.DataFrame, r: pd.DataFrame) -> None:
    """Record the plain-merge outcome of the supported keys the report numbers (and of the grain)."""
    shown = [ev for ev in ranked if ev.supported and not ev.grain][:MAX_REPORTED_KEYS]
    shown += [ev for ev in ranked if ev.grain and ev.supported]
    for ev in shown:
        add_raw_merge_facts(ev, q, r)


def _is_dimension(ev: KeyEvidence) -> bool:
    """A column both tables repeat (a year, a district, a category): it groups rows, it identifies none."""
    return (
        ev.verdict == JOIN
        and len(ev.left_cols) == 1
        and ev.cardinality == "N:N"
        and not ev.measure_like
        and ev.shared_distinct >= 2
    )


def _grain_key(
    q: pd.DataFrame,
    r: pd.DataFrame,
    measured: Sequence[KeyEvidence],
    left_profile: TableProfile | None,
    right_profile: TableProfile | None,
    left_total: int | None,
    right_total: int | None,
) -> KeyEvidence | None:
    """The shared dimension columns taken together, when there are at least two of them.

    Two statistical tables are joined on the grain they share (year and sex and district), after each
    is aggregated to it. No single dimension names that grain and no composite of them is a key, so
    they are measured together: the report then states how many combinations the tables share, how
    large the raw merge on them is, and how many rows the aggregated join has.

    Args:
        q: The left table, or a sample of it.
        r: The right table, or a sample of it.
        measured: The keys measured so far.
        left_profile: The footer facts of the left table, when ``q`` is a sample.
        right_profile: The footer facts of the right table, when ``r`` is a sample.
        left_total: The rows of the whole left table, when ``q`` is a sample.
        right_total: The rows of the whole right table, when ``r`` is a sample.

    Returns:
        The grain as a key flagged ``grain``, or None when there are fewer than two dimensions, the
        composite search already measured them, or they do not join.
    """
    dims = [ev for ev in measured if _is_dimension(ev)]
    if len(dims) < 2:
        return None
    dims.sort(key=lambda ev: (-min(ev.left_match_rate, ev.right_match_rate), ev.shared_distinct))
    chosen = dims[:MAX_KEY_COLUMNS]
    left_cols = tuple(ev.left_cols[0] for ev in chosen)
    right_cols = tuple(ev.right_cols[0] for ev in chosen)
    if any((ev.left_cols, ev.right_cols) == (left_cols, right_cols) for ev in measured):
        return None  # the composite search already found exactly this key
    grain = measure_key(
        q,
        r,
        left_cols,
        right_cols,
        min(ev.name_score for ev in chosen),
        left_profile,
        right_profile,
        left_total,
        right_total,
    )
    grain.grain = True
    return grain if grain.verdict == JOIN else None
