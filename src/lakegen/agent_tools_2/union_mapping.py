"""Tiered union evidence for a pair of tables.

Valentine paper: two relations are UNIONABLE when they have the same arity and a 1-1 mapping of
semantically equivalent attributes; VIEW-UNIONABLE when projections of them are. The old rule
(average COMA score of the returned pairs >= 0.5) never checked how many attributes actually map,
so unrelated tables sharing a few name-similar columns were called unions.

This module builds a 1-1 mapping from COMA's name-similar pairs among the tables' REAL columns and
decides by how much of each table the mapping covers:

  UNION          >= 80% of BOTH tables' columns map (the same-arity mapping), >= 2 columns
  SUBSET UNION   >= 80% of the SMALLER table's columns map, the larger has extra columns, >= 3
  PARTIAL UNION  >= 50% of BOTH tables' columns map, >= 3 columns
  NO UNION       otherwise

Values do not decide anything. A value-domain veto was tried and rejected: it discarded 22-42% of
genuine unions, because free-text columns from different sources (department names, job titles)
share neither vocabulary nor format. Values only add a WARNING to a mapped pair when the pair would
corrupt a concat: a text-vs-number clash, a magnitude clash, or an empty column. Unlike a join, a
union needs no shared rows.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lakegen.agent_tools_2.column_values import as_number, column_kind, map_uniques, mostly_numeric
from lakegen.agent_tools_2.schema_matching import SCHEMA_MATCH_THRESHOLD

UNION = "UNION"
SUBSET_UNION = "SUBSET_UNION"
PARTIAL_UNION = "PARTIAL_UNION"
NO_UNION = "NO_UNION"

UNION_COVERAGE = 0.8           # of both tables' columns
SUBSET_COVERAGE = 0.8          # of the smaller table's columns
PARTIAL_COVERAGE = 0.5         # of both tables' columns
UNION_MIN_COLUMNS = 2
PARTIAL_MIN_COLUMNS = 3        # SUBSET and PARTIAL

# A numeric pair is only warned about when its ranges do not overlap AND its medians differ by
# more than an order of magnitude (max of range IoU and median ratio below this).
MIN_NUMERIC_SIMILARITY = 0.1
VALUE_SAMPLE = 5000

# Header-parsing artefacts, not attributes: '', 'Unnamed: 3', '_duplicated_0', 'Column1', 'Field2'.
_PLACEHOLDER = re.compile(r"^(|unnamed[: ]*\d*|_duplicated_\d+|column\d+|field\d+)$", re.IGNORECASE)
_KIND_NAMES = {"num": "number", "str": "text", "dt": "date"}


@dataclass
class MappedColumn:
    left: str
    right: str
    name_score: float
    warning: str = ""      # non-empty only when a concat of this pair would mix or corrupt values


@dataclass
class UnionMapping:
    aligned: list[MappedColumn]
    n_left: int            # real (non-placeholder) columns
    n_right: int
    placeholders: int = 0  # placeholder header columns ignored on both sides
    verdict: str = NO_UNION
    cov_left: float = 0.0
    cov_right: float = 0.0
    warnings: list[MappedColumn] = field(default_factory=list)

    @property
    def cov_large(self) -> float:
        """Share of the LARGER table's columns mapped (the smaller of the two coverages)."""
        return min(self.cov_left, self.cov_right)

    @property
    def cov_small(self) -> float:
        """Share of the SMALLER table's columns mapped."""
        return max(self.cov_left, self.cov_right)

    @property
    def supported(self) -> bool:
        return self.verdict != NO_UNION


def real_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are attributes: not '', 'Unnamed: 3', '_duplicated_0' (header artefacts)."""
    return [c for c in df.columns if not _PLACEHOLDER.match(str(c).strip())]


def _kind(series: pd.Series) -> str:
    kind = column_kind(series)
    return "num" if kind == "str" and mostly_numeric(series) else kind


def _numbers(series: pd.Series) -> np.ndarray:
    values = series.dropna()
    if len(values) > VALUE_SAMPLE:
        values = values.sample(VALUE_SAMPLE, random_state=0)
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        return pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    parsed = map_uniques(values, as_number)
    return pd.to_numeric(parsed, errors="coerce").dropna().to_numpy(dtype=float)


def _numeric_warning(a: pd.Series, b: pd.Series) -> str:
    x, y = _numbers(a), _numbers(b)
    if not len(x) or not len(y):
        return "column is empty on one side"
    (x_low, x_mid, x_high), (y_low, y_mid, y_high) = (
        np.percentile(x, [5, 50, 95]),
        np.percentile(y, [5, 50, 95]),
    )
    overlap = min(x_high, y_high) - max(x_low, y_low)
    span = max(x_high, y_high) - min(x_low, y_low)
    if span > 0 and overlap >= 0:
        iou = overlap / span
    else:
        iou = 1.0 if x_low == x_high == y_low == y_high else 0.0
    if x_mid == y_mid:
        ratio = 1.0
    elif x_mid * y_mid > 0:
        ratio = min(abs(x_mid), abs(y_mid)) / max(abs(x_mid), abs(y_mid))
    else:
        ratio = 0.0
    if max(iou, ratio) >= MIN_NUMERIC_SIMILARITY:
        return ""
    return (
        f"different magnitude (value ranges overlap {iou:.0%}; "
        f"medians {x_mid:,.4g} vs {y_mid:,.4g})"
    )


def _pair_warning(a: pd.Series, b: pd.Series) -> str:
    """A note worth showing on a MAPPED pair -- only what would bite a concat."""
    if not a.notna().any() or not b.notna().any():
        return "column is empty on one side"
    kind_a, kind_b = _kind(a), _kind(b)
    if kind_a != kind_b:
        return (
            f"type clash: {_KIND_NAMES[kind_a]} vs {_KIND_NAMES[kind_b]} "
            "-- a concat will mix types"
        )
    if kind_a == "num":
        return _numeric_warning(a, b)
    return ""


def union_by_names(
    q: pd.DataFrame,
    r: pd.DataFrame,
    coma_matches: Sequence[tuple[str, str, float]],
) -> UnionMapping:
    """1-1 mapping by name similarity among REAL columns, then a coverage tier."""
    real_q, real_r = set(real_columns(q)), set(real_columns(r))
    candidates = sorted(
        (
            m for m in coma_matches
            if m[2] >= SCHEMA_MATCH_THRESHOLD and m[0] in real_q and m[1] in real_r
        ),
        key=lambda m: -m[2],
    )
    used_left: set[str] = set()
    used_right: set[str] = set()
    aligned: list[MappedColumn] = []
    for left, right, score in candidates:
        if left in used_left or right in used_right:
            continue
        aligned.append(MappedColumn(left, right, score, _pair_warning(q[left], r[right])))
        used_left.add(left)
        used_right.add(right)

    mapping = UnionMapping(
        aligned=aligned,
        n_left=len(real_q),
        n_right=len(real_r),
        placeholders=(q.shape[1] - len(real_q)) + (r.shape[1] - len(real_r)),
        warnings=[m for m in aligned if m.warning],
    )
    mapping.cov_left = len(aligned) / len(real_q) if real_q else 0.0
    mapping.cov_right = len(aligned) / len(real_r) if real_r else 0.0
    count = len(aligned)
    if count >= UNION_MIN_COLUMNS and mapping.cov_large >= UNION_COVERAGE:
        mapping.verdict = UNION
    elif count >= PARTIAL_MIN_COLUMNS and mapping.cov_small >= SUBSET_COVERAGE:
        mapping.verdict = SUBSET_UNION
    elif count >= PARTIAL_MIN_COLUMNS and mapping.cov_large >= PARTIAL_COVERAGE:
        mapping.verdict = PARTIAL_UNION
    return mapping


# ------------------------------------------------------------------ report

MAX_REPORTED_PAIRS = 8
MAX_REPORTED_WARNINGS = 4


def _percent(share: float) -> str:
    return f"{share * 100:.0f}%"


def format_union_section(left: str, right: str, mapping: UnionMapping) -> str:
    """The union half of the `check_join_union` report."""
    count = len(mapping.aligned)
    if mapping.verdict == UNION:
        head = (
            f"UNION: '{left}' unions with '{right}'; {count} columns map 1-1 "
            f"({_percent(mapping.cov_left)} of the left table's columns, "
            f"{_percent(mapping.cov_right)} of the right's)."
        )
    elif mapping.verdict == SUBSET_UNION:
        smaller = left if mapping.cov_left >= mapping.cov_right else right
        larger = right if smaller == left else left
        extra = abs(mapping.n_left - mapping.n_right)
        head = (
            f"SUBSET UNION: '{left}' and '{right}' union on {count} columns -- every column of the "
            f"smaller table '{smaller}' maps ({_percent(mapping.cov_small)}), and '{larger}' has "
            f"{extra} more (a concat fills them with missing values)."
        )
    elif mapping.verdict == PARTIAL_UNION:
        head = (
            f"PARTIAL UNION: only {count} columns map 1-1 ({_percent(mapping.cov_left)} of the left "
            f"table's columns, {_percent(mapping.cov_right)} of the right's); union them on these "
            "columns only."
        )
    else:
        head = (
            f"NO UNION: only {count} of {mapping.n_left} / {mapping.n_right} columns of '{left}' / "
            f"'{right}' map by name (a union needs >= 80% of one table's columns, or >= 50% of both "
            "with >= 3 columns)."
        )
    lines = [head]
    lines.extend(f"  {m.left} -> {m.right}" for m in mapping.aligned[:MAX_REPORTED_PAIRS])
    if count > MAX_REPORTED_PAIRS:
        lines.append(f"  ... and {count - MAX_REPORTED_PAIRS} more")
    lines.extend(
        f"  warning: {m.left} -> {m.right}: {m.warning}"
        for m in mapping.warnings[:MAX_REPORTED_WARNINGS]
    )
    if mapping.placeholders:
        lines.append(
            f"  ignored {mapping.placeholders} placeholder header column(s) "
            "(empty, 'Unnamed: N' or '_duplicated_N' names): not real attributes."
        )
    return "\n".join(lines)
