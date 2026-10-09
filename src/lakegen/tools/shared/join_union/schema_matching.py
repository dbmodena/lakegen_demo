"""Valentine column matching: the column pairs that join and union evidence start from.

COMA scores a pair of columns on two grounds: ``NameCM`` is the similarity of the two names, ``InstancesCM``
the TF-IDF similarity of their cell values. The names settle most pairs (identical names score 1.0), but a
column renamed `zona_2001`, `pcd` or `col_3` can only be recognised by its values. Matching therefore runs
in two passes:

  1. every column pair is scored on its names (cheap: 0.03 s for a 10-column table, a few seconds for 250);
  2. the columns that found no name partner at or above the threshold, and whose values can carry the
     evidence, are scored on their values against each other. A value score of 0.56 or more passes the
     threshold on its own (``VALUE_EVIDENCE_WEIGHT`` * 0.56 = 0.5).

Scoring values for every pair was tried and rejected: it costs seconds on wide tables (a quadratic number of
TF-IDF comparisons: 153 s for two 250-column tables) and proposes a dozen coincidental pairs for two
unrelated tables, where this restriction proposes about one.

Measured on 220 real table pairs from the OrQa lakes (the right table's column names perturbed, the truth
being the key columns of executed reference solutions and the identical names of concatenated tables),
against name-only matching, as the share of true partners that reach the threshold:

  * join keys: unchanged names 99%, abbreviated or misspelled 88% -> 93%, opaque names (`col_3`)
    1% -> 67%, a mix of all of them 78% -> 94%;
  * unionable tables: the mix 74% -> 80%;
  * time per pair: median 0.03 s -> 0.05 s, slowest 0.15 s -> 0.36 s.

A match is a candidate, not a verdict. A COMA score is no evidence that two tables join or union (identical
names score 1.0 everywhere in open data: `Date`, `Name`, `Parent Department`); `join_keys` measures the
values of each candidate pair and `union_mapping` measures how much of each table maps.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from collections.abc import Mapping

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

from lakegen.tools.shared.join_union.column_values import can_match_on_values

logger = logging.getLogger(__name__)

# Valentine reads dates such as "Thu Jan 01 00:00:00 EST 2009" through dateutil, which warns about every
# time zone name it does not know; the warning says nothing the tool could act on.
warnings.filterwarnings("ignore", message=r"tzname .* identified but not understood")

# A column pair is a candidate for a join key / union mapping at or above this combined score.
SCHEMA_MATCH_THRESHOLD = 0.5
# Rows given to COMA. A random sample: the first rows of a sorted table show one corner of its values.
COMA_SAMPLE_ROWS = 1000
# COMA keeps a pair only when its score is within this share of the best score of BOTH its columns.
COMA_DELTA = 0.15
# Weight of a value match against a name match.
VALUE_EVIDENCE_WEIGHT = 0.9
# Columns per table scored on their values (the cost is quadratic in it); the most varied are kept.
MAX_VALUE_COLUMNS = 40
# Long text cells (WKT geometries run to 16,000 characters) cost COMA seconds and add no evidence.
MAX_CELL_CHARS = 64


class ColumnMatch(tuple):
    """``(left, right, score)`` plus the two parts the score comes from.

    It unpacks and compares as the three-tuple, so code written for plain ``(left, right, score)``
    triples takes it unchanged. ``name_score`` and ``instance_score`` are None when not reported.
    """

    name_score: float | None
    instance_score: float | None

    def __new__(
        cls,
        left: str,
        right: str,
        score: float,
        name_score: float | None = None,
        instance_score: float | None = None,
    ) -> ColumnMatch:
        match = super().__new__(cls, (left, right, score))
        match.name_score = name_score
        match.instance_score = instance_score
        return match

    @property
    def by_values(self) -> bool:
        """Passes the threshold on its values although the two names are not similar enough."""
        return self.name_score is not None and self.name_score < SCHEMA_MATCH_THRESHOLD <= self[2]


def _bounded_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """At most ``COMA_SAMPLE_ROWS`` random rows, text cells cut to ``MAX_CELL_CHARS``, names as text."""
    sample = (
        frame.sample(COMA_SAMPLE_ROWS, random_state=0) if len(frame) > COMA_SAMPLE_ROWS else frame.copy()
    )
    sample.columns = [str(column) for column in sample.columns]
    for column in sample.columns:
        if pd.api.types.is_string_dtype(sample[column]):
            sample[column] = sample[column].str.slice(0, MAX_CELL_CHARS)
    return sample


def _run_coma(q: pd.DataFrame, r: pd.DataFrame, *, use_instances: bool) -> Mapping:
    return valentine_match(
        [q, r],
        Coma(use_instances=use_instances, delta=COMA_DELTA),
        instance_sample_size=COMA_SAMPLE_ROWS,
    )


def _free_columns(sample: pd.DataFrame, best_name: Mapping[str, float]) -> list[str]:
    """The columns without a name partner whose values can carry the evidence, the most varied first."""
    free = [
        (int(sample[column].nunique(dropna=True)), column)
        for column in sample.columns
        if best_name[column] < SCHEMA_MATCH_THRESHOLD and can_match_on_values(sample[column])
    ]
    free.sort(key=lambda item: -item[0])
    return [column for _, column in free[:MAX_VALUE_COLUMNS]]


def _value_matches(
    sample_q: pd.DataFrame, sample_r: pd.DataFrame, columns_q: list[str], columns_r: list[str]
) -> dict[tuple[str, str], float]:
    """COMA's value similarity for the given columns of the two samples."""
    try:
        raw = _run_coma(sample_q[columns_q], sample_r[columns_r], use_instances=True)
    except Exception:  # noqa: BLE001 - the value pass is an addition; the name pass already stands
        logger.warning("COMA value matching failed; matching on names only", exc_info=True)
        return {}
    details = getattr(raw, "details", None) or {}
    return {
        (str(pair.source_column), str(pair.target_column)): float(
            details.get(pair, {}).get("InstancesCM", 0.0)
        )
        for pair in raw
    }


def match_columns(q: pd.DataFrame, r: pd.DataFrame) -> dict[tuple[str, str], ColumnMatch]:
    """Score the column correspondences between ``q`` and ``r`` with Valentine's COMA.

    ``delta=0`` keeps every pair with a positive score instead of only the best partner of each column:
    the pairs are measured against the values afterwards, and the second-best name can be the real key
    (`ID` / `Parent_ID`).
    """
    sample_q, sample_r = _bounded_sample(q), _bounded_sample(r)
    matches: dict[tuple[str, str], ColumnMatch] = {}
    best_q: defaultdict[str, float] = defaultdict(float)
    best_r: defaultdict[str, float] = defaultdict(float)
    for pair, score in _run_coma(sample_q, sample_r, use_instances=False).items():
        left, right = str(pair.source_column), str(pair.target_column)
        matches[(left, right)] = ColumnMatch(left, right, float(score), float(score))
        best_q[left] = max(best_q[left], float(score))
        best_r[right] = max(best_r[right], float(score))
    free_q, free_r = _free_columns(sample_q, best_q), _free_columns(sample_r, best_r)
    if free_q and free_r:
        for (left, right), similarity in _value_matches(sample_q, sample_r, free_q, free_r).items():
            score = VALUE_EVIDENCE_WEIGHT * similarity
            if score < SCHEMA_MATCH_THRESHOLD:
                continue
            named = matches.get((left, right))
            matches[(left, right)] = ColumnMatch(
                left, right, score, named.name_score if named else 0.0, similarity
            )
    return matches


def match_pair(q: pd.DataFrame, r: pd.DataFrame) -> list[ColumnMatch]:
    """The candidate column pairs of a table pair (score at or above the threshold), best first."""
    ranked: list[ColumnMatch] = []
    for (left, right), match in match_columns(q, r).items():
        if isinstance(match, ColumnMatch):
            candidate = ColumnMatch(left, right, round(match[2], 3), match.name_score, match.instance_score)
        else:  # a plain score, from a caller that replaced ``match_columns``
            candidate = ColumnMatch(left, right, round(float(match), 3))
        if candidate[2] >= SCHEMA_MATCH_THRESHOLD:
            ranked.append(candidate)
    ranked.sort(key=lambda match: -match[2])
    return ranked
