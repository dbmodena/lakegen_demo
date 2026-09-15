"""Valentine schema matching with the table-pair criteria used by OrQa.

OrQa scores a pair of tables with schema-only COMA and reads everything off the
column-pair scores:

- the pair gate passes when the average score over every returned column pair,
  or the single best score, reaches 0.5;
- a union is supported when the average score reaches 0.5, aligned on the
  column pairs that score at least 0.5;
- a join is supported when the best score reaches 0.5, and that pair is the key.

Unlike OrQa, which samples the first rows and columns, every row and column of
both tables is matched.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

SCHEMA_MATCH_THRESHOLD = 0.5
SM_MACRO_AVG_THRESHOLD = 0.5
SM_MICRO_AVG_THRESHOLD = 0.5

ColumnMatch = tuple[str, str, float]


def match_columns(q: pd.DataFrame, r: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Score column correspondences between ``q`` and ``r`` with schema-only COMA."""
    # Valentine samples 1000 rows by default; None makes it use every row.
    raw = valentine_match([q, r], Coma(use_instances=False), instance_sample_size=None)
    return {
        (str(pair.source_column), str(pair.target_column)): float(score)
        for pair, score in raw.items()
    }


def verify_pair_schema(q: pd.DataFrame, r: pd.DataFrame) -> dict[str, object]:
    """Return OrQa's pair evidence: ranked matches, average and best score."""
    start = time.time()
    matches = match_columns(q, r)
    elapsed = time.time() - start

    match_list: list[ColumnMatch] = sorted(
        ((q_col, r_col, round(score, 3)) for (q_col, r_col), score in matches.items()),
        key=lambda match: -match[2],
    )
    average = sum(matches.values()) / len(matches) if matches else 0.0
    return {
        "matches": match_list,
        "sm_macro_avg": round(average, 3),
        # The best single column match, so one excellent join key keeps a
        # pair alive even when the schemas differ overall.
        "sm_micro_avg": match_list[0][2] if match_list else 0.0,
        "sm_n_matches": len(match_list),
        "sm_time": round(elapsed, 3),
    }


def passes_schema_gate(
    evidence: dict[str, object],
    macro_threshold: float = SM_MACRO_AVG_THRESHOLD,
    micro_threshold: float = SM_MICRO_AVG_THRESHOLD,
) -> bool:
    """A pair proceeds when the schemas match globally or one column pair matches strongly."""
    return (
        evidence["sm_macro_avg"] >= macro_threshold
        or evidence["sm_micro_avg"] >= micro_threshold
    )


def union_evidence(evidence: dict[str, object], q_columns: Sequence[str]) -> dict[str, object]:
    """Aligned column pairs for a union, and whether the average score supports one."""
    confident = [
        (q_col, r_col, score)
        for q_col, r_col, score in evidence["matches"]
        if score >= SCHEMA_MATCH_THRESHOLD
    ]
    matched_q_columns = {q_col for q_col, _, _ in confident}
    return {
        "q_columns": [q_col for q_col, _, _ in confident],
        "r_columns": [r_col for _, r_col, _ in confident],
        "column_scores": [score for _, _, score in confident],
        "union_column_ratio": (
            round(len(matched_q_columns) / len(q_columns), 4) if q_columns else 0.0
        ),
        "supported": evidence["sm_macro_avg"] >= SM_MACRO_AVG_THRESHOLD,
    }


def join_evidence(evidence: dict[str, object]) -> dict[str, object]:
    """The best column pair as join key, and whether its score supports a join."""
    matches = evidence["matches"]
    if not matches:
        return {"key": None, "score": 0.0, "supported": False, "alternatives": []}
    q_key, r_key, score = matches[0]
    return {
        "key": (q_key, r_key),
        "score": score,
        "supported": evidence["sm_micro_avg"] >= SM_MICRO_AVG_THRESHOLD,
        "alternatives": [
            match for match in matches[1:] if match[2] >= SCHEMA_MATCH_THRESHOLD
        ],
    }
