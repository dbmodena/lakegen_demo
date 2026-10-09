"""Valentine schema matching: the name-similar column pairs that join and union evidence start from.

Schema-only COMA (`use_instances=False`) scores column-NAME similarity between two tables. It never
reads values, so a sample of the first rows gives the same ranked output as the full table --
measured on OrQa's data: identical output, 0.0 s instead of 62 s on a 19M-row pair -- and a sample
is all it is given.

A COMA score is not evidence that two tables join or union: identical names score 1.0 everywhere in
open data (`Date`, `Name`, `Parent Department`). `join_keys` measures the values of each candidate
pair and `union_mapping` measures how much of each table maps; this module only proposes candidates.
"""

from __future__ import annotations

import time

import pandas as pd
from valentine import valentine_match
from valentine.algorithms import Coma

# A column pair is a candidate for a join key / union mapping at or above this name score.
SCHEMA_MATCH_THRESHOLD = 0.5
# Rows given to COMA. Schema-only matching ignores values, so more rows only cost time.
COMA_SAMPLE_ROWS = 1000

ColumnMatch = tuple[str, str, float]


def match_columns(q: pd.DataFrame, r: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Score column correspondences between ``q`` and ``r`` with schema-only COMA."""
    raw = valentine_match(
        [q.head(COMA_SAMPLE_ROWS), r.head(COMA_SAMPLE_ROWS)],
        Coma(use_instances=False),
        instance_sample_size=COMA_SAMPLE_ROWS,
    )
    return {
        (str(pair.source_column), str(pair.target_column)): float(score)
        for pair, score in raw.items()
    }


def verify_pair_schema(q: pd.DataFrame, r: pd.DataFrame) -> dict[str, object]:
    """Return the ranked column-pair matches for a table pair, with their average and best score."""
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
        "sm_micro_avg": match_list[0][2] if match_list else 0.0,
        "sm_n_matches": len(match_list),
        "sm_time": round(elapsed, 3),
    }
