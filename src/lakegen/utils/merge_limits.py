"""How large a merge result may be: one rule for the check that predicts it and the guard that enforces it.

The sandbox wraps ``pandas.merge`` and stops a merge whose result is both larger than
``MERGE_BLOWUP_MIN_ROWS`` and more than ``MERGE_BLOWUP_FACTOR`` times its larger input
(``CARTESIAN_PRODUCT_DETECTED``): the join key is not unique on either side and every row met every
row. The join/union check applies the same rule to the key columns of two tables before any code
is written, so the agent hears about the blow-up from the tool and not from a failed run.
"""

from __future__ import annotations

MERGE_BLOWUP_MIN_ROWS = 50_000
MERGE_BLOWUP_FACTOR = 25


def merge_blows_up(result_rows: int, left_rows: int, right_rows: int) -> bool:
    """Whether a merge of two frames of ``left_rows`` and ``right_rows`` into ``result_rows`` is stopped."""
    baseline = max(left_rows, right_rows, 1)
    return result_rows > MERGE_BLOWUP_MIN_ROWS and result_rows > baseline * MERGE_BLOWUP_FACTOR


# What a merge of two tables on a key yields, by size.
EMPTY = "EMPTY"  # no rows
SAFE = "SAFE"  # no more rows than the larger input
FANOUT = "FANOUT"  # more rows than the larger input: rows repeat
EXPLODES = "EXPLODES"  # the sandbox stops it


def classify_merge(result_rows: int, left_rows: int, right_rows: int) -> str:
    """The size class of a merge: EMPTY, SAFE, FANOUT or EXPLODES."""
    if result_rows <= 0:
        return EMPTY
    if merge_blows_up(result_rows, left_rows, right_rows):
        return EXPLODES
    if result_rows > max(left_rows, right_rows):
        return FANOUT
    return SAFE
