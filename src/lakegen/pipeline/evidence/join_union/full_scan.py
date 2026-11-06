"""Count the best candidate keys over every row of their columns, not over a sample.

The pair check reads at most ``PAIR_SAMPLE_ROWS`` rows of each table to match the columns and to rank the
candidate keys. That is enough to find the keys, but not to size the join: the rows of a join grow with the
product of the repeats on both sides, so a sample of 20,000 of 1,000,000 rows counts about 1/2,500 of the
merge. Measured on the reference merges of the OrQa lakes, the sampled count of a many-to-many key was 4x to
100x too small (808 million rows, counted as 197 million), and a rare shared value could be missed entirely.

This module re-measures the top keys on the key columns alone, which a parquet file reads without the other
columns. A table beyond ``FULL_SCAN_MAX_ROWS`` is read as a large evenly spaced sample and the counts are
scaled up (an unbiased estimate of the join size), and the evidence says so.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lakegen.pipeline.evidence.join_union.join_keys import (
    NO_OVERLAP,
    KeyEvidence,
    add_raw_merge_facts,
    measure_key,
)
from lakegen.pipeline.evidence.join_union.key_search import rank_keys
from lakegen.utils.table_io import TableProfile, read_table_sample

logger = logging.getLogger(__name__)

# Key columns are read in full up to this many cells per table (a 6-million-row key takes about 3 s and
# a few hundred MB; a composite key of four columns is read to 4,000,000 rows).
FULL_SCAN_MAX_CELLS = 16_000_000
# Beyond it, a sample of this many cells is read and the sizes are scaled up.
SCALED_SAMPLE_CELLS = 3_000_000
# Candidate keys re-measured, best first, within this time (the rest keep their sampled, scaled sizes).
FULL_SCAN_KEYS = 6
FULL_SCAN_SECONDS = 12.0
# What reading and counting one cell of a key column costs (6 million int64 rows: 3 s; 12 million cells: 4.2 s).
SECONDS_PER_CELL = 4e-7
# Keys with no shared value in the sample but a strong name match: a rare shared value may have been missed.
RESCAN_NO_OVERLAP_KEYS = 3
RESCAN_NAME_SCORE = 0.8


@dataclass(frozen=True)
class TableSide:
    """One table of a pair: where it is, the sample read from it, and what the file says of the whole.

    Attributes:
        name: The file name, as the agent gave it.
        path: The file.
        sample: The rows read (all of them for a small table).
        total_rows: Rows of the whole table; None when the file does not say (a CSV cut at the sample size).
        profile: The footer facts of the table, read only when ``sample`` is a sample.
    """

    name: str
    path: Path
    sample: pd.DataFrame
    total_rows: int | None
    profile: TableProfile | None = None

    @property
    def sampled(self) -> bool:
        """Whether ``sample`` holds fewer rows than the table."""
        return self.total_rows is not None and self.total_rows > len(self.sample)


def _targets(ranked: Sequence[KeyEvidence]) -> list[KeyEvidence]:
    chosen = list(ranked[:FULL_SCAN_KEYS])
    chosen += [ev for ev in ranked[FULL_SCAN_KEYS:] if ev.grain or ev.by_cue]
    rare = [
        ev
        for ev in ranked[FULL_SCAN_KEYS:]
        if ev.verdict == NO_OVERLAP
        and ev.name_score >= RESCAN_NAME_SCORE
        and not any("nested" in note for note in ev.notes)
    ]
    return chosen + rare[:RESCAN_NO_OVERLAP_KEYS]


def _rows_to_read(side: TableSide, columns: int) -> int:
    """Every row of the table when its key columns are few enough cells, else a large sample."""
    cells = side.total_rows * columns
    return side.total_rows if cells <= FULL_SCAN_MAX_CELLS else SCALED_SAMPLE_CELLS // columns


def _seconds_to_measure(left: TableSide, right: TableSide, columns: int) -> float:
    return (
        SECONDS_PER_CELL * columns * (_rows_to_read(left, columns) + _rows_to_read(right, columns))
    )


def _read_key_columns(side: TableSide, columns: Sequence[str]) -> pd.DataFrame:
    wanted = list(dict.fromkeys(columns))
    frame, _ = read_table_sample(side.path, _rows_to_read(side, len(wanted)), columns=wanted)
    return frame


def refine_on_whole_tables(
    ranked: list[KeyEvidence],
    left: TableSide,
    right: TableSide,
    *,
    seconds: float = FULL_SCAN_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> list[KeyEvidence]:
    """The ranked keys, the best of them re-measured on the key columns of the whole tables.

    Returns the list unchanged when neither table was sampled (the evidence already covers every row)
    or when a table's size is unknown (a CSV cut at the sample size). A key that cannot be re-read keeps
    its sampled evidence.

    Args:
        ranked: The candidate keys measured on the samples, best first.
        left: The left table.
        right: The right table.
        seconds: The time the re-measuring may take; keys that would not finish in it are skipped.
        clock: The clock the time is measured on.

    Returns:
        The keys, the re-measured ones replaced, ranked again.
    """
    if not (left.sampled or right.sampled) or left.total_rows is None or right.total_rows is None:
        return ranked
    deadline = clock() + seconds
    refined = list(ranked)
    position = {id(ev): index for index, ev in enumerate(ranked)}
    for ev in _targets(ranked):
        # a key that would not finish within the time left is skipped, not started
        if clock() + _seconds_to_measure(left, right, len(ev.left_cols)) > deadline:
            continue
        try:
            q = _read_key_columns(left, ev.left_cols)
            r = _read_key_columns(right, ev.right_cols)
            fresh = measure_key(
                q,
                r,
                ev.left_cols,
                ev.right_cols,
                ev.name_score,
                left.profile,
                right.profile,
                left.total_rows,
                right.total_rows,
            )
        except Exception:  # noqa: BLE001 - an unreadable column leaves the sampled evidence in place
            logger.debug(
                "Could not re-measure %s = %s on the whole tables; keeping the sampled evidence",
                ev.left_cols,
                ev.right_cols,
                exc_info=True,
            )
            continue
        fresh.by_values, fresh.grain, fresh.by_cue = ev.by_values, ev.grain, ev.by_cue
        if (
            fresh.supported
        ):  # what a plain merge does on the rows just read, before they are released
            add_raw_merge_facts(fresh, q, r)
        refined[position[id(ev)]] = fresh
    return rank_keys(refined)
