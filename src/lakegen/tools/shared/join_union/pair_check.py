"""The ``check_join_union`` tool's flow for one pair of tables.

Read a bounded sample of both tables, match their columns with Valentine, rank the candidate join keys on
the sample, count the best keys over the key columns of the whole tables (so the size of the join is the
real one), map the columns for a union, and write the report the agents read.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from lakegen.integrations.table_io import read_profile, read_table_sample
from lakegen.tools.shared.join_union.full_scan import (
    FULL_SCAN_SECONDS,
    TableSide,
    refine_on_whole_tables,
)
from lakegen.tools.shared.join_union.join_keys import KeyEvidence
from lakegen.tools.shared.join_union.join_report import format_join_section
from lakegen.tools.shared.join_union.key_search import MAX_REPORTED_KEYS, find_join_keys
from lakegen.tools.shared.join_union.schema_matching import ColumnMatch, match_pair
from lakegen.tools.shared.join_union.union_mapping import (
    UnionMapping,
    format_union_section,
    union_by_names,
)

# Rows read per table to match the columns and rank the candidate keys. Shared-value counts and column
# mappings only need a bounded sample; the sizes of the best keys are counted over all rows afterwards
# (full_scan). Measured on 73 real 200k-1M-row UK pairs: 20,000 rows take about 0.3 s per pair (p90 1 s)
# and found 41 of 45 joins; 200,000 rows take about 1 s (p90 3 s) and found all 45. The joins the smaller
# sample misses are low-overlap ones (0.2%-15% of rows match), which the whole-table count recovers for
# the keys that reach it.
PAIR_SAMPLE_ROWS = 20_000


@dataclass
class PairAnalysis:
    """Everything the check found out about a pair of tables."""

    left: TableSide
    right: TableSide
    matches: list[ColumnMatch]
    keys: list[KeyEvidence]
    union: UnionMapping


def read_side(table_dir: Path, file_name: str) -> TableSide:
    """One table of a pair: a bounded sample, its row count, and (for a sample) its footer profile.

    The profile (parquet footer statistics) is read only when the frame is a sample of a larger
    table: a random sample of a row-number column (1..N) is no longer consecutive, so only the
    whole-table range tells the check it is not a join key. A whole table needs no profile.
    """
    path = Path(table_dir) / file_name.strip()
    frame, total = read_table_sample(path, PAIR_SAMPLE_ROWS)
    sampled = total is not None and total > len(frame)
    return TableSide(file_name, path, frame, total, read_profile(path) if sampled else None)


def analyse_pair(left: TableSide, right: TableSide, *, scan_seconds: float = FULL_SCAN_SECONDS) -> PairAnalysis:
    """Match, measure and size the join and union of two tables."""
    matches = match_pair(left.sample, right.sample)
    keys = find_join_keys(
        left.sample, right.sample, matches, left.profile, right.profile,
        left.total_rows, right.total_rows,
    )
    keys = refine_on_whole_tables(keys, left, right, seconds=scan_seconds)
    return PairAnalysis(left, right, matches, keys, union_by_names(left.sample, right.sample, matches))


def _sampling_note(side: TableSide) -> str:
    if side.sampled:
        return f"'{side.name}' was sampled ({len(side.sample):,} of {side.total_rows:,} rows)"
    if side.total_rows is None and len(side.sample) >= PAIR_SAMPLE_ROWS:
        return f"'{side.name}' was cut to its first {len(side.sample):,} rows"
    return ""


def _note(analysis: PairAnalysis) -> str:
    """Why the numbers below are or are not over every row."""
    left, right = analysis.left, analysis.right
    sampled = [s for s in (_sampling_note(left), _sampling_note(right)) if "sampled" in s]
    cut = [s for s in (_sampling_note(left), _sampling_note(right)) if "cut" in s]
    parts = []
    if sampled:
        shown = [ev for ev in analysis.keys if not ev.grain][:MAX_REPORTED_KEYS]
        scaled = any(not ev.exact for ev in shown)
        parts.append(
            "; ".join(sampled) + " to match the columns; the join sizes are counted over every row of "
            "the key columns" + (", except those marked ~, which are scaled up from a sample" if scaled else "")
        )
    if cut:
        parts.append(
            "; ".join(cut) + " -- shared-value counts are lower bounds and uniqueness is an upper bound"
        )
    return "Note: " + ". ".join(parts) + "." if parts else ""


def format_report(analysis: PairAnalysis) -> str:
    """Report whether two tables join and whether they union, from measured values and coverage."""
    left, right = analysis.left, analysis.right
    sizes = "; ".join(
        f"'{side.name}' {(side.total_rows or len(side.sample)):,} rows x {side.sample.shape[1]} columns"
        for side in (left, right)
    )
    lines = [
        f"Join/union check between '{left.name}' and '{right.name}' (Valentine column matching; {sizes})."
    ]
    if note := _note(analysis):
        lines.append(note)
    lines.append(format_join_section(left.name, right.name, analysis.keys, left.sample, right.sample))
    lines.append(format_union_section(left.name, right.name, analysis.union))
    return "\n".join(lines)


_REPORTS: OrderedDict[tuple, str] = OrderedDict()
_REPORT_CACHE_SIZE = 64
_REPORTS_LOCK = threading.Lock()  # the tools run on thread pools


def _stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def check_join_union(table_dir: Path, file_name_1: str, file_name_2: str) -> str:
    """Measure whether two tables join (shared key values) or union (mappable columns).

    The report of a pair is kept while the two files are unchanged: counting the key columns of large
    tables takes seconds, and an agent asks about the same pair again when it re-plans.
    """
    try:
        paths = [Path(table_dir) / name.strip() for name in (file_name_1, file_name_2)]
        key = (*(str(path) for path in paths), *(_stamp(path) for path in paths))
        with _REPORTS_LOCK:
            if key in _REPORTS:
                _REPORTS.move_to_end(key)
                return _REPORTS[key]
        analysis = analyse_pair(read_side(table_dir, file_name_1), read_side(table_dir, file_name_2))
        report = format_report(analysis)
    except Exception as e:
        return f"Error checking join/union between '{file_name_1}' and '{file_name_2}': {e}. Try different tables."
    with _REPORTS_LOCK:
        _REPORTS[key] = report
        while len(_REPORTS) > _REPORT_CACHE_SIZE:
            _REPORTS.popitem(last=False)
    return report
