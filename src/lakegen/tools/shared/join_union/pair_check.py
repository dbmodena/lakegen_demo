"""The ``check_join_union`` tool's flow for one pair of tables: read both, match their columns with Valentine, measure the join keys and the union mapping, and write the report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lakegen.integrations.table_io import TableProfile, read_profile, read_table_sample
from lakegen.tools.shared.join_union.join_keys import find_join_keys, format_join_section
from lakegen.tools.shared.join_union.schema_matching import verify_pair_schema
from lakegen.tools.shared.join_union.union_mapping import format_union_section, union_by_names

# Rows measured per table by check_join_union. Shared-value counts and column mappings only need a
# bounded sample; COMA is given a much smaller one (schema_matching.COMA_SAMPLE_ROWS). Measured on 73
# real 200k-1M-row UK pairs: 20,000 rows take about 0.3 s per pair (p90 1 s) and found 41 of 45 joins;
# 200,000 rows take about 1 s (p90 3 s) and found all 45. The joins the smaller sample misses are
# low-overlap ones (0.2%-15% of rows match).
PAIR_SAMPLE_ROWS = 20_000


def _sampling_note(name: str, frame: pd.DataFrame, total: int | None) -> str:
    if total is not None and total > len(frame):
        return f"'{name}' was sampled ({len(frame):,} of {total:,} rows)"
    if total is None and len(frame) >= PAIR_SAMPLE_ROWS:
        return f"'{name}' was cut to its first {len(frame):,} rows"
    return ""


def format_report(
    file_name_1: str,
    file_name_2: str,
    q: pd.DataFrame,
    r: pd.DataFrame,
    totals: tuple[int | None, int | None],
    matches: list[tuple[str, str, float]],
    profiles: tuple[TableProfile | None, TableProfile | None] = (None, None),
) -> str:
    """Report whether two tables join and whether they union, from measured values and coverage.

    ``profiles`` are the whole-table facts of each side when ``q`` / ``r`` are samples; without them a
    frame is taken to be the whole table.
    """
    sizes = "; ".join(
        f"'{name}' {len(frame):,} rows x {frame.shape[1]} columns"
        for name, frame in ((file_name_1, q), (file_name_2, r))
    )
    lines = [
        f"Join/union check between '{file_name_1}' and '{file_name_2}' "
        f"(Valentine column matching; values measured on {sizes})."
    ]
    sampled = [
        note
        for note in (
            _sampling_note(file_name_1, q, totals[0]),
            _sampling_note(file_name_2, r, totals[1]),
        )
        if note
    ]
    if sampled:
        lines.append(
            "Note: " + "; ".join(sampled) + " -- shared-value counts are lower bounds and "
            "uniqueness is an upper bound."
        )
    lines.append(
        format_join_section(
            file_name_1, file_name_2, find_join_keys(q, r, matches, profiles[0], profiles[1]), q, r
        )
    )
    lines.append(format_union_section(file_name_1, file_name_2, union_by_names(q, r, matches)))
    return "\n".join(lines)


def _read_side(
    table_dir: Path, file_name: str
) -> tuple[pd.DataFrame, int | None, TableProfile | None]:
    """One table of a pair as (frame, total rows, profile).

    The profile (parquet footer statistics) is read only when the frame is a sample of a larger
    table: a random sample of a row-number column (1..N) is no longer consecutive, so only the
    whole-table range tells the check it is not a join key. A whole table needs no profile.
    """
    path = Path(table_dir) / file_name.strip()
    frame, total = read_table_sample(path, PAIR_SAMPLE_ROWS)
    sampled = total is not None and total > len(frame)
    return frame, total, read_profile(path) if sampled else None


def check_join_union(table_dir: Path, file_name_1: str, file_name_2: str) -> str:
    """Measure whether two tables join (shared key values) or union (mappable columns)."""
    try:
        (q, q_total, q_profile), (r, r_total, r_profile) = (
            _read_side(table_dir, file_name) for file_name in (file_name_1, file_name_2)
        )
        matches = verify_pair_schema(q, r)["matches"]
        report = format_report(
            file_name_1, file_name_2, q, r, (q_total, r_total), matches, (q_profile, r_profile)
        )
    except Exception as e:
        return f"Error checking join/union between '{file_name_1}' and '{file_name_2}': {e}. Try different tables."

    return report
