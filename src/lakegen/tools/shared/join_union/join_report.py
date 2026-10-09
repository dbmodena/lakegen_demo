"""The join half of the `check_join_union` report: the ranked keys with their measured facts."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from lakegen.domain.merge_limits import EMPTY, EXPLODES, classify_merge
from lakegen.tools.shared.join_union.join_keys import (
    NO_OVERLAP,
    KeyEvidence,
    percent,
    rows_text,
)
from lakegen.tools.shared.join_union.key_search import MAX_REPORTED_KEYS
from lakegen.tools.shared.join_union.layout import (
    ESSENTIAL,
    EXTRA,
    HEADLINE,
    JOIN_SECTION_CHARS,
    USEFUL,
    fit_lines,
)
from lakegen.tools.shared.join_union.schema_matching import SCHEMA_MATCH_THRESHOLD

# a plain merge that keeps less than this share of the normalised join rows is reported
RAW_MATCH_SHARE = 0.5
# the left join is shown when it differs from the inner join by more than this share
LEFT_JOIN_SHOWN_ABOVE = 0.05
# values shown for a key without shared values, and how long each may be
SAMPLE_VALUES = 3
SAMPLE_VALUE_CHARS = 24
# the list of the other shared columns takes at most this much of the line
ALSO_SHARED_CHARS = 600


def _raw_note(ev: KeyEvidence) -> str:
    """A warning when the columns do not merge as stored, though their values match once normalised."""
    if not ev.raw_checked or ev.size_class == EMPTY:
        return ""
    if ev.raw_inner_total is None:
        return (
            "as stored these columns cannot be merged (pandas refuses to merge text with numbers): "
            "convert one side first, e.g. pd.to_numeric(col) or col.astype(str)"
        )
    if ev.raw_inner_total == 0:
        return (
            "as stored the columns share no value, so a plain merge returns 0 rows; they match after "
            "normalising (str.strip().str.casefold(), numbers read as numbers)"
        )
    if ev.raw_inner_total > 1.5 * ev.inner_total + 1:
        stopped = classify_merge(ev.raw_inner_total, ev.left_total, ev.right_total) == EXPLODES
        return (
            f"as stored a plain merge returns {rows_text(ev, ev.raw_inner_total)} rows, not "
            f"{rows_text(ev, ev.inner_total)}{' (the sandbox stops it)' if stopped else ''}: missing keys "
            "match each other in pandas; drop them first (df.dropna(subset=key))"
        )
    if ev.raw_inner_total < RAW_MATCH_SHARE * ev.inner_total:
        return (
            f"as stored the columns match on only {rows_text(ev, ev.raw_inner_total)} of "
            f"{rows_text(ev, ev.inner_total)} rows; normalise them first (str.strip().str.casefold(), "
            "numbers read as numbers)"
        )
    return ""


def _sample_values(series: pd.Series) -> str:
    values = series.dropna().astype(str).drop_duplicates().head(SAMPLE_VALUES)
    return ", ".join(
        "'" + (v if len(v) <= SAMPLE_VALUE_CHARS else v[: SAMPLE_VALUE_CHARS - 3] + "...") + "'"
        for v in values
    )


def _key_text(ev: KeyEvidence, left: str, right: str) -> str:
    return (
        f"{' + '.join(ev.left_cols)} ({left}) = {' + '.join(ev.right_cols)} ({right})"
    )


def _facts(ev: KeyEvidence) -> str:
    # left/right, not the file names: real names are ~75 characters and the tool output is budgeted
    differs = abs(ev.left_join_total - ev.inner_total) > LEFT_JOIN_SHOWN_ABOVE * max(ev.inner_total, 1)
    left_join = (
        f" (left join {rows_text(ev, ev.left_join_total)})"
        if differs and ev.size_class != EXPLODES
        else ""
    )
    return (
        f"{ev.cardinality}, {ev.shared_distinct:,} shared key values; "
        f"{percent(ev.left_match_rate)} of the left table's rows and "
        f"{percent(ev.right_match_rate)} of the right table's rows find a partner; "
        f"inner join = {rows_text(ev, ev.inner_total)} rows{left_join}"
    )


def _short(ev: KeyEvidence) -> str:
    """One shared column pair in a line: `anno (N:N, 34)` or `zona = zona_2001 (N:N, 19)`."""
    left, right = ev.left_cols[0], ev.right_cols[0]
    names = left if left == right else f"{left} = {right}"
    return f"{names} ({ev.cardinality}, {ev.shared_distinct:,})"


def _note_priority(note: str) -> int:
    return ESSENTIAL if note.startswith(("EXPLODES", "fan-out")) else USEFUL


def _key_lines(ev: KeyEvidence, left: str, right: str, number: int | None = None) -> list[tuple[int, str]]:
    head = f"  {number}. " if number else "  "
    lines = [(ESSENTIAL, f"{head}{_key_text(ev, left, right)} -- {_facts(ev)}")]
    if ev.by_values:
        lines.append((USEFUL, "     note: named differently; Valentine matched the two columns on their values"))
    lines.extend((_note_priority(note), f"     note: {note}") for note in ev.notes)
    if raw := _raw_note(ev):
        lines.append((ESSENTIAL, f"     note: {raw}"))
    return lines


def _also_shared(keys: Sequence[KeyEvidence]) -> str:
    """The other columns both tables repeat, smallest domain first, within a line's worth of text."""
    names: list[str] = []
    size = 0
    for ev in keys:
        entry = _short(ev)
        if size + len(entry) > ALSO_SHARED_CHARS and names:
            break
        names.append(entry)
        size += len(entry) + 2
    more = f", +{len(keys) - len(names)} more" if len(keys) > len(names) else ""
    return f"  also shared (aggregate before merging on these): {', '.join(names)}{more}"


def format_join_section(
    left: str,
    right: str,
    ranked: Sequence[KeyEvidence],
    q: pd.DataFrame,
    r: pd.DataFrame,
    limit: int = JOIN_SECTION_CHARS,
) -> str:
    """The join half of the `check_join_union` report: ranked candidates with measured facts."""
    if not ranked:
        return (
            f"NO JOIN: no column pair between '{left}' and '{right}' has similar names or similar "
            f"values (score >= {SCHEMA_MATCH_THRESHOLD}), so no values were compared. Only "
            "name-similar columns are tested; a join on differently named columns is detected only "
            "when their values match clearly."
        )
    keys = [ev for ev in ranked if not ev.grain]
    grain = next((ev for ev in ranked if ev.grain), None)
    supported = [ev for ev in keys if ev.supported]
    if supported:
        shown = supported[:MAX_REPORTED_KEYS]
        if any(ev.unique_side for ev in shown):
            head = f"JOIN: '{left}' joins '{right}'; candidate keys, best first:"
        else:
            head = (
                "JOIN (many-to-many only): the tables share values on these columns, but none "
                "identifies rows on either side; aggregate one side to the key before merging:"
            )
        lines: list[tuple[int, str]] = [(HEADLINE, head)]
        for number, ev in enumerate(shown, 1):
            lines.extend(_key_lines(ev, left, right, number))
        # the grouping columns the tables share, smallest domain first (a district, a year, a category)
        also = sorted(
            (
                ev for ev in supported[MAX_REPORTED_KEYS:]
                if len(ev.left_cols) == 1 and ev.shared_distinct >= 2 and not ev.measure_like
                and min(ev.left_distinct, ev.right_distinct) > 1
            ),
            key=lambda ev: ev.shared_distinct,
        )
        if also:
            lines.append((EXTRA, _also_shared(also)))
        if grain is not None:
            grain_lines = _key_lines(grain, left, right)
            label = "  all the shared repeated columns together: "
            lines.append((USEFUL, label + grain_lines[0][1].strip()))
            lines.extend((USEFUL, text) for _, text in grain_lines[1:])
        return fit_lines(lines, limit)

    counted = (
        "an inner join on these would be EMPTY: 0 rows"
        if all(ev.exact for ev in keys[:MAX_REPORTED_KEYS])
        else "an inner join on these returns no useful rows"
    )
    lines = [(
        HEADLINE,
        f"NO JOIN: no name-similar key between '{left}' and '{right}' is backed by shared values "
        f"({counted}; an outer join or a union needs no overlap). Only name-similar columns are tested.",
    )]
    for ev in keys[:MAX_REPORTED_KEYS]:
        lines.append((ESSENTIAL, f"  - {_key_text(ev, left, right)}: {'; '.join(ev.notes) or 'no shared values'}"))
        if ev.verdict == NO_OVERLAP and ev.shared_distinct == 0:
            left_sample = _sample_values(q[ev.left_cols[0]])
            right_sample = _sample_values(r[ev.right_cols[0]])
            lines.append((USEFUL, f"    values: {left} {left_sample} | {right} {right_sample}"))
    return fit_lines(lines, limit)
