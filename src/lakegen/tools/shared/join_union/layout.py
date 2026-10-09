"""The room a report section has: when it is too long, the least important lines go first.

The tool output is cut at 4,000 characters from the end, and the union verdict is the last thing in it.
Real file names run to 75 characters and some NYC columns to 150, and each appears twice per key line,
so a long join section would push the union verdict out. Each section therefore has its own budget and
drops its own lowest-priority lines (the list of other shared columns, then notes, then extra pairs)
before the cut can reach the next section.
"""

from __future__ import annotations

from collections.abc import Sequence

JOIN_SECTION_CHARS = 2200
UNION_SECTION_CHARS = 1000

# priorities: lower is kept longer
HEADLINE = 0        # never dropped
ESSENTIAL = 1       # the keys, the first mapped pairs, a size warning
USEFUL = 2          # the notes, the grain, the later pairs
EXTRA = 3           # the list of other shared columns, the warnings


def fit_lines(lines: Sequence[tuple[int, str]], limit: int) -> str:
    """The lines in order, the lowest-priority (and last) ones dropped until the text fits ``limit``.

    The headline and the first essential line (the best key, the first mapped pair) always stay.
    """
    first_essential = next((i for i, (priority, _) in enumerate(lines) if priority == ESSENTIAL), -1)
    kept = list(range(len(lines)))
    size = sum(len(text) + 1 for _, text in lines)
    for index in sorted(range(len(lines)), key=lambda i: (-lines[i][0], -i)):
        if size <= limit:
            break
        if lines[index][0] == HEADLINE or index == first_essential:
            continue
        size -= len(lines[index][1]) + 1
        kept.remove(index)
    return "\n".join(lines[index][1] for index in kept)
