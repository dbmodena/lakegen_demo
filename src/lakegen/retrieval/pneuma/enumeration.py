"""Pneuma-Seeker's table enumeration: the siblings of a found table in a partitioned family."""

from __future__ import annotations

import re

from lakegen.retrieval.pneuma.seeker import _ALPHA_RUN, _DIGIT_RUN

# An identifier with no real word in it is an opaque id, not a series name.
_ENUMERATION_MIN_WORD = 3
# More siblings than this is a filename convention, not a table family.
_ENUMERATION_MAX_SIBLINGS = 25
# A sibling carries no evidence of its own, so it must stay below its seed.
_ENUMERATION_DECAY = 0.9


def _enumeration_pattern(stem: str) -> str | None:
    r"""Generalize the digit runs of a table identifier into a sibling pattern.

    ``water_body_testing_2020`` becomes ``water_body_testing_\d{4}``, the
    paper's own example. Everything outside a digit run is regex-escaped, so an
    identifier can never act as a pattern.

    Identifiers that carry no real word get no pattern, which is what keeps a
    lake of opaque Socrata ids out of this: ``43nn-pn8j`` would otherwise
    generalize to a shape matching unrelated datasets. The word test runs on
    the original stem, never on the digit-stripped one -- stripping joins
    ``pn`` and ``j`` into a three-letter "word" and invents a family.

    Args:
        stem: The table identifier.

    Returns:
        The sibling pattern, or ``None`` for an identifier without a digit run
        or without a word of at least ``_ENUMERATION_MIN_WORD`` letters.
    """
    if not _DIGIT_RUN.search(stem):
        return None
    if not any(len(word) >= _ENUMERATION_MIN_WORD for word in _ALPHA_RUN.findall(stem)):
        return None
    parts: list[str] = []
    index = 0
    for match in _DIGIT_RUN.finditer(stem):
        parts.append(re.escape(stem[index : match.start()]))
        parts.append(f"\\d{{{len(match.group())}}}")
        index = match.end()
    parts.append(re.escape(stem[index:]))
    return "".join(parts)
