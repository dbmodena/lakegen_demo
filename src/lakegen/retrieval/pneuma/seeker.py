"""Pneuma-Seeker's content search: the scan of the tables' text columns for the question's entities."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from lakegen.retrieval.pneuma import cell_scan

# Text columns are scanned in isolated worker processes, one file and one
# bounded chunk at a time. This keeps Pneuma-Seeker's content scan independent
# from any optional retrieval implementation.
ScanResult = tuple[dict[str, dict[str, int]], tuple[str, ...], int]


def _patterns(terms: Sequence[str], pattern: Callable[[str], str]) -> list[tuple[str, str]]:
    return [(term, pattern(term)) for term in terms]


def scan_many(
    paths: Sequence[Path],
    terms: Sequence[str],
    *,
    max_columns: int | None,
    max_rows: int | None,
    workers: int,
    pattern: Callable[[str], str],
) -> list[ScanResult | None]:
    """Scan the text columns of each file for the terms, in worker processes.

    Args:
        paths: The Parquet files.
        terms: The entities to scan for.
        max_columns: How many text columns to scan per file, or None for all.
        max_rows: How many rows to read per file, or None for all.
        workers: How many worker processes to run at most.
        pattern: Turns a term into its regex.

    Returns:
        Per file, ``(term -> column -> occurrences, scanned columns, columns
        that are not text)``, or None when the file crashed its worker.
    """
    patterns = _patterns(terms, pattern)
    return cell_scan.run(
        "scan_text",
        [(str(path), patterns, max_columns, max_rows) for path in paths],
        workers=workers,
    )


def probe_many(
    paths: Sequence[Path],
    terms: Sequence[str],
    *,
    max_columns: int | None,
    max_rows: int,
    workers: int,
    pattern: Callable[[str], str],
) -> list[int]:
    """Count the rows of each file's prefix that match a term, in worker processes.

    Args:
        paths: The Parquet files.
        terms: The entities to look for.
        max_columns: How many castable columns to test per file, or None for all.
        max_rows: The length of each file's prefix.
        workers: How many worker processes to run at most.
        pattern: Turns a term into its regex.

    Returns:
        The matching rows per file; 0 for a file that crashed its worker.
    """
    patterns = _patterns(terms, pattern)
    return [
        count or 0
        for count in cell_scan.run(
            "probe",
            [(str(path), patterns, max_columns, max_rows) for path in paths],
            workers=workers,
        )
    ]


# More entities than a question plausibly names; each one is another regex per
# scanned column.
_MAX_SCAN_TERMS = 12

_NON_ALNUM_RUN = re.compile(r"[^A-Za-z0-9]+")
_DIGIT_RUN = re.compile(r"\d+")
_ALPHA_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _entity_pattern(entity: str) -> str:
    """The reference implementation's pattern for one entity, or ``""`` for none.

    The entity is split on every run of characters other than ASCII letters and
    digits, and the parts are joined by exactly one arbitrary character, so
    ``New York`` also matches ``NEW-YORK`` and ``new_york``. A match must start
    and end at a boundary -- the edge of the text, or a character that is not a
    letter, digit, underscore, or hyphen -- so ``art`` does not match
    ``Department`` and ``2020`` does not match ``120201``. One consequence the
    reference code shares: ``borough`` does not match a column named
    ``borough_name``, because an underscore is not a boundary. Case is ignored.

    Only letters and digits reach the pattern, so nothing an agent supplies can
    act as regex syntax, and an entity with neither (``.*``) has no pattern.

    Args:
        entity: The entity, as written.

    Returns:
        The case-insensitive, boundary-anchored regex, or ``""``.
    """
    parts = [re.escape(part) for part in _NON_ALNUM_RUN.split(entity) if part]
    if not parts:
        return ""
    return "(?i)(^|[^A-Za-z0-9_-])" + ".".join(parts) + "($|[^A-Za-z0-9_-])"


def _scan_terms(entities: Sequence[str] | None) -> list[str]:
    """The entities to scan for, as listed, each once.

    Only whitespace is normalized: the pattern ignores case anyway, and the
    entity is shown back to the agent as written. Entities sharing a pattern
    (``New York``, ``new-york``) are one entity, or it would count twice in the
    mean and in the coverage.

    Args:
        entities: The entities, as given; None for none.

    Returns:
        At most ``_MAX_SCAN_TERMS`` terms, in their first order.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for value in entities or ():
        term = " ".join(str(value).split())
        pattern = _entity_pattern(term).casefold()
        if pattern and pattern not in seen:
            seen.add(pattern)
            terms.append(term)
    return terms[:_MAX_SCAN_TERMS]
