"""Term identity for the strict-AND keyword banlist."""

from __future__ import annotations

import itertools
import re
import unicodedata
from typing import Callable, Iterable, Iterator, Sequence

_POSSESSIVE = re.compile(r"['’ʼ]s$", re.IGNORECASE)


def _word_parts(word: str) -> Iterator[str]:
    """Split one whitespace-delimited word as WordDelimiterGraphFilter does.

    Boundaries fall on every non-alphanumeric character (hyphens of any kind,
    dots, slashes, apostrophes, underscores), between letters and digits, and
    from a lowercase to an uppercase letter. Accents are dropped, as Solr's
    ASCIIFoldingFilter does.
    """
    part = ""
    previous = ""
    for char in unicodedata.normalize("NFKD", _POSSESSIVE.sub("", word)):
        if unicodedata.combining(char):
            continue
        kind = "d" if char.isdigit() else "u" if char.isupper() else "l" if char.isalpha() else ""
        if not kind:
            if part:
                yield part.casefold()
            part = ""
            continue
        if part and ((kind == "d") != (previous == "d") or (kind == "u" and previous == "l")):
            yield part.casefold()
            part = ""
        part += char
        previous = kind
    if part:
        yield part.casefold()


def keyword_terms(values: Iterable[str]) -> frozenset[str]:
    """Return the terms Solr ANDs together for these keywords.

    The lexical index queries ``text`` (which every other queried field is
    copied into) through a whitespace tokenizer and a word-delimiter filter, so
    ``["catchment", "network contribution"]``, ``["catchment network",
    "contribution"]`` and ``"Network-Contribution catchment"`` are one query.
    Comparing whole keywords, or whitespace-split words, would let a regrouped
    or re-punctuated copy of a zero-result query slip past the banlist.

    Stemming and synonyms are not modelled, so a plural still looks like a new
    term: that misses a ban, it never invents one.
    """
    return frozenset(
        part
        for value in values
        for word in str(value).split()
        for part in _word_parts(word)
    )


def split_keywords(values: Iterable[str]) -> list[str]:
    """Split values into the individual words Solr ANDs together, deduplicated
    and in first-occurrence order.

    A companion to :func:`keyword_terms` for callers that must issue one word
    per AND clause -- e.g. building the actual search request -- rather than
    just comparing term sets against the banlist.
    """
    seen: set[str] = set()
    words: list[str] = []
    for value in values:
        for word in str(value).split():
            for part in _word_parts(word):
                if part not in seen:
                    seen.add(part)
                    words.append(part)
    return words


def minimal_failing_subsets(
    terms: Iterable[str],
    count: Callable[[Sequence[str]], int],
    *,
    max_probes: int = 64,
) -> tuple[list[frozenset[str]], dict[str, int]]:
    """Find the smallest word sets inside a zero-result query that match nothing.

    ``count`` says how many tables contain every word it is given. A strict AND
    can only lose matches as words are added, so a query is doomed by any subset
    that already matches nothing, and banning the whole query hides which words
    are to blame. Sets are probed smallest first, and only when every set one
    word smaller matched something, so no probe is spent on a superset of a set
    already known to fail and every set returned is minimal.

    Returns those sets, plus how many tables contain each word alone. Both can be
    partial when ``max_probes`` runs out; the sets found are still valid bans.
    A query with no failing proper subset returns ``[]``: its words each match
    tables, and only the whole combination matches none.
    """
    words = sorted(set(terms))
    failing: list[frozenset[str]] = []
    matching: set[frozenset[str]] = set()
    alone: dict[str, int] = {}
    probes = 0
    for size in range(1, len(words)):
        for combination in itertools.combinations(words, size):
            subset = frozenset(combination)
            if any(known <= subset for known in failing):
                continue
            if size > 1 and any(subset - {word} not in matching for word in subset):
                continue
            if probes >= max_probes:
                return failing, alone
            probes += 1
            matched = count(list(combination))
            if size == 1:
                alone[combination[0]] = matched
            if matched:
                matching.add(subset)
            else:
                failing.append(subset)
    return failing, alone
