"""Term identity for the strict-AND keyword banlist."""

from __future__ import annotations

import itertools
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

_POSSESSIVE = re.compile(r"['’ʼ]s$", re.IGNORECASE)


def _word_parts(word: str) -> Iterator[str]:
    """Split one whitespace-delimited word as WordDelimiterGraphFilter does.

    Boundaries fall on every non-alphanumeric character (hyphens of any kind,
    dots, slashes, apostrophes, underscores), between letters and digits, and
    from a lowercase to an uppercase letter. Accents are dropped, as Solr's
    ASCIIFoldingFilter does.

    Args:
        word: One whitespace-delimited word, possessive "'s" included.

    Yields:
        The casefolded parts, in order.
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

    Args:
        values: The keywords of one query.

    Returns:
        The casefolded terms.
    """
    return frozenset(
        part for value in values for word in str(value).split() for part in _word_parts(word)
    )


def split_keywords(values: Iterable[str]) -> list[str]:
    """Split values into the individual words Solr ANDs together.

    Deduplicated and in first-occurrence order. A companion to
    :func:`keyword_terms` for callers that must issue one word per AND clause
    -- e.g. building the actual search request -- rather than just comparing
    term sets against the banlist.

    Args:
        values: The keywords of one query.

    Returns:
        The casefolded words.
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

    Args:
        terms: The words of a query that matched nothing.
        count: How many tables contain every word it is given.
        max_probes: The most ``count`` calls to make.

    Returns:
        The minimal failing word sets, and the table count of each word alone.
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


def _ban_order(terms: frozenset[str]) -> tuple[int, list[str]]:
    return len(terms), sorted(terms)


def banned_subset(bans: Iterable[frozenset[str]], keywords: Iterable[str]) -> frozenset[str] | None:
    """The smallest banned word set inside ``keywords``, or None.

    A strict AND query only loses matches as words are added, so a query that
    contains a set known to match nothing matches nothing too.

    Args:
        bans: The word sets known to match nothing.
        keywords: The keywords of the proposed query.

    Returns:
        The smallest ban the query contains (fewest words, then alphabetical), or None.
    """
    proposed = keyword_terms(keywords)
    return min((known for known in bans if known <= proposed), key=_ban_order, default=None)


def add_ban(bans: list[frozenset[str]], terms: frozenset[str]) -> None:
    """Add one word set that matches nothing to ``bans``, in place, dropping the supersets it makes redundant."""
    if not terms or any(known <= terms for known in bans):
        return
    bans[:] = sorted([known for known in bans if not terms < known] + [terms], key=_ban_order)


def shrink_and_ban(
    bans: list[frozenset[str]],
    keywords: Iterable[str],
    count: Callable[[Sequence[str]], int] | None,
) -> str | None:
    """Ban a zero-result AND query as its smallest failing word sets, and say what was learned.

    ``count`` tells the sets apart; without it, or for a single word, or when
    a probe fails, the whole query is banned.

    Args:
        bans: The word sets known to match nothing, changed in place.
        keywords: The keywords of the query that matched nothing.
        count: How many tables contain every word it is given, or None.

    Returns:
        What the probes learned, for the agent; None when nothing beyond the
        empty result itself was learned.
    """
    terms = keyword_terms(keywords)
    if count is None or len(terms) < 2:
        add_ban(bans, terms)
        return None
    try:
        failing, alone = minimal_failing_subsets(terms, count)
    except Exception:
        logger.debug(
            "Word-set probe failed; banning the whole query %s", sorted(terms), exc_info=True
        )
        add_ban(bans, terms)
        return None
    alone_text = ", ".join(f"{word} {matched}" for word, matched in sorted(alone.items()))
    if failing:
        for subset in failing:
            add_ban(bans, subset)
        sets_text = ", ".join("{" + ", ".join(sorted(subset)) + "}" for subset in failing)
        text = f"Smallest word sets that already match nothing: {sets_text}."
        if alone_text:
            text += f" Tables containing each word alone: {alone_text}."
        return text
    add_ban(bans, terms)
    return f"Each word matches tables alone ({alone_text})." if alone_text else None
