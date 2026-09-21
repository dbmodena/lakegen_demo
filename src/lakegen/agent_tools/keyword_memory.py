"""Deterministic, zero-LLM-cost keyword-search bookkeeping for the Phase 1&2
discovery agent's `search_tables` tool.

Ported from the scratchpad research that validated this design (see
`keyword_memory_v6_test.py`'s "distinctive_memory_freq" arm): on 48 UK OrQa
questions, this loop alone raised gold reaching the candidate pool at all
from 66% to 85-89%, while using *fewer* average search rounds, not more.

Nothing here decides anything for the agent. Every function either tracks
state (`add_no_result_ban`/`check_ban`) or builds a report string
(`local_coverage_report`) that gets appended to `search_tables`'s normal
response -- the agent still chooses what to keep, drop, or add.

Only meaningful for KEYWORD and HYBRID retrieval modes: SEMANTIC mode's
retriever takes no `keywords` argument at all (pure question-embedding KNN,
no AND-query, no true zero-hit case), so callers gate use of this module on
`retrieval_config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID)`.
"""
from __future__ import annotations

import re


def toks(keywords: list) -> frozenset:
    """Flatten a keyword/phrase list into a single casefolded set of word
    tokens. This is what makes ban-checking invariant to phrase grouping,
    list order, and case: ["Combined Authority", "BUC"] and ["buc",
    "combined", "authority"] produce the identical frozenset, and Solr's own
    AND query is equally order/grouping-invariant, so this is the correct
    granularity for detecting when one attempt is a subset/superset of
    another."""
    return frozenset(t for k in keywords for t in re.findall(r"\w+", str(k).casefold()))


def add_no_result_ban(
    banned_superset_of: list[tuple[frozenset, list]],
    new_tokens: frozenset,
    new_keywords: list,
) -> None:
    """Record a proven zero-hit combination, maintaining a minimal antichain:
    if a later, smaller zero-hit combination is a subset of an existing
    banned entry, the bigger entry is now redundant (any superset of the new,
    smaller set is *also* a superset of it) and is dropped, keeping only the
    smaller, more general one.

    Example: [credit, bank, times, square] is banned, then [credit, bank] is
    also found to return zero hits -- the first entry becomes redundant and
    is pruned; only [credit, bank] (and anything that is a superset of it)
    stays banned."""
    if any(existing <= new_tokens for existing, _ in banned_superset_of):
        return
    banned_superset_of[:] = [
        (zset, kw) for zset, kw in banned_superset_of if not (zset > new_tokens)
    ]
    banned_superset_of.append((new_tokens, new_keywords))


def check_ban(
    proposed_tokens: frozenset,
    banned_superset_of: list[tuple[frozenset, list]],
) -> str | None:
    """None if the proposed keyword set is not banned. Otherwise a short
    reason naming the banned subset it contains -- a set is a superset of
    itself, so this also covers an exact repeat of a banned combination."""
    for zset, zkw in banned_superset_of:
        if proposed_tokens >= zset:
            return f"identical or superset to a banned zero-result combination: {zkw}"
    return None


_QCW_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "is", "are", "was",
    "were", "how", "many", "what", "which", "total", "all", "and", "or",
    "by", "at", "according", "recorded", "during", "that", "this", "with",
    "from", "have", "been", "each", "into", "over", "does", "did", "there",
}


def question_content_words(question: str) -> list[str]:
    """The question's own distinctive words, in first-appearance order,
    stopwords and short tokens dropped, case-preserved for display but
    deduplicated case-insensitively."""
    words = re.findall(r"[A-Za-z0-9]+", question)
    seen: list[str] = []
    for w in words:
        if (
            len(w) > 2
            and w.casefold() not in _QCW_STOP
            and w.casefold() not in {s.casefold() for s in seen}
        ):
            seen.append(w)
    return seen


def _candidate_text_blob(meta: dict) -> str:
    cols = meta.get("columns.name") or []
    parts = [
        str(meta.get("title", "")),
        str(meta.get("description", "")),
        " ".join(str(c) for c in cols),
    ]
    return " ".join(parts).casefold()


def local_term_coverage(candidate_metas: list[dict], word: str) -> int:
    """How many of the ALREADY-RETRIEVED candidates (title/description/
    column names -- no extra Solr call, just inspecting what's already in
    hand) contain this word. Used for the too-broad/truncated case: tells
    the agent which word from the question would actually narrow THIS pool,
    rather than guessing from a global rarity number that may not reflect
    the local neighbourhood at all (e.g. a word can be globally common but
    still be exactly the right one for this specific question)."""
    wl = word.casefold()
    return sum(1 for meta in candidate_metas if wl in _candidate_text_blob(meta))


def local_coverage_report(
    candidate_metas: list[dict], question: str, current_keywords: list
) -> str:
    """Every question word's hit count against the currently visible
    candidates, sorted most-discriminating first. Appended to a truncated/
    at-cap search response; the agent decides what (if anything) to do with
    it."""
    n = len(candidate_metas)
    current_toks = toks(current_keywords)
    rows = []
    for w in question_content_words(question):
        cov = local_term_coverage(candidate_metas, w)
        marker = " <- already in your query" if w.casefold() in current_toks else ""
        rows.append((cov, w, marker))
    rows.sort(key=lambda r: r[0])
    lines = [
        f"  {w!r}: matches {cov}/{n} of your current candidates{marker}"
        for cov, w, marker in rows
    ]
    header = (
        f"Local word-coverage check across your current {n} candidates (every "
        "question word checked against the tables you actually retrieved, not "
        f"the whole catalog; lower = more discriminating -- a word matching all "
        f"{n} isn't narrowing anything and may be worth dropping; a word "
        "matching a small subset is a strong candidate to add):"
    )
    return header + "\n" + "\n".join(lines)


def solr_term_doc_frequency(solr_client, term: str, cache: dict[str, int]) -> int:
    """Corpus-wide hit count for ONE token, OR-matched, through the
    retriever's existing public `select` interface -- no retriever code is
    modified. Used ONLY for the zero-hit case, where there is no local pool
    to inspect: a word matching almost nothing anywhere is likely a row
    value, not catalog metadata. `cache` is caller-owned (e.g. a dict on
    P12State) so repeated words across rounds don't re-issue the same
    query."""
    if term in cache:
        return cache[term]
    try:
        response = solr_client.select(tokens=[term], q_op="OR", rows=0)
        n = int(response.get("response", {}).get("numFound", 0))
    except Exception:  # noqa: BLE001
        n = -1
    cache[term] = n
    return n


def global_rarity_report(
    solr_client, keywords: list, cache: dict[str, int], catalog_size_hint: int | None = None
) -> str:
    """Per-word corpus-wide rarity for a just-failed (zero-hit) keyword set.
    Appended to a zero-hit search response alongside the forced-
    reformulation instruction."""
    words = sorted(
        {t for k in keywords for t in re.findall(r"\w+", str(k))}, key=str.casefold
    )
    size_note = f" of ~{catalog_size_hint:,} tables in the catalog" if catalog_size_hint else ""
    lines = [
        f"  {w!r}: appears in {solr_term_doc_frequency(solr_client, w, cache)}{size_note}"
        for w in words
    ]
    return (
        "This combination returned 0 hits -- no local pool to check, so here is "
        "each word's rarity across the whole catalog (higher = less likely to be "
        "the problem):\n" + "\n".join(lines)
    )
