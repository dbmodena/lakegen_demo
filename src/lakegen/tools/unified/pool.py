"""The candidate pool of a unified discovery round: what retrieval found, fused across searches.

The pool lives on the round's ``P12State``: every candidate the searches
surfaced (``all_candidates``), its best rank and its reciprocal-rank score
across searches, and the catalogue entry recorded at its best rank. A
search folds its hits in (``fold_hits``), the pool is re-ranked and cut to
``top_k`` with the carried tables of earlier rounds put first (``rank``),
and the agent sees its first ``initial_candidates`` (``show``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lakegen.integrations.solr.catalogue import match_local_csv, solr_metadata_from_doc
from lakegen.tools.unified.state import P12State

RRF_K = 60.0
"""The reciprocal-rank constant: a hit at rank r contributes 1 / (60 + r)."""


def fold_hits(
    state: P12State, hits: Sequence[Any], *, keywords: list[str], attempt: int, all_files: list[str], top_k: int,
) -> tuple[list[str], list[str]]:
    """Fold a search's hits into the pool: the local files it found, and the banned ones it skipped.

    A hit maps to a local file, or is dropped; a banned file is skipped
    (and reported, since a search that found only banned tables counts as
    empty); the first ``top_k`` new files are this search's candidates. A
    candidate's catalogue entry is the one at its best rank so far.
    """
    current: list[str] = []
    banned: list[str] = []
    for hit in hits:
        matched = match_local_csv(hit.document, all_files)
        if matched is None or matched in current:
            continue
        if matched.casefold() in state.excluded_tables:
            if matched not in banned:
                banned.append(matched)
            continue
        current.append(matched)
        state.candidate_scores[matched] = state.candidate_scores.get(matched, 0.0) + 1.0 / (RRF_K + hit.rank)
        previous_rank = state.best_ranks.get(matched)
        if previous_rank is None or hit.rank < previous_rank:
            state.best_ranks[matched] = hit.rank
            state.solr_meta[matched] = solr_metadata_from_doc(hit.document)
            state.solr_meta[matched]["retrieval"] = hit.to_log_dict()
            state.solr_meta[matched]["best_attempt"] = attempt
            state.solr_meta[matched]["best_keywords"] = list(keywords)
        if matched not in state.all_candidates:
            state.all_candidates.append(matched)
        if len(current) >= top_k:
            break
    return current, banned


def rank(state: P12State, top_k: int) -> list[str]:
    """The pool by fused score, cut to ``top_k``, the carried tables first.

    Fusing at most two distinct searches with reciprocal-rank contributions
    keeps the first search's own order. A table carried from an earlier
    round already proved it serves part of the question: it is re-surfaced
    whether or not this round found it again, even at the cost of the
    weakest fresh candidate.
    """
    state.all_candidates.sort(key=lambda candidate: (
        -state.candidate_scores.get(candidate, 0.0),
        state.best_ranks.get(candidate, 10**9),
        candidate,
    ))
    state.all_candidates = state.all_candidates[:top_k]
    for table in state.carried_tables:
        if table.casefold() in state.excluded_tables:
            continue
        if table in state.all_candidates:
            state.all_candidates.remove(table)
        state.all_candidates.insert(0, table)
        if table not in state.solr_meta and table in state.carried_metadata:
            state.solr_meta[table] = state.carried_metadata[table]
    if len(state.all_candidates) > top_k:
        state.all_candidates = state.all_candidates[:top_k]
    return state.all_candidates


def show(state: P12State, initial_candidates: int) -> list[str]:
    """The candidates the agent sees now: the first ``initial_candidates`` of the pool."""
    state.visible_candidate_count = min(initial_candidates, len(state.all_candidates))
    return state.all_candidates[: state.visible_candidate_count]
