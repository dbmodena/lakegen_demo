"""Fusing two rankings into one: the rule the hybrid retrievers share.

A fused hit's score is either the weighted sum of the branches' min-max
normalised scores (``FusionMethod.WEIGHTED``, ``alpha`` on the first branch)
or the sum of their reciprocal ranks (``FusionMethod.RRF``, which ignores
``alpha``). The ranking is by fused score, then the better branch rank,
then the key, and the top ``k`` hits are ranked 1..k.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from lakegen.config.retrieval import FusionMethod, RetrievalConfig
from lakegen.retrieval.models import RetrievalHit, min_max_normalize


def best_finite_hits(hits: Sequence[RetrievalHit]) -> dict[str, RetrievalHit]:
    """Defensively collapse duplicate branch candidates to their best score."""
    best: dict[str, RetrievalHit] = {}
    for hit in hits:
        if not math.isfinite(hit.score):
            continue
        previous = best.get(hit.key)
        if previous is None or hit.score > previous.score:
            best[hit.key] = hit
    return best


def fused_score(
    config: RetrievalConfig,
    first: RetrievalHit | None,
    second: RetrievalHit | None,
    first_normalized: float,
    second_normalized: float,
) -> float:
    if config.fusion_method == FusionMethod.RRF:
        return sum(
            1.0 / (config.rrf_k + hit.rank)
            for hit in (first, second)
            if hit is not None and hit.rank > 0
        )
    return config.alpha * first_normalized + (1.0 - config.alpha) * second_normalized


def fuse(
    first: Mapping[str, RetrievalHit],
    second: Mapping[str, RetrievalHit],
    *,
    config: RetrievalConfig,
    top_k: int,
    candidate_count: int,
    first_scores: Mapping[str, float] | None = None,
    second_scores: Mapping[str, float] | None = None,
) -> list[RetrievalHit]:
    """The two branches' best hits (by key) as one ranking of at most ``top_k``.

    The first branch fills a hit's lexical slots and the second its semantic
    ones, and a table found by both keeps the first branch's document. The
    score maps default to the hits' own scores; a caller that completed a
    branch's missing signals passes the completed maps, which are what gets
    normalised. ``candidate_count`` is the branches' fetch depth, which orders
    the hits a branch did not rank last.
    """
    first_scores = dict(first_scores) if first_scores is not None else {key: hit.score for key, hit in first.items()}
    second_scores = dict(second_scores) if second_scores is not None else {key: hit.score for key, hit in second.items()}
    normalized_first = min_max_normalize(first_scores)
    normalized_second = min_max_normalize(second_scores)
    fused: list[RetrievalHit] = []
    for key in set(first) | set(second):
        first_hit, second_hit = first.get(key), second.get(key)
        first_normalized = normalized_first.get(key, 0.0)
        second_normalized = normalized_second.get(key, 0.0)
        score = fused_score(config, first_hit, second_hit, first_normalized, second_normalized)
        if not math.isfinite(score):
            continue
        fused.append(
            RetrievalHit(
                document=(first_hit or second_hit).document,
                score=score,
                lexical_score=first_scores.get(key),
                semantic_score=second_scores.get(key),
                normalized_lexical_score=first_normalized,
                normalized_semantic_score=second_normalized,
                lexical_rank=first_hit.rank if first_hit else None,
                semantic_rank=second_hit.rank if second_hit else None,
            )
        )
    no_rank = candidate_count + 1
    fused.sort(
        key=lambda hit: (
            -hit.score,
            min(hit.lexical_rank or no_rank, hit.semantic_rank or no_rank),
            hit.key,
        )
    )
    for rank, hit in enumerate(fused[:top_k], 1):
        hit.rank = rank
    return fused[:top_k]
