"""Fusing rankings: the helpers the hybrid retrievers share."""

from __future__ import annotations

import math
from collections.abc import Sequence

from lakegen.retrieval.models import RetrievalHit


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
