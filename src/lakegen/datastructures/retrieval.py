"""Retrieval results as the modes return and log them: hits, runs and their keys."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


def document_key(document: dict[str, Any]) -> str:
    """Return a stable table-document key shared by both retrieval branches."""
    for field_name in ("resource_id", "id", "dataset_id"):
        value = document.get(field_name)
        if value is not None and str(value).strip():
            return f"{field_name}:{value}"
    identity = {
        name: document.get(name) for name in ("title", "description", "dataset_url", "download_url")
    }
    encoded = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize finite scores, using 1.0 for a non-empty constant list.

    A constant list has no spread, so the usual formula is undefined. Assigning
    one preserves the fact that every item was positively retrieved by that
    branch; empty and non-finite inputs contribute no signal.

    It lives here rather than beside its first caller because ``retrievers``
    imports ``pneuma``, so a retriever that needs it cannot import it from there.

    Args:
        scores: The scores by key.

    Returns:
        The finite scores scaled to [0, 1], by key.
    """
    finite = {key: float(value) for key, value in scores.items() if math.isfinite(value)}
    if not finite:
        return {}
    low = min(finite.values())
    high = max(finite.values())
    if high == low:
        return {key: 1.0 for key in finite}
    scale = high - low
    return {key: (value - low) / scale for key, value in finite.items()}


@dataclass
class RetrievalHit:
    """One retrieved table document, with its scores and ranks from each branch.

    Attributes:
        document: The Solr (or catalogue) document.
        score: The final score the ranking is ordered by.
        rank: The 1-based rank in the final ranking.
        lexical_score: The lexical branch's score, when that branch found it.
        semantic_score: The semantic branch's score, when that branch found it.
        normalized_lexical_score: The lexical score scaled for fusion.
        normalized_semantic_score: The semantic score scaled for fusion.
        lexical_rank: The rank in the lexical branch, when that branch found it.
        semantic_rank: The rank in the semantic branch, when that branch found it.
        rerank_score: Set when a second-stage reranker scored this hit (see
            retrieval.rerank).
    """

    document: dict[str, Any]
    score: float
    rank: int = 0
    lexical_score: float | None = None
    semantic_score: float | None = None
    normalized_lexical_score: float = 0.0
    normalized_semantic_score: float = 0.0
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    rerank_score: float | None = None

    @property
    def key(self) -> str:
        """The document's stable key (``document_key``)."""
        return document_key(self.document)

    def to_log_dict(self) -> dict[str, Any]:
        """The hit for the retrieval log: its scores and the document's ids and title, not the document."""
        result = asdict(self)
        result.pop("document", None)
        result["document_key"] = self.key
        result["resource_id"] = self.document.get("resource_id")
        result["dataset_id"] = self.document.get("dataset_id")
        result["title"] = self.document.get("title")
        return result


@dataclass
class RetrievalRun:
    """One retrieval call: its configuration, outcome and hits, as the retrieval log keeps it.

    Attributes:
        mode: The retrieval mode.
        question: The question retrieved for.
        keywords: The keywords searched.
        top_k: How many hits were requested.
        representation_version: The table representation the index holds.
        embedding_model: The embedding model of the semantic branch.
        status: ``succeeded`` or ``failed``.
        error: The error, when the call failed.
        duration_seconds: How long the call took.
        lexical_query_fields: The Solr fields the lexical branch queried.
        alpha: The fusion weight; None for a mode that does not fuse.
        candidate_multiplier: How many times ``top_k`` each fused branch retrieves; None
            for a mode that does not fuse.
        missing_signal_policy: How ``hybrid`` scores a hit one branch missed; None for the
            other modes.
        fusion_method: How the branches were fused; None for a mode that does not fuse.
        rrf_k: The reciprocal-rank-fusion constant; None unless the fusion is RRF.
        job_id: The batch job the call belongs to.
        source_path: The question's path in its input file.
        source_id: The question's id in its input file.
        execution_attempt: The batch's attempt number for the question.
        experiment_id: The experiment the call belongs to.
        retrieval_attempt: The retrieval round within the run.
        rerank_model: The reranker applied to the top ``rerank_depth`` hits, if any; a
            reranker that failed leaves the retrieval order and its error here.
        rerank_depth: How many top hits the reranker scored.
        rerank_seconds: How long the reranker took.
        rerank_error: The reranker's error, when it failed.
        hits: The hits, in ranking order.
    """

    mode: str
    question: str
    keywords: list[str]
    top_k: int
    representation_version: str
    embedding_model: str
    status: str = "succeeded"
    error: str = ""
    duration_seconds: float | None = None
    lexical_query_fields: str | None = None
    alpha: float | None = None
    candidate_multiplier: int | None = None
    missing_signal_policy: str | None = None
    fusion_method: str | None = None
    rrf_k: int | None = None
    job_id: str | None = None
    source_path: str | None = None
    source_id: str | int | None = None
    execution_attempt: int | None = None
    experiment_id: str | None = None
    retrieval_attempt: int | None = None
    rerank_model: str | None = None
    rerank_depth: int | None = None
    rerank_seconds: float | None = None
    rerank_error: str = ""
    hits: list[RetrievalHit] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        """The run for the retrieval log, each hit in its log form."""
        payload = asdict(self)
        payload["hits"] = [hit.to_log_dict() for hit in self.hits]
        return payload


def hits_from_solr_response(response: dict[str, Any], limit: int) -> list[RetrievalHit]:
    """Validate scores, de-duplicate documents, and preserve branch ranks."""
    best: dict[str, RetrievalHit] = {}
    docs = response.get("response", {}).get("docs", [])
    for source_rank, document in enumerate(docs, 1):
        if not isinstance(document, dict):
            continue
        try:
            score = float(document.get("score"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        key = document_key(document)
        previous = best.get(key)
        if previous is None or score > previous.score:
            best[key] = RetrievalHit(
                document=document,
                score=score,
                rank=source_rank,
            )

    ordered = sorted(best.values(), key=lambda hit: (hit.rank, -hit.score, hit.key))
    for rank, hit in enumerate(ordered[:limit], 1):
        hit.rank = rank
    return ordered[:limit]
