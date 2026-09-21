from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any


def document_key(document: dict[str, Any]) -> str:
    """Return a stable table-document key shared by both retrieval branches."""
    for field_name in ("resource_id", "id", "dataset_id"):
        value = document.get(field_name)
        if value is not None and str(value).strip():
            return f"{field_name}:{value}"
    identity = {
        name: document.get(name)
        for name in ("title", "description", "dataset_url", "download_url")
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
    document: dict[str, Any]
    score: float
    rank: int = 0
    lexical_score: float | None = None
    semantic_score: float | None = None
    normalized_lexical_score: float = 0.0
    normalized_semantic_score: float = 0.0
    lexical_rank: int | None = None
    semantic_rank: int | None = None

    @property
    def key(self) -> str:
        return document_key(self.document)

    def finite_score(self) -> bool:
        return math.isfinite(self.score)

    def to_log_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("document", None)
        result["document_key"] = self.key
        result["resource_id"] = self.document.get("resource_id")
        result["dataset_id"] = self.document.get("dataset_id")
        result["title"] = self.document.get("title")
        return result


@dataclass
class RetrievalRun:
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
    hits: list[RetrievalHit] = field(default_factory=list)
    # The literal q_op ("AND" or "OR") and entities passed to retrieve() --
    # previously dropped before reaching RetrievalRun, so an observer could
    # not distinguish an AND search from its OR-fallback retry, or see
    # entity-mode input at all.
    q_op: str | None = None
    entities: list[str] | None = None
    # top_k above holds whatever requested_k the caller actually passed
    # (e.g. Phase12ToolsManager._search's widened fetch_k, not the
    # experiment's configured value) -- configured_top_k is the retriever's
    # own RetrievalConfig.top_k, kept separately so a UI can show both the
    # real configured width and the literal fetch width used for this call.
    configured_top_k: int | None = None

    def to_log_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hits"] = [hit.to_log_dict() for hit in self.hits]
        return payload
