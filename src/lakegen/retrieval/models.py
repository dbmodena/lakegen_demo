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
    lexical_query_fields: str | None = None
    alpha: float | None = None
    candidate_multiplier: int | None = None
    missing_signal_policy: str | None = None
    hits: list[RetrievalHit] = field(default_factory=list)

    def to_log_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hits"] = [hit.to_log_dict() for hit in self.hits]
        return payload
