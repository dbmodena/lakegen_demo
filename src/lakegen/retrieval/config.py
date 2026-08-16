"""Configuration shared by keyword, semantic, and hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import os


class RetrievalMode(StrEnum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    PNEUMA = "pneuma"


class MissingSignalPolicy(StrEnum):
    ZERO = "zero"
    RESCORE = "rescore"


class FusionMethod(StrEnum):
    WEIGHTED = "weighted"
    RRF = "rrf"


@dataclass(frozen=True)
class RetrievalConfig:
    """Reproducible retrieval settings for a LakeGen experiment."""

    mode: RetrievalMode = RetrievalMode.KEYWORD
    top_k: int = 10
    alpha: float = 0.5
    candidate_multiplier: int = 5
    representation_version: str = "metadata-v1"
    embedding_model: str = "bge-m3"
    embedding_base_url: str = "http://localhost:11434"
    vector_field: str = "table_embedding"
    lexical_query_fields: str | None = None
    missing_signal_policy: MissingSignalPolicy = MissingSignalPolicy.ZERO
    fusion_method: FusionMethod = FusionMethod.WEIGHTED
    rrf_k: int = 60
    pneuma_index_name: str = "lakegen"
    pneuma_base_url: str = "http://localhost:8765"
    pneuma_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RetrievalMode(self.mode))
        object.__setattr__(
            self,
            "missing_signal_policy",
            MissingSignalPolicy(self.missing_signal_policy),
        )
        object.__setattr__(self, "fusion_method", FusionMethod(self.fusion_method))
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if self.pneuma_timeout_seconds <= 0:
            raise ValueError("pneuma_timeout_seconds must be greater than zero")
        for name in ("representation_version", "embedding_model", "vector_field"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        for name in (
            "pneuma_index_name",
            "pneuma_base_url",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if self.lexical_query_fields is not None:
            query_fields = self.lexical_query_fields.strip()
            object.__setattr__(self, "lexical_query_fields", query_fields or None)

    @property
    def branch_candidate_count(self) -> int:
        return self.top_k * self.candidate_multiplier

    def with_mode(self, mode: RetrievalMode | str) -> "RetrievalConfig":
        return replace(self, mode=RetrievalMode(mode))

    @classmethod
    def from_env(
        cls,
        *,
        mode: RetrievalMode | str | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        candidate_multiplier: int | None = None,
    ) -> "RetrievalConfig":
        selected_mode = mode or os.environ.get(
            "LAKEGEN_RETRIEVAL_MODE", RetrievalMode.KEYWORD
        )
        return cls(
            mode=RetrievalMode(selected_mode),
            top_k=top_k
            if top_k is not None
            else int(os.environ.get("LAKEGEN_RETRIEVAL_TOP_K", "10")),
            alpha=alpha
            if alpha is not None
            else float(os.environ.get("LAKEGEN_HYBRID_ALPHA", "0.5")),
            candidate_multiplier=candidate_multiplier
            if candidate_multiplier is not None
            else int(os.environ.get("LAKEGEN_CANDIDATE_MULTIPLIER", "5")),
            representation_version=os.environ.get(
                "LAKEGEN_REPRESENTATION_VERSION", "metadata-v1"
            ),
            embedding_model=os.environ.get(
                "LAKEGEN_EMBEDDING_MODEL", "bge-m3"
            ),
            embedding_base_url=os.environ.get(
                "LAKEGEN_EMBEDDING_BASE_URL", "http://localhost:11434"
            ),
            vector_field=os.environ.get(
                "LAKEGEN_VECTOR_FIELD", "table_embedding"
            ),
            lexical_query_fields=os.environ.get("LAKEGEN_BM25_QUERY_FIELDS"),
            missing_signal_policy=MissingSignalPolicy(
                os.environ.get("LAKEGEN_MISSING_SIGNAL_POLICY", "zero")
            ),
            fusion_method=FusionMethod(
                os.environ.get("LAKEGEN_FUSION_METHOD", "weighted")
            ),
            rrf_k=int(os.environ.get("LAKEGEN_RRF_K", "60")),
            pneuma_index_name=os.environ.get("LAKEGEN_PNEUMA_INDEX_NAME", "lakegen"),
            pneuma_base_url=os.environ.get(
                "LAKEGEN_PNEUMA_BASE_URL", "http://localhost:8765"
            ),
            pneuma_timeout_seconds=float(
                os.environ.get("LAKEGEN_PNEUMA_TIMEOUT_SECONDS", "120")
            ),
        )
