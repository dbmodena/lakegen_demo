"""The registered retrieval modes: how each composes its retriever from the service's branches."""

from __future__ import annotations

from lakegen.config.retrieval import (
    RetrievalMode,
)
from lakegen.retrieval.registry import (
    Branches,
    RetrievalRequest,
    RetrievalResult,
    register,
)


@register(RetrievalMode.KEYWORD)
class KeywordMode:
    """BM25 over the keywords; the lexical rank is the rank."""

    def __init__(self, branches: Branches) -> None:
        self.keyword = branches.keyword

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Fetch ``lexical_fetch_k`` hits (at least ``top_k``) and keep the first ``top_k``."""
        fetch_k = max(request.lexical_fetch_k or request.top_k, request.top_k)
        hits = self.keyword.retrieve(list(request.keywords), top_k=fetch_k, q_op=request.q_op)[
            : request.top_k
        ]
        for rank, hit in enumerate(hits, 1):
            hit.rank = rank
            hit.lexical_rank = rank
        return RetrievalResult(hits)


@register(RetrievalMode.SEMANTIC)
class SemanticMode:
    """KNN over the question's embedding."""

    def __init__(self, branches: Branches) -> None:
        self.semantic = branches.require("semantic")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """The ``top_k`` nearest tables to the question."""
        return RetrievalResult(self.semantic.retrieve(request.question, top_k=request.top_k))


@register(RetrievalMode.HYBRID)
class HybridMode:
    """BM25 and KNN fused; reports what the lexical branch matched."""

    def __init__(self, branches: Branches) -> None:
        self.hybrid = branches.require("hybrid")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """The fused ranking, with the lexical branch's match count."""
        hits, lexical_hit_count = self.hybrid.retrieve_with_lexical_count(
            request.question, list(request.keywords), top_k=request.top_k
        )
        return RetrievalResult(hits, lexical_hit_count)


@register(RetrievalMode.PNEUMA, RetrievalMode.PNEUMA_SEEKER)
class PneumaMode:
    """Pneuma's index, with the Seeker's content search when the mode asks for it."""

    def __init__(self, branches: Branches) -> None:
        self.pneuma = branches.require("pneuma")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Pneuma's ranking for the question, its entities and keywords."""
        return RetrievalResult(
            self.pneuma.retrieve(
                request.question,
                top_k=request.top_k,
                entities=list(request.entities) or None,
                keywords=list(request.keywords),
            )
        )


@register(RetrievalMode.DUCKDB_AGENTIC)
class DuckDBMode:
    """DuckDB over the local Parquet files."""

    def __init__(self, branches: Branches) -> None:
        self.duckdb = branches.require("duckdb_agentic")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """The tables whose files DuckDB matched the keywords in."""
        return RetrievalResult(
            self.duckdb.retrieve(request.question, list(request.keywords), top_k=request.top_k)
        )


@register(RetrievalMode.HYBRID_DUCKDB_SEMANTIC)
class DuckDBSemanticMode:
    """DuckDB and KNN fused."""

    def __init__(self, branches: Branches) -> None:
        self.fused = branches.require("duckdb_semantic")

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """The DuckDB and semantic rankings fused."""
        return RetrievalResult(
            self.fused.retrieve(request.question, list(request.keywords), top_k=request.top_k)
        )
