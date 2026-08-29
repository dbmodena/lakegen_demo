"""Table retrievers implemented on top of Apache Solr/Lucene."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import time
from typing import Any

from src.client_solr import LocalSolrClient

from lakegen.retrieval.config import (
    FusionMethod,
    MissingSignalPolicy,
    RetrievalConfig,
    RetrievalMode,
)
from lakegen.retrieval.embeddings import EmbeddingModel, get_embedding_model
from lakegen.retrieval.duckdb_agentic import DuckDBAgenticRetriever
from lakegen.retrieval.models import RetrievalHit, RetrievalRun, document_key
from lakegen.retrieval.pneuma import (
    DocumentResolver,
    PneumaClient,
    PneumaRetriever,
    SolrPneumaDocumentResolver,
)


MissingScoreResolver = Callable[
    [str, dict[str, Any], str, Sequence[str]], float | None
]
RunObserver = Callable[[RetrievalRun], None]


def _response_hits(response: dict[str, Any], limit: int) -> list[RetrievalHit]:
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


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize finite scores, using 1.0 for a non-empty constant list.

    A constant list has no spread, so the usual formula is undefined. Assigning
    one preserves the fact that every item was positively retrieved by that
    branch; empty and non-finite inputs contribute no signal.
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


def _best_finite_hits(hits: Sequence[RetrievalHit]) -> dict[str, RetrievalHit]:
    """Defensively collapse duplicate branch candidates to their best score."""
    best: dict[str, RetrievalHit] = {}
    for hit in hits:
        if not math.isfinite(hit.score):
            continue
        previous = best.get(hit.key)
        if previous is None or hit.score > previous.score:
            best[hit.key] = hit
    return best


class KeywordRetriever:
    """Keyword-based table retrieval via Solr/Lucene BM25.

    LakeGen deliberately delegates BM25 scoring to Solr/Lucene and preserves
    the baseline eDisMax and query-operator settings. Phase 1 keywords remain
    the query. Field weighting can still be configured in Solr without a local
    reimplementation of the BM25 formula.

    Reference: Robertson and Zaragoza, 2009, DOI: 10.1561/1500000019.
    """

    def __init__(
        self,
        solr: LocalSolrClient,
        *,
        query_fields: str | None = None,
    ) -> None:
        self.solr = solr
        self.query_fields = query_fields

    def retrieve(
        self,
        keywords: Sequence[str],
        *,
        top_k: int,
        q_op: str = "AND",
    ) -> list[RetrievalHit]:
        if isinstance(keywords, str):
            raise TypeError("keywords must be a sequence, not a string")
        clean_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        if not clean_keywords:
            return []
        params: dict[str, Any] = {
            "q_op": q_op,
            "rows": top_k,
            "fl": "*,score",
        }
        # Omit qf by default so the pre-existing Solr baseline is unchanged.
        if self.query_fields is not None:
            params["qf"] = self.query_fields
        response = self.solr.select(tokens=clean_keywords, **params)
        hits = _response_hits(response, top_k)
        for hit in hits:
            hit.lexical_score = hit.score
            hit.lexical_rank = hit.rank
        return hits


class SemanticRetriever:
    """Dense retriever inspired by DPR's dual-encoder paradigm.

    The natural-language question and the versioned table representation are
    embedded separately with one fixed pretrained multilingual model. Solr KNN
    performs cosine retrieval. This does not reproduce DPR training, paired
    encoders, in-batch negatives, or hard-negative mining.

    References: Karpukhin et al., 2020, DOI: 10.18653/v1/2020.emnlp-main.550;
    Ji et al., TARGET.
    """

    def __init__(
        self,
        solr: LocalSolrClient,
        config: RetrievalConfig,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.solr = solr
        self.config = config
        self.embedding_model = embedding_model

    @staticmethod
    def _quoted_filter(field: str, value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'{field}:"{escaped}"'

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        vector = self.embedding_model.encode_query(question)
        response = self.solr.knn_select(
            vector,
            vector_field=self.config.vector_field,
            top_k=top_k,
            rows=top_k,
            filters=(
                self._quoted_filter(
                    "representation_version", self.config.representation_version
                ),
                self._quoted_filter("embedding_model", self.config.embedding_model),
            ),
        )
        hits = _response_hits(response, top_k)
        for hit in hits:
            hit.semantic_score = hit.score
            hit.semantic_rank = hit.rank
        return hits


class HybridRetriever:
    """Pneuma-inspired hybrid table retriever.

    It retrieves ``candidate_multiplier * top_k`` candidates independently
    from BM25 and dense KNN, normalizes each finite score list separately, and
    applies a configurable weighted fusion.

    Reference: Balaka et al., 2025, DOI: 10.1145/3725337.
    """

    def __init__(
        self,
        lexical: KeywordRetriever,
        semantic: SemanticRetriever,
        config: RetrievalConfig,
        *,
        missing_score_resolver: MissingScoreResolver | None = None,
    ) -> None:
        self.lexical = lexical
        self.semantic = semantic
        self.config = config
        self.missing_score_resolver = missing_score_resolver

    def _resolve_missing(
        self,
        branch: str,
        document: dict[str, Any],
        question: str,
        keywords: Sequence[str],
    ) -> float | None:
        if self.config.missing_signal_policy == MissingSignalPolicy.ZERO:
            return None
        if self.missing_score_resolver is None:
            raise RuntimeError(
                "missing_signal_policy='rescore' requires a reliable "
                "missing_score_resolver for this Solr schema"
            )
        value = self.missing_score_resolver(branch, document, question, keywords)
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def retrieve(
        self,
        question: str,
        keywords: Sequence[str],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        # Weighted fusion has two exact, useful boundary conditions.  Returning
        # the selected branch directly avoids normalization/tie-breaking from
        # perturbing its documents, order, scores, ranks, or top_k.  RRF is a
        # separate rank-fusion baseline and intentionally ignores alpha.
        if self.config.fusion_method == FusionMethod.WEIGHTED:
            if self.config.alpha == 1.0:
                return self.lexical.retrieve(keywords, top_k=top_k)[:top_k]
            if self.config.alpha == 0.0:
                return self.semantic.retrieve(question, top_k=top_k)[:top_k]

        candidate_count = top_k * self.config.candidate_multiplier
        lexical_hits = self.lexical.retrieve(keywords, top_k=candidate_count)
        semantic_hits = self.semantic.retrieve(question, top_k=candidate_count)

        lexical_by_key = _best_finite_hits(lexical_hits)
        semantic_by_key = _best_finite_hits(semantic_hits)
        all_keys = set(lexical_by_key) | set(semantic_by_key)
        documents = {
            key: (lexical_by_key.get(key) or semantic_by_key[key]).document
            for key in all_keys
        }

        lexical_scores = {key: hit.score for key, hit in lexical_by_key.items()}
        semantic_scores = {key: hit.score for key, hit in semantic_by_key.items()}
        for key in all_keys:
            if key not in lexical_scores:
                score = self._resolve_missing(
                    "lexical", documents[key], question, keywords
                )
                if score is not None:
                    lexical_scores[key] = score
            if key not in semantic_scores:
                score = self._resolve_missing(
                    "semantic", documents[key], question, keywords
                )
                if score is not None:
                    semantic_scores[key] = score

        normalized_lexical = min_max_normalize(lexical_scores)
        normalized_semantic = min_max_normalize(semantic_scores)
        fused: list[RetrievalHit] = []
        for key in all_keys:
            lexical_score = lexical_scores.get(key)
            semantic_score = semantic_scores.get(key)
            lexical_normalized = normalized_lexical.get(key, 0.0)
            semantic_normalized = normalized_semantic.get(key, 0.0)
            lexical_hit = lexical_by_key.get(key)
            semantic_hit = semantic_by_key.get(key)
            if self.config.fusion_method == FusionMethod.RRF:
                score = sum(
                    1.0 / (self.config.rrf_k + branch_hit.rank)
                    for branch_hit in (lexical_hit, semantic_hit)
                    if branch_hit is not None and branch_hit.rank > 0
                )
            else:
                score = (
                    self.config.alpha * lexical_normalized
                    + (1.0 - self.config.alpha) * semantic_normalized
                )
            if not math.isfinite(score):
                continue
            fused.append(
                RetrievalHit(
                    document=documents[key],
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    normalized_lexical_score=lexical_normalized,
                    normalized_semantic_score=semantic_normalized,
                    lexical_rank=lexical_hit.rank if lexical_hit else None,
                    semantic_rank=semantic_hit.rank if semantic_hit else None,
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


class TableRetrievalService:
    """Uniform entry point used by LakeGen Phase 2 and benchmark code."""

    def __init__(
        self,
        solr: LocalSolrClient,
        config: RetrievalConfig,
        *,
        embedding_model: EmbeddingModel | None = None,
        observer: RunObserver | None = None,
        missing_score_resolver: MissingScoreResolver | None = None,
        pneuma_client: PneumaClient | None = None,
        pneuma_document_resolver: DocumentResolver | None = None,
        table_dir: str | None = None,
    ) -> None:
        self.solr = solr
        self.config = config
        self.observer = observer
        self.keyword = KeywordRetriever(
            solr,
            query_fields=config.lexical_query_fields,
        )
        self.semantic: SemanticRetriever | None = None
        if config.mode in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID):
            model = embedding_model or get_embedding_model(
                config.embedding_model, config.embedding_base_url
            )
            self.semantic = SemanticRetriever(solr, config, model)
        self.hybrid = (
            HybridRetriever(
                self.keyword,
                self.semantic,
                config,
                missing_score_resolver=missing_score_resolver,
            )
            if self.semantic is not None and config.mode == RetrievalMode.HYBRID
            else None
        )
        self.pneuma = (
            PneumaRetriever(
                config,
                pneuma_document_resolver or SolrPneumaDocumentResolver(solr),
                client=pneuma_client,
            )
            if config.mode == RetrievalMode.PNEUMA
            else None
        )
        self.duckdb_agentic = (
            DuckDBAgenticRetriever(config, table_dir)
            if config.mode == RetrievalMode.DUCKDB_AGENTIC and table_dir is not None
            else None
        )
        if config.mode == RetrievalMode.DUCKDB_AGENTIC and self.duckdb_agentic is None:
            raise ValueError("duckdb_agentic retrieval requires a local table_dir")

    def retrieve(
        self,
        *,
        question: str,
        keywords: Sequence[str],
        top_k: int | None = None,
        lexical_fetch_k: int | None = None,
        q_op: str = "AND",
    ) -> list[RetrievalHit]:
        requested_k = top_k or self.config.top_k
        started = time.monotonic()
        try:
            if self.config.mode == RetrievalMode.KEYWORD:
                fetch_k = max(lexical_fetch_k or requested_k, requested_k)
                hits = self.keyword.retrieve(
                    keywords,
                    top_k=fetch_k,
                    q_op=q_op,
                )[:requested_k]
                for rank, hit in enumerate(hits, 1):
                    hit.rank = rank
                    hit.lexical_rank = rank
            elif self.config.mode == RetrievalMode.SEMANTIC:
                assert self.semantic is not None
                hits = self.semantic.retrieve(question, top_k=requested_k)
            elif self.config.mode == RetrievalMode.HYBRID:
                assert self.hybrid is not None
                hits = self.hybrid.retrieve(question, keywords, top_k=requested_k)
            elif self.config.mode == RetrievalMode.PNEUMA:
                assert self.pneuma is not None
                hits = self.pneuma.retrieve(question, top_k=requested_k)
            else:
                assert self.duckdb_agentic is not None
                hits = self.duckdb_agentic.retrieve(
                    question, keywords, top_k=requested_k
                )
        except Exception as exc:
            run = self._run_record(
                question=question,
                keywords=keywords,
                requested_k=requested_k,
                hits=[],
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=time.monotonic() - started,
            )
            if self.observer is not None:
                self.observer(run)
            raise

        run = self._run_record(
            question=question,
            keywords=keywords,
            requested_k=requested_k,
            hits=hits,
            duration_seconds=time.monotonic() - started,
        )
        if self.observer is not None:
            self.observer(run)
        return hits

    def _run_record(
        self,
        *,
        question: str,
        keywords: Sequence[str],
        requested_k: int,
        hits: list[RetrievalHit],
        status: str = "succeeded",
        error: str = "",
        duration_seconds: float,
    ) -> RetrievalRun:
        return RetrievalRun(
            mode=self.config.mode,
            question=question,
            keywords=list(keywords),
            top_k=requested_k,
            representation_version=self.config.representation_version,
            embedding_model=self.config.embedding_model,
            status=status,
            error=error,
            duration_seconds=round(duration_seconds, 6),
            lexical_query_fields=self.config.lexical_query_fields,
            alpha=self.config.alpha if self.config.mode == RetrievalMode.HYBRID else None,
            candidate_multiplier=(
                self.config.candidate_multiplier
                if self.config.mode == RetrievalMode.HYBRID
                else None
            ),
            missing_signal_policy=(
                self.config.missing_signal_policy
                if self.config.mode == RetrievalMode.HYBRID
                else None
            ),
            fusion_method=(
                self.config.fusion_method
                if self.config.mode == RetrievalMode.HYBRID
                else None
            ),
            rrf_k=(
                self.config.rrf_k
                if self.config.mode == RetrievalMode.HYBRID
                and self.config.fusion_method == FusionMethod.RRF
                else None
            ),
            hits=hits,
        )
