"""The retrieval service: the composition root that prepares the branches, builds the mode and reranks."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from lakegen.config.retrieval import (
    FusionMethod,
    RetrievalConfig,
    RetrievalMode,
)
from lakegen.datastructures.retrieval import (
    RetrievalHit,
    RetrievalRun,
)
from lakegen.llm.embeddings import EmbeddingModel, get_embedding_model
from lakegen.llm.rerank import OCICohereReranker, Reranker, rerank_hits
from lakegen.retrieval.modes import builtin as modes  # noqa: F401  (registers the modes)
from lakegen.retrieval.modes.duckdb_agentic import DuckDBAgenticRetriever
from lakegen.retrieval.modes.duckdb_semantic import DuckDBSemanticRetriever
from lakegen.retrieval.modes.hybrid import HybridRetriever, MissingScoreResolver
from lakegen.retrieval.modes.keyword import KeywordRetriever
from lakegen.retrieval.modes.pneuma import PneumaRetriever
from lakegen.retrieval.modes.semantic import SemanticRetriever
from lakegen.retrieval.pneuma import (
    DocumentResolver,
    EntityExtractor,
    PneumaClient,
    SolrPneumaDocumentResolver,
)
from lakegen.retrieval.registry import (
    Branches,
    ModeRetriever,
    RetrievalRequest,
    build,
)
from lakegen.retrieval.solr.client import SolrClient

logger = logging.getLogger(__name__)

SEARCH_FETCH = 50
"""How many tables each branch of ``search_for`` fetches before fusion."""
RRF_K = 60


def _reciprocal_rank_fusion(branches: list[list[RetrievalHit]]) -> list[RetrievalHit]:
    """One ranking from several: each table scores the sum of its reciprocal ranks, with its best branch's hit."""
    scores: dict[str, float] = {}
    best: dict[str, RetrievalHit] = {}
    for hits in branches:
        for position, hit in enumerate(hits, 1):
            scores[hit.key] = scores.get(hit.key, 0.0) + 1.0 / (RRF_K + position)
            if hit.key not in best or position < best[hit.key].rank:
                best[hit.key] = hit
    ranked = sorted(scores, key=lambda key: -scores[key])
    return [
        replace(best[key], score=scores[key], rank=position)
        for position, key in enumerate(ranked, 1)
    ]


RunObserver = Callable[[RetrievalRun], None]


class RetrievalService:
    """The composition root of retrieval: the branches, the mode's retriever, the second stage.

    It prepares the branches the configured mode needs (Solr's BM25, the
    embedding model's KNN, the local tables for DuckDB and the Seeker, the
    Pneuma client), builds the mode's retriever from the registry, reranks
    when the mode is reranked, and records every run for the observer.

    Args:
        solr: The lake's Solr client.
        config: The retrieval configuration.
        embedding_model: The embedding model of the semantic branch; built from
            the configuration when not given.
        observer: Called with the record of every run, failed ones included.
        missing_score_resolver: How the hybrid mode scores a hit one branch missed.
        pneuma_client: The Pneuma service client; the Pneuma retriever builds its own
            when not given.
        pneuma_document_resolver: Maps Pneuma's table ids to Solr documents; a
            ``SolrPneumaDocumentResolver`` on ``solr`` when not given.
        pneuma_entity_extractor: Extracts the question's entities for the Seeker.
        table_dir: The lake's local tables, needed by the DuckDB modes and the
            Pneuma modes other than plain ``pneuma``.
        reranker: The second stage; an ``OCICohereReranker`` of the configured
            model when not given.

    Attributes:
        solr: The lake's Solr client.
        config: The retrieval configuration.
        observer: Called with the record of every run.
        reranker: The second stage, for the modes ``RetrievalMode.reranked`` names.
        last_lexical_hit_count: What the lexical branch matched in the latest
            ``retrieve()``, or None when the mode did not report one.
        keyword: The keyword branch.
        semantic: The vector branch, when the mode uses embeddings.
        hybrid: The keyword plus vector branch of the ``hybrid`` mode.
        pneuma: The Pneuma branch of the Pneuma modes.
        duckdb_agentic: The DuckDB branch of the DuckDB modes.
        duckdb_semantic: The DuckDB plus semantic branch.
        branches: The branches above, as the mode's retriever was built from them.
        mode_retriever: The retriever of the configured mode.

    Raises:
        ValueError: When a Pneuma or DuckDB mode is configured without a
            ``table_dir`` it needs.
    """

    def __init__(
        self,
        solr: SolrClient,
        config: RetrievalConfig,
        *,
        embedding_model: EmbeddingModel | None = None,
        observer: RunObserver | None = None,
        missing_score_resolver: MissingScoreResolver | None = None,
        pneuma_client: PneumaClient | None = None,
        pneuma_document_resolver: DocumentResolver | None = None,
        pneuma_entity_extractor: EntityExtractor | None = None,
        table_dir: str | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.solr = solr
        self.config = config
        self.observer = observer
        # The second stage, for the modes RetrievalMode.reranked names.
        self.reranker: Reranker | None = None
        if config.rerank_model and config.mode.reranked:
            self.reranker = reranker or OCICohereReranker(config.rerank_model)
        # What the lexical branch matched in the latest hybrid retrieve(), or
        # None when it did not report one. Fused hits cannot show this.
        self.last_lexical_hit_count: int | None = None
        self.keyword = KeywordRetriever(
            solr,
            query_fields=config.lexical_query_fields,
            or_fallback=config.keyword_or_fallback,
        )
        self.semantic: SemanticRetriever | None = None
        if config.mode.uses_embeddings:
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
        # Both Pneuma modalities are the same retriever; it reads the mode to
        # decide whether content search and enumeration run at all.
        self.pneuma = (
            PneumaRetriever(
                config,
                pneuma_document_resolver or SolrPneumaDocumentResolver(solr),
                client=pneuma_client,
                table_dir=table_dir,
                entity_extractor=pneuma_entity_extractor,
            )
            if config.mode.is_pneuma
            and (table_dir is not None or config.mode is RetrievalMode.PNEUMA)
            else None
        )
        if config.mode.is_pneuma and self.pneuma is None:
            raise ValueError(f"{config.mode.value} retrieval requires a local table_dir")
        uses_duckdb = config.mode.uses_duckdb
        self.duckdb_agentic = (
            DuckDBAgenticRetriever(config, table_dir)
            if uses_duckdb and table_dir is not None
            else None
        )
        if uses_duckdb and self.duckdb_agentic is None:
            raise ValueError(f"{config.mode.value} retrieval requires a local table_dir")
        self.duckdb_semantic = (
            DuckDBSemanticRetriever(
                self.duckdb_agentic,
                self.semantic,
                config,
                pneuma_document_resolver or SolrPneumaDocumentResolver(solr),
            )
            if config.mode == RetrievalMode.HYBRID_DUCKDB_SEMANTIC
            and self.duckdb_agentic is not None
            and self.semantic is not None
            else None
        )
        self.branches = Branches(
            config=config,
            keyword=self.keyword,
            semantic=self.semantic,
            hybrid=self.hybrid,
            pneuma=self.pneuma,
            duckdb_agentic=self.duckdb_agentic,
            duckdb_semantic=self.duckdb_semantic,
        )
        self.mode_retriever: ModeRetriever = build(config.mode, self.branches)

    def retrieve(
        self,
        *,
        question: str,
        keywords: Sequence[str],
        top_k: int | None = None,
        lexical_fetch_k: int | None = None,
        q_op: str = "AND",
        entities: Sequence[str] | None = None,
    ) -> list[RetrievalHit]:
        """Rank the lake's tables for a question with the configured mode, then rerank.

        Every call, failed ones included, is recorded and passed to the observer.

        Args:
            question: The question.
            keywords: The search keywords.
            top_k: How many tables to return; the configured ``top_k`` when not given.
            lexical_fetch_k: How many lexical hits to fetch before cutting to ``top_k``.
            q_op: The Solr query operator joining the keywords.
            entities: Named entities of the question, for the modes that search contents.

        Returns:
            The ranked tables, best first.
        """
        requested_k = top_k or self.config.top_k
        started = time.monotonic()
        self.last_lexical_hit_count = None
        request = RetrievalRequest(
            question=question,
            keywords=tuple(keywords),
            top_k=requested_k,
            lexical_fetch_k=lexical_fetch_k,
            q_op=q_op,
            entities=tuple(entities or ()),
        )
        try:
            result = self.mode_retriever.retrieve(request)
            hits = result.hits
            self.last_lexical_hit_count = result.lexical_hit_count
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

        hits, rerank = self._rerank(question, hits)
        run = self._run_record(
            question=question,
            keywords=keywords,
            requested_k=requested_k,
            hits=hits,
            duration_seconds=time.monotonic() - started,
            rerank=rerank,
        )
        if self.observer is not None:
            self.observer(run)
        return hits

    def _rerank(
        self, question: str, hits: list[RetrievalHit]
    ) -> tuple[list[RetrievalHit], dict[str, Any] | None]:
        """Reorder the top ``rerank_depth`` hits, and what to record about it.

        A reranker that fails leaves the retrieval order: the search itself
        succeeded, and an outage of the second stage must not take it down.

        Args:
            question: The question the hits were retrieved for.
            hits: The retrieved tables, best first.

        Returns:
            The hits in their new order, and the rerank fields of the run record
            (None when nothing was reranked).
        """
        if self.reranker is None or not hits:
            return hits, None
        record: dict[str, Any] = {
            "rerank_model": self.config.rerank_model,
            "rerank_depth": self.config.rerank_depth,
        }
        started = time.monotonic()
        try:
            hits = rerank_hits(question, hits, self.reranker, depth=self.config.rerank_depth)
        except Exception as exc:
            record["rerank_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Reranking failed; keeping the retrieval order: %s", exc)
        record["rerank_seconds"] = round(time.monotonic() - started, 6)
        return hits, record

    def search_for(self, text: str, words: Sequence[str], *, top_k: int) -> list[RetrievalHit]:
        """Search the catalogue for one requirement: the tables that mention any of its words or mean its text.

        The lexical branch runs with OR over the words and the semantic branch
        over the text, in the modes that have them; the two rankings are fused
        by reciprocal rank and reranked on the text. The run is recorded and
        passed to the observer like ``retrieve``'s.

        Args:
            text: What is looked for, as the agent wrote it.
            words: Its words, for the lexical branch.
            top_k: How many tables to return.

        Returns:
            The ranked tables, best first.
        """
        started = time.monotonic()
        branches: list[list[RetrievalHit]] = []
        if words and self.config.mode in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID):
            branches.append(self.keyword.retrieve(list(words), top_k=SEARCH_FETCH, q_op="OR"))
        if self.semantic is not None and text.strip():
            branches.append(self.semantic.retrieve(text, top_k=SEARCH_FETCH))
        hits = _reciprocal_rank_fusion(branches)
        hits, rerank = self._rerank(text, hits[: max(top_k, self.config.rerank_depth)])
        hits = [replace(hit, rank=position) for position, hit in enumerate(hits[:top_k], 1)]
        run = self._run_record(
            question=text,
            keywords=words,
            requested_k=top_k,
            hits=hits,
            duration_seconds=time.monotonic() - started,
            rerank=rerank,
        )
        if self.observer is not None:
            self.observer(run)
        return hits

    def lexical_match_count(self, keywords: Sequence[str]) -> int:
        """How many tables contain every keyword; a diagnostic, not a retrieval run."""
        return self.keyword.count(keywords)

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
        rerank: dict[str, Any] | None = None,
    ) -> RetrievalRun:
        fuses = self.config.mode.fuses
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
            alpha=self.config.alpha if fuses else None,
            candidate_multiplier=self.config.candidate_multiplier if fuses else None,
            # Only ``hybrid`` can rescore a missing signal; the DuckDB + semantic
            # hybrid always scores it 0.
            missing_signal_policy=(
                self.config.missing_signal_policy
                if self.config.mode == RetrievalMode.HYBRID
                else None
            ),
            fusion_method=self.config.fusion_method if fuses else None,
            rrf_k=(
                self.config.rrf_k
                if fuses and self.config.fusion_method == FusionMethod.RRF
                else None
            ),
            **(rerank or {}),
            hits=hits,
        )
