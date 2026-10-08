"""Hybrid retrieval: the keyword and semantic rankings fused, weighted or by reciprocal rank."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from lakegen.retrieval.config import (
    FusionMethod,
    MissingSignalPolicy,
    RetrievalConfig,
)
from lakegen.retrieval.fusion import best_finite_hits, fuse
from lakegen.retrieval.keyword import KeywordRetriever
from lakegen.retrieval.models import (
    RetrievalHit,
)
from lakegen.retrieval.semantic import SemanticRetriever

MissingScoreResolver = Callable[
    [str, dict[str, Any], str, Sequence[str]], float | None
]


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
        return self.retrieve_with_lexical_count(question, keywords, top_k=top_k)[0]

    def retrieve_with_lexical_count(
        self,
        question: str,
        keywords: Sequence[str],
        *,
        top_k: int,
    ) -> tuple[list[RetrievalHit], int | None]:
        """Fuse both branches and report how many tables the lexical one matched.

        Fusion returns tables even when the strict-AND lexical branch matches
        none, and truncating to ``top_k`` can drop every lexical hit from the
        result, so the fused hits cannot say whether the keywords matched. The
        count is ``None`` when the lexical branch was not run (``alpha == 0``).
        """
        # Weighted fusion has two exact, useful boundary conditions.  Returning
        # the selected branch directly avoids normalization/tie-breaking from
        # perturbing its documents, order, scores, ranks, or top_k.  RRF is a
        # separate rank-fusion baseline and intentionally ignores alpha.
        if self.config.fusion_method == FusionMethod.WEIGHTED:
            if self.config.alpha == 1.0:
                lexical_hits = self.lexical.retrieve(keywords, top_k=top_k)
                return lexical_hits[:top_k], len(lexical_hits)
            if self.config.alpha == 0.0:
                return self.semantic.retrieve(question, top_k=top_k)[:top_k], None

        candidate_count = top_k * self.config.candidate_multiplier
        lexical_hits = self.lexical.retrieve(keywords, top_k=candidate_count)
        semantic_hits = self.semantic.retrieve(question, top_k=candidate_count)

        lexical_by_key = best_finite_hits(lexical_hits)
        semantic_by_key = best_finite_hits(semantic_hits)
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

        fused = fuse(
            lexical_by_key, semantic_by_key, config=self.config, top_k=top_k,
            candidate_count=candidate_count, first_scores=lexical_scores, second_scores=semantic_scores,
        )
        return fused, len(lexical_hits)
