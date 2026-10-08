"""DuckDB-semantic retrieval: the agentic DuckDB search fused with the semantic ranking."""

from __future__ import annotations

from collections.abc import Sequence

from lakegen.retrieval.config import (
    FusionMethod,
    RetrievalConfig,
)
from lakegen.retrieval.duckdb_agentic import DuckDBAgenticRetriever, evidence_note
from lakegen.retrieval.fusion import best_finite_hits
from lakegen.retrieval.models import (
    RetrievalHit,
    document_key,
    min_max_normalize,
)
from lakegen.retrieval.pneuma import (
    DocumentResolver,
)
from lakegen.retrieval.semantic import SemanticRetriever


class DuckDBSemanticHybridRetriever:
    """Fusion of the DuckDB Parquet search and dense KNN.

    ``hybrid`` fuses BM25 with KNN; this fuses the ``duckdb_agentic`` ranking
    with KNN instead, by ``fusion_method`` as ``hybrid`` does: ``rrf`` sums
    ``1 / (rrf_k + rank)`` over the branches, ``weighted`` min-max normalizes
    each branch's scores and takes ``alpha * duckdb + (1 - alpha) * semantic``.
    A table missing from a branch scores 0 there. Each branch retrieves
    ``top_k * candidate_multiplier`` candidates. The DuckDB branch fills the
    lexical fields of a hit, KNN the semantic ones.

    DuckDB names a table by its Parquet file and Solr by its catalog document,
    so each DuckDB hit is resolved to the catalog document of the same file
    before fusion; otherwise one table would count once per branch and never
    fuse. A file without a catalog document keeps DuckDB's own. DuckDB's
    evidence stays on the fused hit, as it does in ``duckdb_agentic``.
    """

    def __init__(
        self,
        duckdb_branch: DuckDBAgenticRetriever,
        semantic: SemanticRetriever,
        config: RetrievalConfig,
        resolver: DocumentResolver,
    ) -> None:
        self.duckdb = duckdb_branch
        self.semantic = semantic
        self.config = config
        self.resolver = resolver

    def _duckdb_hits(
        self, question: str, keywords: Sequence[str], candidate_count: int
    ) -> dict[str, RetrievalHit]:
        """DuckDB's hits keyed like the semantic branch's, best rank first."""
        by_key: dict[str, RetrievalHit] = {}
        for hit in self.duckdb.retrieve(question, keywords, top_k=candidate_count):
            document = hit.document
            catalog = self.resolver(str(document.get("resource_id") or ""))
            if catalog is not None:
                evidence = document["duckdb_evidence"]
                description = str(catalog.get("description") or "").strip()
                document = {
                    **catalog,
                    "description": (description + " " if description else "")
                    + evidence_note(evidence),
                    "duckdb_evidence": evidence,
                }
            key = document_key(document)
            if key not in by_key:
                by_key[key] = RetrievalHit(
                    document=document, score=hit.score, rank=hit.rank
                )
        return by_key

    def retrieve(
        self,
        question: str,
        keywords: Sequence[str],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        candidate_count = top_k * self.config.candidate_multiplier
        duckdb_by_key = self._duckdb_hits(question, keywords, candidate_count)
        semantic_by_key = best_finite_hits(
            self.semantic.retrieve(question, top_k=candidate_count)
        )
        duckdb_scores = {key: hit.score for key, hit in duckdb_by_key.items()}
        semantic_scores = {key: hit.score for key, hit in semantic_by_key.items()}
        normalized_duckdb = min_max_normalize(duckdb_scores)
        normalized_semantic = min_max_normalize(semantic_scores)

        fused: list[RetrievalHit] = []
        for key in set(duckdb_by_key) | set(semantic_by_key):
            duckdb_hit = duckdb_by_key.get(key)
            semantic_hit = semantic_by_key.get(key)
            # The DuckDB document is the catalog one plus DuckDB's evidence, so
            # it is the richer of the two whenever both branches found the table.
            document = (duckdb_hit or semantic_hit).document
            if self.config.fusion_method == FusionMethod.RRF:
                score = sum(
                    1.0 / (self.config.rrf_k + branch_hit.rank)
                    for branch_hit in (duckdb_hit, semantic_hit)
                    if branch_hit is not None and branch_hit.rank > 0
                )
            else:
                score = (
                    self.config.alpha * normalized_duckdb.get(key, 0.0)
                    + (1.0 - self.config.alpha) * normalized_semantic.get(key, 0.0)
                )
            fused.append(
                RetrievalHit(
                    document=document,
                    score=score,
                    lexical_score=duckdb_scores.get(key),
                    semantic_score=semantic_scores.get(key),
                    normalized_lexical_score=normalized_duckdb.get(key, 0.0),
                    normalized_semantic_score=normalized_semantic.get(key, 0.0),
                    lexical_rank=duckdb_hit.rank if duckdb_hit else None,
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


# --- the modes: one retriever per RetrievalMode, composed from the branches --
