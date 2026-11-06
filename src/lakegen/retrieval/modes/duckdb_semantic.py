"""DuckDB-semantic retrieval: the agentic DuckDB search fused with the semantic ranking."""

from __future__ import annotations

from collections.abc import Sequence

from lakegen.config.retrieval import (
    RetrievalConfig,
)
from lakegen.datastructures.retrieval import (
    RetrievalHit,
    document_key,
)
from lakegen.retrieval.modes.duckdb_agentic import DuckDBAgenticRetriever, evidence_note
from lakegen.retrieval.modes.fusion import best_finite_hits, fuse
from lakegen.retrieval.modes.semantic import SemanticRetriever
from lakegen.retrieval.pneuma import (
    DocumentResolver,
)


class DuckDBSemanticRetriever:
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

    Args:
        duckdb_branch: The DuckDB search over the local files.
        semantic: The KNN branch.
        config: The retrieval configuration (fusion method, alpha, rrf_k,
            candidate multiplier).
        resolver: Maps a file's resource id to its catalog document.

    Attributes:
        duckdb: The DuckDB branch.
        semantic: The KNN branch.
        config: The retrieval configuration.
        resolver: Maps a file's resource id to its catalog document.
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
                by_key[key] = RetrievalHit(document=document, score=hit.score, rank=hit.rank)
        return by_key

    def retrieve(
        self,
        question: str,
        keywords: Sequence[str],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        """Fuse the DuckDB and semantic rankings into at most ``top_k`` hits.

        Args:
            question: The question.
            keywords: The keywords DuckDB searches the files for.
            top_k: How many tables to return.

        Returns:
            The fused ranking.
        """
        candidate_count = top_k * self.config.candidate_multiplier
        duckdb_by_key = self._duckdb_hits(question, keywords, candidate_count)
        semantic_by_key = best_finite_hits(self.semantic.retrieve(question, top_k=candidate_count))
        # The DuckDB document is the catalog one plus DuckDB's evidence, so it
        # is the richer of the two whenever both branches found the table: the
        # first branch's document wins in the fusion.
        return fuse(
            duckdb_by_key,
            semantic_by_key,
            config=self.config,
            top_k=top_k,
            candidate_count=candidate_count,
        )


# --- the modes: one retriever per RetrievalMode, composed from the branches --
