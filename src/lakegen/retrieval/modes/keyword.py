"""Keyword retrieval: Solr's BM25 over the catalogue, AND first and OR as a fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lakegen.datastructures.retrieval import (
    RetrievalHit,
    hits_from_solr_response,
)
from lakegen.retrieval.solr.client import SolrClient


class KeywordRetriever:
    """Keyword-based table retrieval via Solr/Lucene BM25.

    LakeGen deliberately delegates BM25 scoring to Solr/Lucene and preserves
    the baseline eDisMax and query-operator settings. Phase 1 keywords remain
    the query. Field weighting can still be configured in Solr without a local
    reimplementation of the BM25 formula.

    Reference: Robertson and Zaragoza, 2009, DOI: 10.1561/1500000019.

    Args:
        solr: The lake's Solr client.
        query_fields: Solr's ``qf``; omitted from the query when None, which
            keeps the Solr baseline.
        or_fallback: Retry a zero-hit AND query as OR.

    Attributes:
        solr: The lake's Solr client.
        query_fields: Solr's ``qf``, or None.
        or_fallback: Whether a zero-hit AND query is retried as OR.
    """

    def __init__(
        self,
        solr: SolrClient,
        *,
        query_fields: str | None = None,
        or_fallback: bool = False,
    ) -> None:
        self.solr = solr
        self.query_fields = query_fields
        # Off by default: preserves the pre-existing strict-AND baseline.
        # When enabled, a zero-hit AND query is retried as OR instead of
        # returning nothing (see retrieve()).
        self.or_fallback = or_fallback

    def retrieve(
        self,
        keywords: Sequence[str],
        *,
        top_k: int,
        q_op: str = "AND",
    ) -> list[RetrievalHit]:
        """The ``top_k`` tables BM25 ranks highest for the keywords.

        Args:
            keywords: The keywords; blank ones are dropped.
            top_k: How many tables to return.
            q_op: The query operator, ``AND`` or ``OR``.

        Returns:
            The hits, with their lexical score and rank filled; none for no keywords.

        Raises:
            TypeError: When ``keywords`` is a string rather than a sequence of them.
        """
        if isinstance(keywords, str):
            raise TypeError("keywords must be a sequence, not a string")
        clean_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        if not clean_keywords:
            return []
        hits = self._search(clean_keywords, top_k=top_k, q_op=q_op)
        if self.or_fallback and not hits and q_op == "AND":
            # q.op=AND requires every tokenized word across every keyword to
            # appear in one document. An entity that lives only in one field
            # (e.g. publisher) alongside a topic word that appears nowhere in
            # the document at all makes that combination unsatisfiable, so
            # the whole query returns zero candidates even though the
            # document is otherwise a strong match. Fall back to a ranked OR
            # pass rather than let retrieval collapse to nothing.
            hits = self._search(clean_keywords, top_k=top_k, q_op="OR")
        return hits

    def _search(self, clean_keywords: list[str], *, top_k: int, q_op: str) -> list[RetrievalHit]:
        params: dict[str, Any] = {
            "q_op": q_op,
            "rows": top_k,
            "fl": "*,score",
        }
        # Omit qf by default so the pre-existing Solr baseline is unchanged.
        if self.query_fields is not None:
            params["qf"] = self.query_fields
        response = self.solr.select(tokens=clean_keywords, **params)
        hits = hits_from_solr_response(response, top_k)
        for hit in hits:
            hit.lexical_score = hit.score
            hit.lexical_rank = hit.rank
        return hits

    def count(self, keywords: Sequence[str]) -> int:
        """How many tables contain every keyword, by the same strict AND query.

        Never retried as OR, whatever ``or_fallback`` says: a count that a
        fallback could inflate would say nothing about which words fail together.

        Args:
            keywords: The keywords; blank ones are dropped.

        Returns:
            Solr's ``numFound``; 0 for no keywords.
        """
        clean_keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        if not clean_keywords:
            return 0
        params: dict[str, Any] = {"q_op": "AND", "rows": 0}
        if self.query_fields is not None:
            params["qf"] = self.query_fields
        response = self.solr.select(tokens=clean_keywords, **params)
        return int(response.get("response", {}).get("numFound", 0))
