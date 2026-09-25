"""Second-stage reranking of a retriever's top candidates.

A reranker scores each candidate table against the question and the top
``rerank_depth`` hits are reordered by that score; hits beyond the depth keep
their retrieval order, and none is added or dropped. The production default
(config.DEFAULT_RERANK_MODEL) is Cohere ``rerank-v4.0-fast`` on OCI: over the top 20 it cost about 0.1 s per search
and, on 100-question UK and NYC benchmarks, raised Recall@10 and Hit@1 for
semantic, keyword, and the hybrid (keyword/duckdb + semantic) retrievers on UK
(experiments/rerank_lab). Deeper pools (50-500) were no better: the reranker
ignores retrieval order, so tables from deep in the list displaced the right
ones more often than they surfaced them.

Cohere's rerank scores each (question, table) pair independently: the same
table gets the same score whatever else is in the request.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import re
from typing import Any, Protocol

from lakegen.core.catalogue import clean_catalogue_text
from lakegen.retrieval.models import RetrievalHit

# duckdb_agentic appends retrieval bookkeeping to a table's description; it
# describes the search, not the table, so the reranker is not shown it.
_EVIDENCE_NOTE = re.compile(r"\s*DuckDB keyword evidence:.*$", re.S)


class Reranker(Protocol):
    def scores(self, question: str, documents: Sequence[str]) -> list[float]:
        """One relevance score per document, in the documents' order."""


def candidate_text(
    document: dict[str, Any], *, max_description: int = 600, max_columns: int = 40
) -> str:
    """What a reranker is shown for one table: title, publisher, description, columns."""
    title = clean_catalogue_text(document.get("title"))
    publisher = clean_catalogue_text(document.get("publisher") or document.get("owner") or "")
    description = clean_catalogue_text(
        _EVIDENCE_NOTE.sub("", str(document.get("description") or ""))
    )
    if len(description) > max_description:
        description = description[:max_description].rsplit(" ", 1)[0] + " …"
    names = [
        str(column.get("name"))
        for column in document.get("columns") or []
        if isinstance(column, dict) and column.get("name")
    ][:max_columns]
    lines = [f"Title: {title}"]
    if publisher:
        lines.append(f"Publisher: {publisher}")
    if description:
        lines.append(f"Description: {description}")
    if names:
        lines.append("Columns: " + ", ".join(names))
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _oci_client() -> tuple[Any, str]:
    """One OCI inference client per process: a service is built per search."""
    import oci

    from lakegen.core.resources import _oci_runtime_config

    config_file, profile, compartment, endpoint = _oci_runtime_config()
    client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=oci.config.from_file(str(config_file), profile),
        service_endpoint=endpoint,
        retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
    )
    return client, compartment


class OCICohereReranker:
    """Cohere rerank served on demand by OCI Generative AI."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def scores(self, question: str, documents: Sequence[str]) -> list[float]:
        from oci.generative_ai_inference import models

        client, compartment = _oci_client()
        response = client.rerank_text(
            models.RerankTextDetails(
                input=question,
                documents=list(documents),
                compartment_id=compartment,
                serving_mode=models.OnDemandServingMode(model_id=self.model_id),
                top_n=len(documents),
            )
        )
        scores = [float("nan")] * len(documents)
        for rank in response.data.document_ranks:
            scores[rank.index] = float(rank.relevance_score)
        if any(score != score for score in scores):
            raise RuntimeError("the reranker did not score every document")
        return scores


def rerank_hits(
    question: str,
    hits: Sequence[RetrievalHit],
    reranker: Reranker,
    *,
    depth: int,
) -> list[RetrievalHit]:
    """``hits`` with the first ``depth`` reordered by reranker score, ties in retrieval order.

    Each reranked hit keeps its retrieval score and branch ranks, gains
    ``rerank_score``, and ``rank`` becomes its position in the returned list.
    """
    head, tail = list(hits[:depth]), list(hits[depth:])
    if not head or not question.strip():
        return list(hits)
    scores = reranker.scores(question, [candidate_text(hit.document) for hit in head])
    if len(scores) != len(head):
        raise RuntimeError(f"{len(scores)} rerank scores for {len(head)} candidates")
    for hit, score in zip(head, scores):
        hit.rerank_score = score
    order = sorted(range(len(head)), key=lambda index: (-scores[index], index))
    reranked = [head[index] for index in order] + tail
    for rank, hit in enumerate(reranked, 1):
        hit.rank = rank
    return reranked
