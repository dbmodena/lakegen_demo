"""Semantic retrieval: the question's embedding against the catalogue's vectors in Solr."""

from __future__ import annotations

from lakegen.config.retrieval import (
    RetrievalConfig,
)
from lakegen.retrieval.embeddings import EmbeddingModel
from lakegen.retrieval.models import (
    RetrievalHit,
    hits_from_solr_response,
)
from lakegen.retrieval.solr_client import LocalSolrClient


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
        hits = hits_from_solr_response(response, top_k)
        for hit in hits:
            hit.semantic_score = hit.score
            hit.semantic_rank = hit.rank
        return hits
