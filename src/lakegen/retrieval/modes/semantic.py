"""Semantic retrieval: the question's embedding against the catalogue's vectors in Solr."""

from __future__ import annotations

from lakegen.config.retrieval import (
    RetrievalConfig,
)
from lakegen.datastructures.retrieval import (
    RetrievalHit,
    hits_from_solr_response,
)
from lakegen.llm.embeddings import EmbeddingModel
from lakegen.retrieval.solr.client import SolrClient


class SemanticRetriever:
    """Dense retriever inspired by DPR's dual-encoder paradigm.

    The natural-language question and the versioned table representation are
    embedded separately with one fixed pretrained multilingual model. Solr KNN
    performs cosine retrieval. This does not reproduce DPR training, paired
    encoders, in-batch negatives, or hard-negative mining.

    References: Karpukhin et al., 2020, DOI: 10.18653/v1/2020.emnlp-main.550;
    Ji et al., TARGET.

    Args:
        solr: The lake's Solr client.
        config: The retrieval configuration (vector field, representation
            version, embedding model).
        embedding_model: Embeds the question.

    Attributes:
        solr: The lake's Solr client.
        config: The retrieval configuration.
        embedding_model: Embeds the question.
    """

    def __init__(
        self,
        solr: SolrClient,
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
        """The ``top_k`` tables nearest to the question, among this representation's vectors.

        Args:
            question: The question, embedded as a query.
            top_k: How many tables to return.

        Returns:
            The hits, with their semantic score and rank filled.
        """
        vector = self.embedding_model.encode_query(question)
        response = self.solr.knn_select(
            vector,
            vector_field=self.config.vector_field,
            top_k=top_k,
            rows=top_k,
            filters=(
                self._quoted_filter("representation_version", self.config.representation_version),
                self._quoted_filter("embedding_model", self.config.embedding_model),
            ),
        )
        hits = hits_from_solr_response(response, top_k)
        for hit in hits:
            hit.semantic_score = hit.score
            hit.semantic_rank = hit.rank
        return hits
