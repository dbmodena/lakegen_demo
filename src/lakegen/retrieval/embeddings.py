"""Configurable multilingual embeddings for Solr vector retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import math
from typing import Protocol

from llama_index.embeddings.ollama import OllamaEmbedding


class EmbeddingGenerationError(RuntimeError):
    """Raised when the embedding provider cannot produce a usable vector."""


class EmbeddingModel(Protocol):
    model_name: str

    def encode_query(self, text: str) -> list[float]: ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


def _validate(vector: Sequence[float]) -> list[float]:
    result = [float(value) for value in vector]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("Embedding model returned an empty or non-finite vector")
    if math.sqrt(sum(value * value for value in result)) == 0.0:
        raise ValueError("Embedding model returned a zero-norm vector")
    return result


class OllamaMultilingualEmbedding:
    """Pretrained multilingual embedding model served by Ollama.

    ``bge-m3`` is the default and remains fixed across semantic and hybrid
    experiments. This is a pretrained shared encoder, not DPR training.
    """

    def __init__(
        self,
        model_name: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self._model = OllamaEmbedding(
            model_name=model_name,
            base_url=base_url,
            embed_batch_size=batch_size,
        )

    def encode_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingGenerationError(
                "Semantic retrieval requires a non-blank question"
            )
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                return _validate(self._model.get_query_embedding(text))
            except Exception as exc:
                last_error = exc
        raise EmbeddingGenerationError(
            "The embedding service failed to produce a valid query vector "
            "after 3 attempts"
        ) from last_error

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [
            _validate(vector)
            for vector in self._model.get_text_embedding_batch(list(texts))
        ]


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str,
    base_url: str,
) -> OllamaMultilingualEmbedding:
    return OllamaMultilingualEmbedding(model_name=model_name, base_url=base_url)
