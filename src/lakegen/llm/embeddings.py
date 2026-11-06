"""Configurable multilingual embeddings for Solr vector retrieval."""

from __future__ import annotations

import hashlib
import logging
import math
import time
import unicodedata
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Protocol

from llama_index.embeddings.ollama import OllamaEmbedding


class EmbeddingGenerationError(RuntimeError):
    """Raised when the embedding provider cannot produce a usable vector."""


class EmbeddingModel(Protocol):
    """What retrieval and indexing need of an embedding model.

    Attributes:
        model_name: The model's name, recorded as the vectors' provenance.
    """

    model_name: str

    def encode_query(self, text: str) -> list[float]:
        """The vector of a question."""
        ...

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """The vectors of table representations, in order."""
        ...


DEFAULT_QUERY_MAX_CHARS = 8192
DEFAULT_RETRY_DELAYS = (0.25, 1.0)
KNOWN_MODEL_DIMENSIONS = {"bge-m3": 1024}
QUERY_NON_FINITE_FALLBACK_PREFIX = "Represent this question for dataset retrieval. "
EMBEDDING_HEALTH_CHECK_QUERY = "LakeGen dataset retrieval health check."
logger = logging.getLogger(__name__)


def _normalize_query(text: str, *, max_chars: int = DEFAULT_QUERY_MAX_CHARS) -> str:
    """Normalize transport-hostile text without changing query semantics."""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in normalized
    )
    normalized = " ".join(normalized.split())
    if not normalized:
        raise EmbeddingGenerationError("Semantic retrieval requires a non-blank question")
    if len(normalized) > max_chars:
        raise EmbeddingGenerationError(
            f"Semantic retrieval question exceeds the configured character limit of {max_chars}"
        )
    return normalized


def _validate(
    vector: Sequence[float],
    *,
    expected_dimension: int | None = None,
    min_norm: float = 1e-12,
    max_norm: float = 1e6,
) -> list[float]:
    result = [float(value) for value in vector]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("Embedding model returned an empty or non-finite vector")
    if expected_dimension is not None and len(result) != expected_dimension:
        raise ValueError(
            f"Embedding model returned dimension {len(result)}; expected {expected_dimension}"
        )
    norm = math.sqrt(sum(value * value for value in result))
    if norm < min_norm:
        raise ValueError("Embedding model returned a zero-norm vector")
    if norm > max_norm:
        raise ValueError(f"Embedding model returned an implausible vector norm ({norm:.6g})")
    return result


def _is_non_finite_provider_failure(exc: Exception) -> bool:
    """Identify deterministic provider failures caused by invalid float output."""
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current).casefold())
        current = current.__cause__
    detail = " ".join(messages)
    return any(
        marker in detail
        for marker in (
            "unsupported value: nan",
            "non-finite vector",
            "zero-norm vector",
        )
    )


class OllamaMultilingualEmbedding:
    """Pretrained multilingual embedding model served by Ollama.

    ``bge-m3`` is the default for the optional Ollama backend.
    This is a pretrained shared encoder, not DPR training.

    Attributes:
        model_name: The Ollama model.
        base_url: The Ollama server's URL.
        retry_delays: The pauses between the attempts of a query embedding.
        query_max_chars: The longest question accepted, after normalisation.
        expected_dimension: The vectors' dimension, when known; vectors of another
            dimension are rejected.

    Args:
        model_name: The Ollama model.
        base_url: The Ollama server's URL.
        batch_size: Texts per embedding request.
        request_timeout: The timeout of every request, in seconds.
        retry_delays: The pauses between the attempts of a query embedding.
        query_max_chars: The longest question accepted, after normalisation.
        expected_dimension: The vectors' dimension; the model's known one when None.
        sleep: The pause function (tests replace it).

    Raises:
        ValueError: The timeout or the character limit is not positive, or a delay is
            negative.
    """

    def __init__(
        self,
        model_name: str = "bge-m3",
        base_url: str = "http://localhost:11434",
        batch_size: int = 16,
        request_timeout: float = 30.0,
        retry_delays: Sequence[float] = DEFAULT_RETRY_DELAYS,
        query_max_chars: int = DEFAULT_QUERY_MAX_CHARS,
        expected_dimension: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")
        if query_max_chars <= 0:
            raise ValueError("query_max_chars must be greater than zero")
        if any(delay < 0 for delay in retry_delays):
            raise ValueError("retry delays must not be negative")
        self.model_name = model_name
        self.base_url = base_url
        self.retry_delays = tuple(float(delay) for delay in retry_delays)
        self.query_max_chars = query_max_chars
        self.expected_dimension = (
            expected_dimension
            if expected_dimension is not None
            else KNOWN_MODEL_DIMENSIONS.get(model_name)
        )
        self._sleep = sleep
        self._model = OllamaEmbedding(
            model_name=model_name,
            base_url=base_url,
            embed_batch_size=batch_size,
            client_kwargs={"timeout": request_timeout},
        )

    def encode_query(self, text: str) -> list[float]:
        """The vector of a question, retried, with a prefixed fallback for non-finite output.

        A failure that repeats as NaN or a zero-norm vector is not retried as
        is: the question is embedded once more behind the retrieval prefix,
        which means the same.

        Args:
            text: The question.

        Returns:
            The validated vector.

        Raises:
            EmbeddingGenerationError: Every attempt failed, or the question is blank or
                too long.
        """
        normalized = _normalize_query(
            text,
            max_chars=getattr(self, "query_max_chars", DEFAULT_QUERY_MAX_CHARS),
        )
        retry_delays = getattr(self, "retry_delays", DEFAULT_RETRY_DELAYS)
        expected_dimension = getattr(self, "expected_dimension", None)
        sleeper = getattr(self, "_sleep", time.sleep)
        started = time.monotonic()
        attempt_count = 0
        last_error: Exception | None = None

        def attempt(candidate: str) -> list[float] | None:
            nonlocal attempt_count, last_error
            for attempt_index in range(len(retry_delays) + 1):
                attempt_count += 1
                try:
                    return _validate(
                        self._model.get_query_embedding(candidate),
                        expected_dimension=expected_dimension,
                    )
                except Exception as exc:
                    last_error = exc
                    logger.debug(
                        "query embedding attempt %d failed: %s: %s",
                        attempt_count,
                        type(exc).__name__,
                        exc,
                    )
                    # Repeating identical text cannot repair deterministic NaN
                    # output from the model. Let the caller switch to the
                    # semantically equivalent retrieval-prefixed query.
                    if _is_non_finite_provider_failure(exc):
                        break
                    if attempt_index < len(retry_delays):
                        sleeper(retry_delays[attempt_index])
            return None

        vector = attempt(normalized)
        if vector is not None:
            return vector

        used_fallback = bool(last_error is not None and _is_non_finite_provider_failure(last_error))
        if used_fallback:
            vector = attempt(QUERY_NON_FINITE_FALLBACK_PREFIX + normalized)
            if vector is not None:
                logger.warning(
                    "Recovered non-finite query embedding: model=%s "
                    "attempts=%d duration_seconds=%.6f query_sha256=%s",
                    getattr(self, "model_name", "unknown"),
                    attempt_count,
                    time.monotonic() - started,
                    hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                )
                return vector

        duration = time.monotonic() - started
        logger.error(
            "Query embedding failed: model=%s base_url=%s attempts=%d "
            "duration_seconds=%.6f fallback=%s error_type=%s error=%s",
            getattr(self, "model_name", "unknown"),
            getattr(self, "base_url", "unknown"),
            attempt_count,
            duration,
            used_fallback,
            type(last_error).__name__,
            last_error,
        )
        raise EmbeddingGenerationError(
            "The embedding service failed to produce a valid query vector "
            f"for model {getattr(self, 'model_name', 'unknown')!r} after "
            f"{attempt_count} attempts in {duration:.3f}s "
            f"(non-finite fallback used: {used_fallback}); last error: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """The validated vectors of the texts, in order; [] for no texts."""
        if not texts:
            return []
        return [
            _validate(
                vector,
                expected_dimension=getattr(self, "expected_dimension", None),
            )
            for vector in self._model.get_text_embedding_batch(list(texts))
        ]


class OCIEmbedding:
    """OCI Cohere retrieval embeddings, fixed at 1024 dimensions for Solr.

    Attributes:
        model_name: The OCI model id, or a dedicated endpoint's OCID.
        compartment_id: The OCI compartment the requests are made in.

    Args:
        model_name: The OCI model id, or a dedicated endpoint's OCID.
    """

    def __init__(self, model_name: str = "cohere.embed-v4.0") -> None:
        import oci

        from lakegen.llm.providers.oci import oci_runtime_config as _oci_runtime_config

        config_file, profile, compartment, endpoint = _oci_runtime_config()
        self.model_name = model_name
        self.compartment_id = compartment
        self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=oci.config.from_file(str(config_file), profile),
            service_endpoint=endpoint,
            retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
            timeout=(10, 60),
        )

    def _encode(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        """The validated vectors of the texts, 16 per request; any failure raises EmbeddingGenerationError."""
        from oci.generative_ai_inference import models

        vectors: list[list[float]] = []
        for start in range(0, len(texts), 16):
            batch = list(texts[start : start + 16])
            serving_mode = (
                models.DedicatedServingMode(endpoint_id=self.model_name)
                if self.model_name.startswith("ocid1.generativeaiendpoint")
                else models.OnDemandServingMode(model_id=self.model_name)
            )
            kwargs = {}
            if self.model_name == "cohere.embed-v4.0":
                kwargs["output_dimensions"] = 1024
            request = models.EmbedTextDetails(
                inputs=batch,
                serving_mode=serving_mode,
                compartment_id=self.compartment_id,
                input_type=input_type,
                truncate="END",
                embedding_types=["float"],
                **kwargs,
            )
            try:
                payload = self._client.embed_text(request).data
                result = getattr(payload, "embeddings", None)
                if result is None:
                    result = (getattr(payload, "embeddings_by_type", None) or {}).get("float")
                if result is None or len(result) != len(batch):
                    raise ValueError("OCI returned an unexpected number of embeddings")
                vectors.extend(_validate(v, expected_dimension=1024) for v in result)
            except Exception as exc:
                raise EmbeddingGenerationError(
                    f"OCI embedding generation failed for {self.model_name!r}: {exc}"
                ) from exc
        return vectors

    def encode_query(self, text: str) -> list[float]:
        """The vector of a question, embedded as a search query."""
        return self._encode([_normalize_query(text)], "SEARCH_QUERY")[0]

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """The vectors of the texts, embedded as search documents."""
        return self._encode(texts, "SEARCH_DOCUMENT")


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str,
    base_url: str,
) -> EmbeddingModel:
    """The embedding model of that name (cached): OCI for ``cohere.``/``ocid1.`` ids, else Ollama."""
    # OCI model identifiers select OCI; existing Ollama configurations still work.
    if model_name.startswith(("cohere.", "ocid1.")):
        return OCIEmbedding(model_name=model_name)
    return OllamaMultilingualEmbedding(model_name=model_name, base_url=base_url)


def check_embedding_health(model_name: str, base_url: str) -> dict[str, object]:
    """Verify provider availability and vector validity before a semantic batch."""
    started = time.monotonic()
    model = get_embedding_model(model_name, base_url)
    vector = model.encode_query(EMBEDDING_HEALTH_CHECK_QUERY)
    return {
        "model": model_name,
        "base_url": base_url,
        "dimension": len(vector),
        "duration_seconds": round(time.monotonic() - started, 6),
    }
