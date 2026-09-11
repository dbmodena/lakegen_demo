"""OpenAI-shaped OCI Generative AI client for Pneuma 0.0.4.

Pneuma currently recognizes remote providers with ``isinstance(OpenAI)``.
This adapter preserves that contract while sending chat and embedding requests
through the OCI SDK.  It intentionally contains no credentials: those are read
from the same OCI config/profile used by LakeGen.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Iterable

from openai import OpenAI


DEFAULT_OCI_LLM_MODEL = "openai.gpt-oss-20b"
DEFAULT_OCI_EMBED_MODEL = "cohere.embed-v4.0"


def resolve_oci_runtime(
    *, config_file: str | None = None, profile: str | None = None
) -> tuple[Any, str, str, str, str]:
    """Load OCI configuration without exposing credential values."""
    try:
        import oci
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError(
            "OCI provider requires the 'oci' package in .venv-pneuma"
        ) from exc

    resolved_file = str(
        Path(config_file or os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"))
        .expanduser()
        .resolve()
    )
    resolved_profile = profile or os.environ.get("OCI_PROFILE", "DEFAULT")
    config = oci.config.from_file(resolved_file, resolved_profile)
    oci.config.validate_config(config)
    compartment_id = (
        os.environ.get("OCI_COMPARTMENT_ID")
        or config.get("oci_compartment_id")
        or ""
    ).strip()
    if not compartment_id:
        raise RuntimeError(
            "OCI compartment ID is missing; set OCI_COMPARTMENT_ID or "
            "oci_compartment_id in the selected OCI profile"
        )
    endpoint = (
        os.environ.get("OCI_SERVICE_ENDPOINT")
        or f"https://inference.generativeai.{config['region']}.oci.oraclecloud.com"
    ).rstrip("/")
    return config, resolved_file, resolved_profile, compartment_id, endpoint


def _retry(operation: Any, *, attempts: int, initial_delay: float) -> Any:
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception:
            if attempt == attempts:
                raise
            print(
                f"[oci] request failed; retrying {attempt}/{attempts - 1} "
                f"in {delay:.1f}s",
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def _message_text(message: Any) -> str:
    contents = getattr(message, "content", None) or []
    return "".join(getattr(item, "text", "") or "" for item in contents)


EMPTY_CHAT_FALLBACK_SUFFIX = (
    "\n\nReturn the requested brief answer as plain text."
)


@dataclass
class OCISettings:
    llm_model: str = DEFAULT_OCI_LLM_MODEL
    embedding_model: str = DEFAULT_OCI_EMBED_MODEL
    embedding_dimensions: int | None = None
    config_file: str | None = None
    profile: str | None = None
    retry_attempts: int = 6
    retry_initial_delay: float = 2.0
    min_llm_output_tokens: int = 1024
    empty_response_attempts: int = 3


class _ChatCompletions:
    def __init__(self, owner: "OCICompatOpenAI") -> None:
        self.owner = owner

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        **_: Any,
    ) -> Any:
        text = self.owner.chat_completion(
            messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
        )


class _Chat:
    def __init__(self, owner: "OCICompatOpenAI") -> None:
        self.completions = _ChatCompletions(owner)


class _Embeddings:
    def __init__(self, owner: "OCICompatOpenAI") -> None:
        self.owner = owner

    def create(
        self,
        *,
        input: str | list[str],  # noqa: A002 - OpenAI compatibility
        model: str | None = None,
        **_: Any,
    ) -> Any:
        items = [input] if isinstance(input, str) else list(input)
        vectors = self.owner.create_embeddings(items, model=model)
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=vector) for i, vector in enumerate(vectors)]
        )


class OCICompatOpenAI(OpenAI):
    """Minimal OpenAI client surface backed by OCI Generative AI."""

    def __init__(self, settings: OCISettings, *, client: Any | None = None) -> None:
        # Initializing OpenAI keeps Pneuma's existing remote-provider type checks
        # valid.  No request is ever sent to this placeholder endpoint.
        super().__init__(api_key="oci-adapter", base_url="http://127.0.0.1")
        try:
            import oci
            from oci.generative_ai_inference import models
        except ImportError as exc:
            raise RuntimeError(
                "OCI provider requires the 'oci' package in .venv-pneuma"
            ) from exc

        config, _, _, compartment_id, endpoint = resolve_oci_runtime(
            config_file=settings.config_file, profile=settings.profile
        )
        self._oci_models = models
        self._oci_client = client or oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint=endpoint,
            retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
        )
        self._oci_settings = settings
        self._oci_compartment_id = compartment_id
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)

    def _serving_mode(self, model: str) -> Any:
        if model.startswith("ocid1.generativeaiendpoint"):
            return self._oci_models.DedicatedServingMode(endpoint_id=model)
        return self._oci_models.OnDemandServingMode(model_id=model)

    def chat_completion(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        model: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> str:
        model_id = self._oci_settings.llm_model
        # Pneuma supplies its OpenAI default model name. Only an explicit OCI
        # identifier is allowed to override the configured OCI model.
        if (
            model
            and ("." in model or model.startswith("ocid1."))
            and not model.startswith("gpt-4o")
        ):
            model_id = model
        message_classes = {
            "system": self._oci_models.SystemMessage,
            "assistant": self._oci_models.AssistantMessage,
            "user": self._oci_models.UserMessage,
        }
        oci_messages = []
        for message in messages:
            role = str(message.get("role", "user")).casefold()
            cls = message_classes.get(role, self._oci_models.UserMessage)
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            oci_messages.append(
                cls(content=[self._oci_models.TextContent(text=content)])
            )
        request_kwargs: dict[str, Any] = {
            "messages": oci_messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "is_stream": False,
            "reasoning_effort": "LOW",
        }
        if seed is not None:
            request_kwargs["seed"] = int(seed)
        token_budget = max(
            int(max_tokens), self._oci_settings.min_llm_output_tokens
        )
        for empty_attempt in range(1, self._oci_settings.empty_response_attempts + 1):
            details = self._oci_models.ChatDetails(
                compartment_id=self._oci_compartment_id,
                serving_mode=self._serving_mode(model_id),
                chat_request=self._oci_models.GenericChatRequest(
                    max_tokens=token_budget, **request_kwargs
                ),
            )
            response = _retry(
                lambda: self._oci_client.chat(details),
                attempts=self._oci_settings.retry_attempts,
                initial_delay=self._oci_settings.retry_initial_delay,
            )
            choices = getattr(response.data.chat_response, "choices", None) or []
            if choices:
                text = _message_text(choices[0].message)
                if text.strip():
                    return text
            if empty_attempt < self._oci_settings.empty_response_attempts:
                if empty_attempt == 1:
                    for message in reversed(oci_messages):
                        contents = getattr(message, "content", None) or []
                        if contents and hasattr(contents[0], "text"):
                            contents[0].text += EMPTY_CHAT_FALLBACK_SUFFIX
                            break
                token_budget *= 2
                print(
                    f"[oci] empty chat response; retrying with "
                    f"neutral_suffix=true max_tokens={token_budget}",
                    flush=True,
                )
        raise RuntimeError("OCI chat response was empty after retries")

    def create_embeddings(
        self,
        inputs: list[str],
        *,
        model: str | None = None,
        input_type: str = "SEARCH_DOCUMENT",
    ) -> list[list[float]]:
        if not inputs:
            return []
        model_id = self._oci_settings.embedding_model
        if model and (model.startswith("cohere.") or model.startswith("ocid1.")):
            model_id = model
        kwargs: dict[str, Any] = {
            "inputs": inputs,
            "serving_mode": self._serving_mode(model_id),
            "compartment_id": self._oci_compartment_id,
            "truncate": "END",
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        if self._oci_settings.embedding_dimensions is not None:
            kwargs["output_dimensions"] = self._oci_settings.embedding_dimensions
        details = self._oci_models.EmbedTextDetails(**kwargs)
        response = _retry(
            lambda: self._oci_client.embed_text(details),
            attempts=self._oci_settings.retry_attempts,
            initial_delay=self._oci_settings.retry_initial_delay,
        )
        vectors = getattr(response.data, "embeddings", None)
        if vectors is None:
            by_type = getattr(response.data, "embeddings_by_type", None) or {}
            vectors = by_type.get("float") if isinstance(by_type, dict) else None
        if vectors is None or len(vectors) != len(inputs):
            raise RuntimeError(
                "OCI embedding response returned an unexpected number of vectors"
            )
        return [list(map(float, vector)) for vector in vectors]
