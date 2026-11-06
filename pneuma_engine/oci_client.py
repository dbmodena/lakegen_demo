"""OpenAI-shaped OCI Generative AI client for Pneuma 0.0.4.

Pneuma recognizes remote providers with ``isinstance(OpenAI)``. This adapter
keeps that contract while sending chat and embedding requests through the OCI
SDK. It holds no credentials: they are read from the OCI config file and
profile LakeGen uses.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_OCI_LLM_MODEL = "openai.gpt-oss-20b"
DEFAULT_OCI_EMBED_MODEL = "cohere.embed-v4.0"
# Appended once to the last message when a chat answer comes back empty.
EMPTY_CHAT_FALLBACK_SUFFIX = "\n\nReturn the requested brief answer as plain text."
_CHAT_ROLES = ("system", "assistant", "user")


def resolve_oci_runtime(
    *, config_file: str | None = None, profile: str | None = None
) -> tuple[Any, str, str, str, str]:
    """Load the OCI configuration without exposing credential values.

    Args:
        config_file: OCI config file; ``OCI_CONFIG_FILE`` or ``~/.oci/config`` when unset.
        profile: Profile in that file; ``OCI_PROFILE`` or ``DEFAULT`` when unset.

    Returns:
        The validated config, the resolved file path, the profile name, the
        compartment ID and the inference endpoint.

    Raises:
        RuntimeError: If the ``oci`` package is missing or no compartment ID is configured.
    """
    try:
        import oci
    except ImportError as exc:  # pragma: no cover - depends on the engine's environment
        raise RuntimeError("the OCI provider needs the 'oci' package in .venv-pneuma") from exc

    resolved_file = str(
        Path(config_file or os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"))
        .expanduser()
        .resolve()
    )
    resolved_profile = profile or os.environ.get("OCI_PROFILE", "DEFAULT")
    config = oci.config.from_file(resolved_file, resolved_profile)
    oci.config.validate_config(config)
    compartment_id = (
        os.environ.get("OCI_COMPARTMENT_ID") or config.get("oci_compartment_id") or ""
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


def _retry(operation: Callable[[], Any], *, attempts: int, initial_delay: float) -> Any:
    """Run ``operation``, retrying any failure with exponential backoff; the last one raises."""
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts:
                raise
            logger.warning(
                "[oci] request failed (%s); retrying %d/%d in %.1fs",
                type(error).__name__,
                attempt,
                attempts - 1,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return None  # unreachable: the last attempt returns or raises


def _message_text(message: Any) -> str:
    contents = getattr(message, "content", None) or []
    return "".join(getattr(item, "text", "") or "" for item in contents)


@dataclass
class OCISettings:
    """How the adapter reaches OCI, and optionally a separate chat endpoint.

    Attributes:
        llm_model: OCI chat model ID, or a dedicated endpoint OCID.
        embedding_model: OCI embedding model ID, or a dedicated endpoint OCID.
        embedding_dimensions: Output dimensions asked of the embedding model; its default when unset.
        config_file: OCI config file; see :func:`resolve_oci_runtime`.
        profile: Profile in the OCI config file.
        retry_attempts: Attempts per request before a failure is raised.
        retry_initial_delay: Seconds before the first retry; doubled after each one.
        min_llm_output_tokens: Floor on a chat request's output budget.
        empty_response_attempts: Chat requests made before an empty answer is an error.
        chat_base_url: OpenAI-compatible endpoint (such as a local Ollama) serving chat
            instead of OCI. Embeddings stay on OCI, so an index keeps the vectors
            it was built with.
        chat_model: Model name at ``chat_base_url``.
        chat_api_key: Key sent to ``chat_base_url``; Ollama ignores it.
        chat_timeout: Seconds a request to ``chat_base_url`` may take.
    """

    llm_model: str = DEFAULT_OCI_LLM_MODEL
    embedding_model: str = DEFAULT_OCI_EMBED_MODEL
    embedding_dimensions: int | None = None
    config_file: str | None = None
    profile: str | None = None
    retry_attempts: int = 6
    retry_initial_delay: float = 2.0
    min_llm_output_tokens: int = 1024
    empty_response_attempts: int = 3
    chat_base_url: str | None = None
    chat_model: str = "gpt-oss:20b"
    chat_api_key: str = "ollama"
    chat_timeout: float = 600.0


class _ChatCompletions:
    def __init__(self, owner: OCICompatOpenAI) -> None:
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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class _Chat:
    def __init__(self, owner: OCICompatOpenAI) -> None:
        self.completions = _ChatCompletions(owner)


class _Embeddings:
    def __init__(self, owner: OCICompatOpenAI) -> None:
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
    """The minimal OpenAI client surface Pneuma uses, backed by OCI Generative AI.

    Args:
        settings: Models, credentials location, retries and the optional chat endpoint.
        client: OCI inference client to use instead of building one from ``settings``.
        chat_client: OpenAI-compatible client to use instead of one for ``settings.chat_base_url``.

    Raises:
        RuntimeError: If the ``oci`` package is missing.
    """

    def __init__(
        self,
        settings: OCISettings,
        *,
        client: Any | None = None,
        chat_client: Any | None = None,
    ) -> None:
        # Initializing OpenAI keeps Pneuma's remote-provider type checks valid.
        # No request is ever sent to this placeholder endpoint.
        super().__init__(api_key="oci-adapter", base_url="http://127.0.0.1")
        try:
            import oci
            from oci.generative_ai_inference import models
        except ImportError as exc:
            raise RuntimeError("the OCI provider needs the 'oci' package in .venv-pneuma") from exc

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
        self._chat_client = chat_client or (
            OpenAI(
                base_url=settings.chat_base_url.rstrip("/"),
                api_key=settings.chat_api_key,
                timeout=settings.chat_timeout,
            )
            if settings.chat_base_url
            else None
        )
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)

    def _serving_mode(self, model: str) -> Any:
        if model.startswith("ocid1.generativeaiendpoint"):
            return self._oci_models.DedicatedServingMode(endpoint_id=model)
        return self._oci_models.OnDemandServingMode(model_id=model)

    def _chat_model_id(self, model: str | None) -> str:
        """Choose the configured OCI model, unless the caller names an OCI model explicitly.

        Pneuma supplies its OpenAI default name (``gpt-4o-mini``), which must
        never replace the configured model.

        Args:
            model: The model the caller asked for, if any.

        Returns:
            The OCI model ID or endpoint OCID to call.
        """
        if (
            model
            and ("." in model or model.startswith("ocid1."))
            and not model.startswith("gpt-4o")
        ):
            return model
        return self._oci_settings.llm_model

    def _oci_messages(self, messages: Iterable[dict[str, Any]]) -> list[Any]:
        message_classes = {
            "system": self._oci_models.SystemMessage,
            "assistant": self._oci_models.AssistantMessage,
            "user": self._oci_models.UserMessage,
        }
        converted = []
        for message in messages:
            role = str(message.get("role", "user")).casefold()
            message_class = message_classes.get(role, self._oci_models.UserMessage)
            converted.append(
                message_class(
                    content=[self._oci_models.TextContent(text=str(message.get("content", "")))]
                )
            )
        return converted

    def _oci_chat_request(
        self,
        oci_messages: list[Any],
        *,
        temperature: float,
        top_p: float,
        seed: int | None,
        json_schema: dict[str, Any] | None,
        json_schema_name: str,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "messages": oci_messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "is_stream": False,
            "reasoning_effort": "LOW",
        }
        if seed is not None:
            request["seed"] = int(seed)
        if json_schema is not None:
            # Strict schema-constrained decoding: the response is generated as
            # JSON matching the schema rather than requested in prose.
            request["response_format"] = self._oci_models.JsonSchemaResponseFormat(
                json_schema=self._oci_models.ResponseJsonSchema(
                    name=json_schema_name, schema=json_schema, is_strict=True
                )
            )
        return request

    def chat_completion(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        model: str | None,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        json_schema: dict[str, Any] | None = None,
        json_schema_name: str = "response",
    ) -> str:
        """Answer a chat on OCI, or on ``chat_base_url`` when one is configured.

        An empty answer is retried with twice the token budget, and the first
        retry of a plain-text request also asks for a brief plain answer.

        Args:
            messages: OpenAI-style messages with ``role`` and ``content``.
            model: Model the caller asked for; only an explicit OCI model ID replaces the configured one.
            max_tokens: Output budget, raised to ``min_llm_output_tokens`` when lower.
            temperature: Sampling temperature.
            top_p: Nucleus sampling mass.
            seed: Sampling seed, for providers that honour one.
            json_schema: JSON schema the answer must match (strict decoding), if any.
            json_schema_name: Name the schema is sent under.

        Returns:
            The answer's text.

        Raises:
            RuntimeError: If every attempt came back empty.
        """
        if self._chat_client is not None:
            return self._endpoint_chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                json_schema=json_schema,
                json_schema_name=json_schema_name,
            )
        model_id = self._chat_model_id(model)
        oci_messages = self._oci_messages(messages)
        request = self._oci_chat_request(
            oci_messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            json_schema=json_schema,
            json_schema_name=json_schema_name,
        )
        token_budget = max(int(max_tokens), self._oci_settings.min_llm_output_tokens)
        for empty_attempt in range(1, self._oci_settings.empty_response_attempts + 1):
            details = self._oci_models.ChatDetails(
                compartment_id=self._oci_compartment_id,
                serving_mode=self._serving_mode(model_id),
                chat_request=self._oci_models.GenericChatRequest(
                    max_tokens=token_budget, **request
                ),
            )
            response = _retry(
                lambda details=details: self._oci_client.chat(details),
                attempts=self._oci_settings.retry_attempts,
                initial_delay=self._oci_settings.retry_initial_delay,
            )
            choices = getattr(response.data.chat_response, "choices", None) or []
            text = _message_text(choices[0].message) if choices else ""
            if text.strip():
                return text
            if empty_attempt < self._oci_settings.empty_response_attempts:
                # The plain-text nudge would contradict a JSON schema.
                if empty_attempt == 1 and json_schema is None:
                    _nudge_last_oci_message(oci_messages)
                token_budget *= 2
                _log_empty_retry("oci", json_schema, token_budget)
        raise RuntimeError("OCI chat response was empty after retries")

    def _endpoint_chat_completion(
        self,
        messages: Iterable[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
        json_schema: dict[str, Any] | None,
        json_schema_name: str,
    ) -> str:
        """Answer a chat on ``chat_base_url`` under the OCI path's contract.

        Same low reasoning effort, token floor, strict JSON schema and
        empty-answer retries as the OCI path, so narration and the judge behave
        the same whichever serves them. The configured ``chat_model`` is always
        used: Pneuma passes its own OpenAI default name.

        Args:
            messages: OpenAI-style messages with ``role`` and ``content``.
            max_tokens: Output budget, raised to ``min_llm_output_tokens`` when lower.
            temperature: Sampling temperature.
            top_p: Nucleus sampling mass.
            seed: Sampling seed, for providers that honour one.
            json_schema: JSON schema the answer must match (strict decoding), if any.
            json_schema_name: Name the schema is sent under.

        Returns:
            The answer's text.

        Raises:
            RuntimeError: If every attempt came back empty.
        """
        settings = self._oci_settings
        chat_messages = [
            {
                "role": role
                if (role := str(message.get("role", "user")).casefold()) in _CHAT_ROLES
                else "user",
                "content": str(message.get("content", "")),
            }
            for message in messages
        ]
        request: dict[str, Any] = {
            "model": settings.chat_model,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "reasoning_effort": "low",
        }
        if seed is not None:
            request["seed"] = int(seed)
        if json_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": json_schema_name, "schema": json_schema, "strict": True},
            }
        token_budget = max(int(max_tokens), settings.min_llm_output_tokens)
        for empty_attempt in range(1, settings.empty_response_attempts + 1):
            response = _retry(
                lambda budget=token_budget: self._chat_client.chat.completions.create(
                    messages=chat_messages, max_tokens=budget, **request
                ),
                attempts=settings.retry_attempts,
                initial_delay=settings.retry_initial_delay,
            )
            choices = getattr(response, "choices", None) or []
            text = (choices[0].message.content or "") if choices else ""
            if text.strip():
                return text
            if empty_attempt < settings.empty_response_attempts:
                if empty_attempt == 1 and json_schema is None and chat_messages:
                    chat_messages[-1]["content"] += EMPTY_CHAT_FALLBACK_SUFFIX
                token_budget *= 2
                _log_empty_retry("chat", json_schema, token_budget)
        raise RuntimeError("chat response was empty after retries")

    def create_embeddings(
        self,
        inputs: list[str],
        *,
        model: str | None = None,
        input_type: str = "SEARCH_DOCUMENT",
    ) -> list[list[float]]:
        """Embed ``inputs`` on OCI, truncating over-long texts at the end.

        Args:
            inputs: Texts to embed.
            model: Model the caller asked for; only a ``cohere.`` or OCID name replaces the configured one.
            input_type: ``SEARCH_DOCUMENT`` for indexed texts, ``SEARCH_QUERY`` for questions.

        Returns:
            One vector per input, in order.

        Raises:
            RuntimeError: If OCI returns a different number of vectors.
        """
        if not inputs:
            return []
        model_id = self._oci_settings.embedding_model
        if model and (model.startswith("cohere.") or model.startswith("ocid1.")):
            model_id = model
        request: dict[str, Any] = {
            "inputs": inputs,
            "serving_mode": self._serving_mode(model_id),
            "compartment_id": self._oci_compartment_id,
            "truncate": "END",
            "input_type": input_type,
            "embedding_types": ["float"],
        }
        if self._oci_settings.embedding_dimensions is not None:
            request["output_dimensions"] = self._oci_settings.embedding_dimensions
        details = self._oci_models.EmbedTextDetails(**request)
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
            raise RuntimeError("OCI embedding response returned an unexpected number of vectors")
        return [list(map(float, vector)) for vector in vectors]


def _nudge_last_oci_message(oci_messages: list[Any]) -> None:
    """Append the plain-text nudge to the last message that holds text."""
    for message in reversed(oci_messages):
        contents = getattr(message, "content", None) or []
        if contents and hasattr(contents[0], "text"):
            contents[0].text += EMPTY_CHAT_FALLBACK_SUFFIX
            return


def _log_empty_retry(channel: str, json_schema: dict[str, Any] | None, token_budget: int) -> None:
    logger.warning(
        "[%s] empty chat response; retrying with neutral_suffix=%s max_tokens=%d",
        channel,
        str(json_schema is None).lower(),
        token_budget,
    )


def add_oci_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that configure the OCI provider and its optional chat endpoint.

    Args:
        parser: Parser of the bootstrap or the server command.
    """
    parser.add_argument("--oci-config-file")
    parser.add_argument("--oci-profile")
    parser.add_argument("--oci-llm-model", default=DEFAULT_OCI_LLM_MODEL)
    parser.add_argument("--oci-embedding-model", default=DEFAULT_OCI_EMBED_MODEL)
    parser.add_argument("--oci-embedding-dimensions", type=int)
    parser.add_argument(
        "--chat-base-url",
        help="with --provider oci: chat requests (column narration, relevance judge) go to "
        "this OpenAI-compatible endpoint instead, e.g. http://127.0.0.1:11434/v1 for a "
        "local Ollama; embeddings stay on OCI",
    )
    parser.add_argument("--chat-model", default="gpt-oss:20b", help="model name at --chat-base-url")


def client_from_arguments(args: argparse.Namespace) -> OCICompatOpenAI:
    """Build the OCI client from the options :func:`add_oci_arguments` added.

    Args:
        args: Parsed command-line options.

    Returns:
        A client ready to stand in for Pneuma's OpenAI client.
    """
    return OCICompatOpenAI(
        OCISettings(
            llm_model=args.oci_llm_model,
            embedding_model=args.oci_embedding_model,
            embedding_dimensions=args.oci_embedding_dimensions,
            config_file=args.oci_config_file,
            profile=args.oci_profile,
            chat_base_url=args.chat_base_url,
            chat_model=args.chat_model,
        )
    )
