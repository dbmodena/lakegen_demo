import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import random
import re
from threading import Lock
import time
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import uuid

import oci
import openai
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.base.llms.types import TextBlock, ToolCallBlock
from llama_index.core.llms import ChatMessage, LLM, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.llms.oci_genai import OCIGenAI
from llama_index.llms.oci_genai.utils import CHAT_MODELS, PROVIDERS, XAIProvider
from llama_index.llms.openai_like import OpenAILike
import llama_index.llms.openai.base as _openai_base

from lakegen.prompts.prompt_manager import PromptManager
from lakegen.retrieval.solr_client import LocalSolrClient, resolve_solr_core
from lakegen.core import groq_offload
from lakegen.core.table_io import list_table_files
from lakegen.core.token_usage import estimate_tokens, extract_total_tokens
from lakegen.core.config import LOG_DIR
from lakegen.retrieval import RetrievalConfig, RetrievalRunLogger, TableRetrievalService
from lakegen.retrieval.models import RetrievalRun


OCI_DEFAULT_PROFILE = "DEFAULT"
OCI_MAX_OUTPUT_TOKENS = 4000
OPENAI_GPT_OSS_MODEL = "openai.gpt-oss-120b"
OPENAI_GPT_OSS_CONTEXT_SIZE = 128_000
OPENAI_GPT_OSS_MAX_OUTPUT_TOKENS = 16_000
META_LLAMA_MODEL = "meta.llama-3.3-70b-instruct"
GENERIC_CHAT_MODELS = {OPENAI_GPT_OSS_MODEL, META_LLAMA_MODEL}
# Local models served through Ollama's OpenAI-compatible endpoint: LakeGen
# model name -> Ollama tag.
OLLAMA_MODELS = {
    "ollama.gpt-oss-20b": "gpt-oss:20b",
    "ollama.qwen3.8-27b": "qwen3.8:27b",
}
# The local Ollama servers load models with a 32k window, not their native
# 128k (gpt-oss) or 256k (qwen3.8).
OLLAMA_CONTEXT_SIZE = 32_768
OLLAMA_TIMEOUT_SECONDS = 600.0
# Hosted models on Together AI's OpenAI-compatible endpoint (key TOGETHERAI_API_KEY):
# LakeGen model name -> Together model id. Same open weights as OCI's gpt-oss-120b.
TOGETHER_MODELS = {"together.gpt-oss-120b": "openai/gpt-oss-120b"}
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_TIMEOUT_SECONDS = 300.0
# Together sets no fixed limits: it answers 429 (over the account's dynamic
# request or token rate) or 503 (platform at capacity), suggests a wait in
# x-ratelimit-reset and asks for exponential backoff. A call keeps retrying for
# this long before its error reaches the pipeline.
TOGETHER_RETRY_WINDOW_SECONDS = 1800.0
TOGETHER_BACKOFF_MAX_SECONDS = 120.0
# 402 = the account's credit limit is reached: calls wait for credit to be added
# (checked every 5 minutes, for up to a day) instead of failing questions.
TOGETHER_CREDIT_WAIT_SECONDS = 86400.0
TOGETHER_CREDIT_CHECK_SECONDS = 300.0
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

logger = logging.getLogger(__name__)


class _OCIGenericProvider(XAIProvider):
    """Use OCI's generic chat format for GPT-OSS and Meta models.

    LlamaIndex 0.7.0 implements the generic OCI request/response format in its
    XAI provider, but does not register OpenAI and registers Meta without tool
    support. OCI's generic endpoint supports tool calls for both models.
    """

    def chat_generation_info(self, response: Any) -> dict[str, Any]:
        chat_response = response.data.chat_response
        info: dict[str, Any] = {
            "finish_reason": chat_response.choices[0].finish_reason,
        }
        usage_total = extract_total_tokens(getattr(chat_response, "usage", None))
        if usage_total:
            info["usage"] = {"total_tokens": usage_total}
        assistant_message = chat_response.choices[0].message
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        formatted_tool_calls = []
        for tool_call in tool_calls:
            name = getattr(tool_call, "name", None)
            if name is None:
                continue
            arguments = getattr(tool_call, "arguments", None) or getattr(
                tool_call, "parameters", None
            )
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            formatted_tool_calls.append(
                {
                    "toolUseId": getattr(tool_call, "id", None) or uuid.uuid4().hex,
                    "name": name,
                    "input": arguments,
                }
            )
        if formatted_tool_calls:
            info["tool_calls"] = formatted_tool_calls
        return info


_STREAM_END = object()


def _next_stream_item(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END


def _merge_generic_stream_tool_calls(
    accumulated: list[dict[str, Any]],
    event_data: dict[str, Any],
) -> None:
    fragments = event_data.get("message", {}).get("toolCalls", [])
    for index, fragment in enumerate(fragments):
        tool_id = fragment.get("id")
        current = next(
            (
                item
                for item in accumulated
                if tool_id is not None and item["toolUseId"] == tool_id
            ),
            None,
        )
        if current is None and index < len(accumulated):
            current = accumulated[index]
        if current is None:
            current = {
                "toolUseId": tool_id or uuid.uuid4().hex,
                "name": fragment.get("name", ""),
                "input": "",
            }
            accumulated.append(current)

        if tool_id:
            current["toolUseId"] = tool_id
        if fragment.get("name"):
            current["name"] = fragment["name"]
        current["input"] += fragment.get("arguments") or ""


def _finalized_stream_tool_calls(
    accumulated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    finalized = []
    for tool_call in accumulated:
        raw_input = (tool_call.get("input") or "").strip() or "{}"
        try:
            json.loads(raw_input)
        except json.JSONDecodeError:
            continue
        if not tool_call.get("name"):
            continue
        finalized.append({**tool_call, "input": raw_input})
    return finalized


def _parse_tool_call_input(raw_input: Any) -> dict[str, Any] | None:
    if isinstance(raw_input, dict):
        return raw_input
    if raw_input is None or not str(raw_input).strip():
        return {}
    try:
        parsed = json.loads(str(raw_input))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class _LakeGenOCIGenAI(OCIGenAI):
    """Compatibility fixes for OCI tool calls and async LlamaIndex agents."""

    _token_usage_total: int = PrivateAttr(default=0)

    @property
    def token_usage_total(self) -> int:
        return self._token_usage_total

    def reset_token_usage(self) -> None:
        self._token_usage_total = 0

    def _record_token_usage(self, value: Any) -> int:
        total = extract_total_tokens(value)
        if total:
            self._token_usage_total += total
        return total

    @property
    def metadata(self):
        return super().metadata.model_copy(
            update={"is_function_calling_model": True}
        )

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[list[ChatMessage]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = list(chat_history or [])
        if user_msg is not None:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tools,
            **({"tool_choice": "REQUIRED"} if tool_required else {}),
            **kwargs,
        }

    async def achat(self, messages, **kwargs):
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    def get_tool_calls_from_response(
        self,
        response,
        error_on_no_tool_call: bool = True,
        **kwargs,
    ) -> list[ToolSelection]:
        selections = []
        for tool_call in response.message.additional_kwargs.get("tool_calls", []):
            name = tool_call.get("name")
            arguments = _parse_tool_call_input(tool_call.get("input"))
            if not name or arguments is None:
                continue
            selections.append(
                ToolSelection(
                    tool_id=tool_call.get("toolUseId") or uuid.uuid4().hex,
                    tool_name=name,
                    tool_kwargs=arguments,
                )
            )

        if not selections and error_on_no_tool_call:
            raise ValueError("Expected at least one complete tool call, but got none.")
        return selections

    def chat(self, messages, **kwargs):
        # Plain calls may go to Groq's free tier instead (LAKEGEN_GROQ_OFFLOAD).
        offloaded = groq_offload.chat(self.model, messages, kwargs, self.temperature)
        if offloaded is not None:
            self._record_token_usage(offloaded.raw)
            return offloaded
        response = super().chat(messages, **kwargs)
        total = self._record_token_usage(response.raw)
        if not total:
            total = self._record_token_usage(response.message.additional_kwargs)
        if not total:
            self._token_usage_total += estimate_tokens(
                messages, kwargs, getattr(response.message, "content", "")
            )
        return response

    def stream_chat(self, messages, **kwargs):
        # ``think`` is an Ollama-specific request option.  Phase 1 and Phase 3
        # use it to request a separate reasoning stream when the backend
        # supports that feature, but OCI rejects unknown request parameters.
        # OCI may still return reasoning fields on its own, so only remove the
        # unsupported request option and continue processing the response.
        kwargs.pop("think", None)
        stream = super().stream_chat(messages, **kwargs)
        def generate():
            accumulated_tool_calls: list[dict[str, Any]] = []
            recorded_for_request = 0
            response_parts: list[str] = []
            for chunk in stream:
                response_parts.append(str(getattr(chunk, "delta", "") or ""))
                raw = chunk.raw if isinstance(chunk.raw, dict) else {}
                raw_data = raw.get("data", "")
                event_data = {}
                if raw_data:
                    try:
                        event_data = json.loads(raw_data)
                        if self.model in GENERIC_CHAT_MODELS:
                            _merge_generic_stream_tool_calls(
                                accumulated_tool_calls, event_data
                            )
                    except json.JSONDecodeError:
                        pass
                usage_total = extract_total_tokens(event_data)
                if usage_total > recorded_for_request:
                    self._token_usage_total += usage_total - recorded_for_request
                    recorded_for_request = usage_total
                    chunk.message.additional_kwargs["usage"] = {
                        "total_tokens": usage_total
                    }
                if (
                    self.model in GENERIC_CHAT_MODELS
                    and event_data.get("finishReason") == "tool_calls"
                ):
                    finalized = _finalized_stream_tool_calls(
                        accumulated_tool_calls
                    )
                    if finalized:
                        chunk.message.additional_kwargs["tool_calls"] = finalized
                yield chunk
            if recorded_for_request == 0:
                self._token_usage_total += estimate_tokens(
                    messages, kwargs, "".join(response_parts)
                )

        return generate()

    async def acomplete(self, prompt, formatted=False, **kwargs):
        return await asyncio.to_thread(
            self.complete,
            prompt,
            formatted=formatted,
            **kwargs,
        )

    async def astream_chat(self, messages, **kwargs):
        stream = await asyncio.to_thread(self.stream_chat, messages, **kwargs)

        async def generate():
            while True:
                item = await asyncio.to_thread(_next_stream_item, stream)
                if item is _STREAM_END:
                    break
                yield item

        return generate()

    async def astream_complete(self, prompt, formatted=False, **kwargs):
        stream = await asyncio.to_thread(
            self.stream_complete,
            prompt,
            formatted=formatted,
            **kwargs,
        )

        async def generate():
            while True:
                item = await asyncio.to_thread(_next_stream_item, stream)
                if item is _STREAM_END:
                    break
                yield item

        return generate()


class _LakeGenOllamaLLM(OpenAILike):
    """A local Ollama model with the token accounting LakeGen reads from OCI."""

    _token_usage_total: int = PrivateAttr(default=0)

    @property
    def token_usage_total(self) -> int:
        return self._token_usage_total

    def reset_token_usage(self) -> None:
        self._token_usage_total = 0

    def _record_response(self, messages, kwargs, response) -> None:
        total = extract_total_tokens(getattr(response, "raw", None))
        self._token_usage_total += total or estimate_tokens(
            messages, kwargs, getattr(response.message, "content", "") or ""
        )

    def chat(self, messages, **kwargs):
        # See _LakeGenOCIGenAI.stream_chat: the OpenAI client rejects ``think``.
        kwargs.pop("think", None)
        response = super().chat(messages, **kwargs)
        self._record_response(messages, kwargs, response)
        return response

    async def achat(self, messages, **kwargs):
        kwargs.pop("think", None)
        response = await super().achat(messages, **kwargs)
        self._record_response(messages, kwargs, response)
        return response

    def stream_chat(self, messages, **kwargs):
        kwargs.pop("think", None)
        stream = super().stream_chat(messages, **kwargs)

        def generate():
            last = None
            for chunk in stream:
                last = chunk
                yield chunk
            if last is not None:
                self._record_response(messages, kwargs, last)

        return generate()

    async def astream_chat(self, messages, **kwargs):
        kwargs.pop("think", None)
        stream = await super().astream_chat(messages, **kwargs)

        async def generate():
            last = None
            async for chunk in stream:
                last = chunk
                yield chunk
            if last is not None:
                self._record_response(messages, kwargs, last)

        return generate()


def _together_wait(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying a Together call, or None when the error is not about load."""
    status = getattr(exc, "status_code", None)
    if status == 402:
        return TOGETHER_CREDIT_CHECK_SECONDS
    if not (isinstance(exc, openai.APIConnectionError) or status in _RETRYABLE_STATUS):
        return None
    suggested = 0.0
    response = getattr(exc, "response", None)
    for header in ("x-ratelimit-reset", "retry-after"):
        try:
            suggested = float(response.headers.get(header))
            break
        except (AttributeError, TypeError, ValueError):
            continue
    backoff = min(TOGETHER_BACKOFF_MAX_SECONDS, 2.0 * 2 ** attempt) * random.uniform(0.5, 1.0)
    return max(suggested, backoff)


def _reasoning_of(raw: Any, delta: bool = False) -> str:
    """Together's ``reasoning`` field of a completion's message (or a stream chunk's delta)."""
    try:
        choice = raw.choices[0]
    except (AttributeError, IndexError, TypeError):
        return ""
    part = getattr(choice, "delta" if delta else "message", None)
    value = getattr(part, "reasoning", None)
    if value is None:
        value = (getattr(part, "model_extra", None) or {}).get("reasoning")
    return value if isinstance(value, str) else ""


# Set while _LakeGenTogetherLLM builds a request, so the history fixes below
# apply to Together only.
_TOGETHER_REQUEST: ContextVar[bool] = ContextVar("lakegen_together_request", default=False)


def _with_reasoning(serialize):
    """Send each assistant turn's reasoning back with it, as Together asks for gpt-oss tool calling.

    The OpenAI client drops thinking from the history it sends. A message
    carries reasoning only when _LakeGenTogetherLLM put it there. For Together
    requests, an assistant turn that is only a tool call is also sent with
    content "" instead of null: Together answers null with "400 prompt cannot be
    empty". Other OpenAI-compatible backends send exactly what they did before.
    """
    def wrapper(messages, *args, **kwargs):
        dicts = serialize(messages, *args, **kwargs)
        if isinstance(dicts, list) and len(dicts) == len(messages):
            together = _TOGETHER_REQUEST.get()
            for message, message_dict in zip(messages, dicts):
                if message_dict.get("role") != "assistant":
                    continue
                reasoning = getattr(message, "additional_kwargs", {}).get("reasoning")
                if reasoning:
                    message_dict["reasoning"] = reasoning
                if together and message_dict.get("content") is None:
                    message_dict["content"] = ""
        return dicts

    wrapper.__wrapped__ = serialize
    return wrapper


if not hasattr(_openai_base.to_openai_message_dicts, "__wrapped__"):
    _openai_base.to_openai_message_dicts = _with_reasoning(_openai_base.to_openai_message_dicts)


# gpt-oss on Together sometimes returns a whole harmony turn as plain text:
# "analysis<reasoning>assistantcommentary to=functions.<tool> json{<arguments>}".
_LEAKED_TOOL_CALL = re.compile(r"(?:assistant)?commentary to=functions\.([A-Za-z_][\w-]*)[^{]{0,40}?\{", re.S)


def _recover_leaked_tool_call(response) -> bool:
    """Turn a tool call that arrived as text into a real one; True when it did.

    Only a reply with no tool call of its own whose text holds the marker and a
    complete JSON object is changed: the call becomes a ToolCallBlock, the text
    before it is kept as the turn's reasoning and the visible content is emptied.
    """
    message = getattr(response, "message", None)
    if message is None or any(isinstance(block, ToolCallBlock) for block in message.blocks):
        return False
    if message.additional_kwargs.get("tool_calls"):
        return False
    text = message.content or ""
    match = _LEAKED_TOOL_CALL.search(text)
    if not match:
        return False
    try:
        arguments, _ = json.JSONDecoder().raw_decode(text[match.end() - 1:])
    except json.JSONDecodeError:
        return False
    if not isinstance(arguments, dict):
        return False
    before = text[: match.start()].strip()
    before = before[len("analysis"):] if before.startswith("analysis") else before
    if before:
        message.additional_kwargs["reasoning"] = (
            message.additional_kwargs.get("reasoning", "") + before).strip()
    message.blocks = [TextBlock(text=""), ToolCallBlock(
        tool_call_id=f"call_{uuid.uuid4().hex[:24]}", tool_name=match.group(1), tool_kwargs=arguments)]
    message.additional_kwargs["recovered_tool_call"] = match.group(1)
    logger.warning("Together: recovered a %s call that arrived as text", match.group(1))
    return True


def _keep_first_tool_call(response) -> None:
    """Drop all but the first tool call of a response, in both places a message keeps them."""
    message = response.message
    blocks = [block for block in message.blocks if isinstance(block, ToolCallBlock)]
    if len(blocks) > 1:
        message.blocks = [block for block in message.blocks if not isinstance(block, ToolCallBlock)] + blocks[:1]
    calls = message.additional_kwargs.get("tool_calls") or []
    if len(calls) > 1:
        message.additional_kwargs["tool_calls"] = calls[:1]


class _LakeGenTogetherLLM(_LakeGenOllamaLLM):
    """A Together AI model through its OpenAI-compatible endpoint, with the same token accounting.

    One tool call per turn, as LakeGen gets from OCI: its coding tools write one
    script file per run, so two calls executed side by side overwrite each other.
    The request asks for no parallel calls and any extra call in a reply is dropped.

    Rate limits are Together's own: a 429/503/connection error is retried with
    exponential backoff (at least x-ratelimit-reset) for up to
    TOGETHER_RETRY_WINDOW_SECONDS; the client's own retries are off (max_retries=0).
    """

    def _prepare_chat_with_tools(self, tools, user_msg=None, chat_history=None, verbose=False,
                                 allow_parallel_tool_calls=False, **kwargs):
        # Asking for no parallel calls makes Together's gpt-oss write more tool
        # calls as plain text, so by default the request allows them and
        # get_tool_calls_from_response keeps the first.
        parallel = os.environ.get("LAKEGEN_TOGETHER_PARALLEL_TOOL_CALLS", "1").strip() != "0"
        return super()._prepare_chat_with_tools(
            tools, user_msg=user_msg, chat_history=chat_history, verbose=verbose,
            allow_parallel_tool_calls=parallel, **kwargs)

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True, **kwargs):
        if os.environ.get("LAKEGEN_TOGETHER_RECOVER_LEAKS", "1").strip() != "0":
            _recover_leaked_tool_call(response)
        _keep_first_tool_call(response)
        return super().get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call, **kwargs)

    def _retrying(self, call):
        started, attempt = time.monotonic(), 0
        while True:
            token = _TOGETHER_REQUEST.set(True)
            try:
                return call()
            except Exception as exc:
                wait = _together_wait(exc, attempt)
                window = TOGETHER_CREDIT_WAIT_SECONDS if getattr(exc, "status_code", None) == 402 else TOGETHER_RETRY_WINDOW_SECONDS
                if wait is None or time.monotonic() - started + wait > window:
                    raise
                logger.warning("Together %s; retry %d in %.0fs", type(exc).__name__, attempt + 1, wait)
                time.sleep(wait)
                attempt += 1
            finally:
                _TOGETHER_REQUEST.reset(token)

    async def _aretrying(self, call):
        started, attempt = time.monotonic(), 0
        while True:
            token = _TOGETHER_REQUEST.set(True)
            try:
                return await call()
            except Exception as exc:
                wait = _together_wait(exc, attempt)
                window = TOGETHER_CREDIT_WAIT_SECONDS if getattr(exc, "status_code", None) == 402 else TOGETHER_RETRY_WINDOW_SECONDS
                if wait is None or time.monotonic() - started + wait > window:
                    raise
                logger.warning("Together %s; retry %d in %.0fs", type(exc).__name__, attempt + 1, wait)
                await asyncio.sleep(wait)
                attempt += 1
            finally:
                _TOGETHER_REQUEST.reset(token)

    @staticmethod
    def _keep_reasoning(response, reasoning: str | None = None):
        """Put Together's reasoning on the reply's message, where the history serializer finds it."""
        if reasoning is None:
            reasoning = _reasoning_of(getattr(response, "raw", None))
        message = getattr(response, "message", None)
        if reasoning and message is not None:
            message.additional_kwargs["reasoning"] = reasoning
        return response

    def chat(self, messages, **kwargs):
        parent = super().chat
        return self._keep_reasoning(self._retrying(lambda: parent(messages, **kwargs)))

    async def achat(self, messages, **kwargs):
        parent = super().achat
        return self._keep_reasoning(await self._aretrying(lambda: parent(messages, **kwargs)))

    def stream_chat(self, messages, **kwargs):
        parent = super().stream_chat

        def first():  # the request is sent when the first chunk is read
            stream = parent(messages, **kwargs)
            return stream, next(stream, _STREAM_END)

        stream, head = self._retrying(first)

        def generate():
            reasoning = ""
            for chunk in ([head] if head is not _STREAM_END else []):
                reasoning += _reasoning_of(getattr(chunk, "raw", None), delta=True)
                yield self._keep_reasoning(chunk, reasoning)
            for chunk in stream:
                reasoning += _reasoning_of(getattr(chunk, "raw", None), delta=True)
                yield self._keep_reasoning(chunk, reasoning)

        return generate()

    async def astream_chat(self, messages, **kwargs):
        parent = super().astream_chat

        async def first():
            stream = await parent(messages, **kwargs)
            try:
                return stream, await stream.__anext__()
            except StopAsyncIteration:
                return stream, _STREAM_END

        stream, head = await self._aretrying(first)

        async def generate():
            reasoning = ""
            if head is not _STREAM_END:
                reasoning += _reasoning_of(getattr(head, "raw", None), delta=True)
                yield self._keep_reasoning(head, reasoning)
            async for chunk in stream:
                reasoning += _reasoning_of(getattr(chunk, "raw", None), delta=True)
                yield self._keep_reasoning(chunk, reasoning)

        return generate()


# Register the models with OCI's GENERIC native chat format until the
# LlamaIndex integration supports their agentic tool calls directly.
CHAT_MODELS.setdefault(OPENAI_GPT_OSS_MODEL, OPENAI_GPT_OSS_CONTEXT_SIZE)
PROVIDERS["openai"] = _OCIGenericProvider()
PROVIDERS["meta"] = _OCIGenericProvider()


def _oci_runtime_config() -> tuple[Path, str, str, str]:
    config_file = Path(
        os.environ.get("OCI_CONFIG_FILE", "~/.oci/config")
    ).expanduser()
    profile = os.environ.get("OCI_PROFILE", OCI_DEFAULT_PROFILE)

    try:
        config = oci.config.from_file(
            file_location=str(config_file),
            profile_name=profile,
        )
        oci.config.validate_config(config)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid OCI configuration in {config_file} (profile {profile!r}): {exc}"
        ) from exc

    compartment_id = (
        os.environ.get("OCI_COMPARTMENT_ID")
        or config.get("oci_compartment_id")
        or ""
    ).strip()
    if not compartment_id:
        raise RuntimeError(
            "OCI compartment ID is missing. Set OCI_COMPARTMENT_ID or add "
            "oci_compartment_id to the OCI config profile."
        )

    service_endpoint = (
        os.environ.get("OCI_SERVICE_ENDPOINT")
        or f"https://inference.generativeai.{config['region']}.oci.oraclecloud.com"
    )
    return config_file, profile, compartment_id, service_endpoint


def get_llm(model: str) -> tuple[LLM, TokenCountingHandler]:
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])

    if model in OLLAMA_MODELS:
        base_url = os.environ.get("LAKEGEN_OLLAMA_BASE_URL", "http://localhost:11434")
        llm = _LakeGenOllamaLLM(
            model=OLLAMA_MODELS[model],
            api_base=f"{base_url.rstrip('/')}/v1",
            api_key="ollama",
            context_window=OLLAMA_CONTEXT_SIZE,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.1,
            max_tokens=OPENAI_GPT_OSS_MAX_OUTPUT_TOKENS,
            timeout=OLLAMA_TIMEOUT_SECONDS,
            callback_manager=Settings.callback_manager,
        )
        return llm, token_counter

    if model in TOGETHER_MODELS:
        api_key = (os.environ.get("TOGETHERAI_API_KEY") or os.environ.get("TOGETHER_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(f"{model} needs a Together AI key: set TOGETHERAI_API_KEY")
        llm = _LakeGenTogetherLLM(
            model=TOGETHER_MODELS[model],
            api_base=os.environ.get("LAKEGEN_TOGETHER_BASE_URL", TOGETHER_BASE_URL),
            api_key=api_key,
            context_window=OPENAI_GPT_OSS_CONTEXT_SIZE,
            is_chat_model=True,
            is_function_calling_model=True,
            # LakeGen's 0.1 by default; Together recommends 1.0 for reasoning models.
            temperature=float(os.environ.get("LAKEGEN_TOGETHER_TEMPERATURE", "0.1")),
            max_tokens=OPENAI_GPT_OSS_MAX_OUTPUT_TOKENS,
            timeout=TOGETHER_TIMEOUT_SECONDS,
            max_retries=0,  # _LakeGenTogetherLLM retries as Together asks
            # Streams then end with a usage chunk, which the token accounting reads.
            additional_kwargs={"stream_options": {"include_usage": True}},
            callback_manager=Settings.callback_manager,
        )
        return llm, token_counter

    config_file, profile, compartment_id, service_endpoint = _oci_runtime_config()
    max_tokens = (
        OPENAI_GPT_OSS_MAX_OUTPUT_TOKENS
        if model == OPENAI_GPT_OSS_MODEL
        else OCI_MAX_OUTPUT_TOKENS
    )
    model_kwargs = {}
    if model == OPENAI_GPT_OSS_MODEL:
        model_kwargs["context_size"] = OPENAI_GPT_OSS_CONTEXT_SIZE
    llm = _LakeGenOCIGenAI(
        model=model,
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile=profile,
        auth_file_location=str(config_file),
        temperature=0.1,
        max_tokens=max_tokens,
        callback_manager=Settings.callback_manager,
        **model_kwargs,
    )

    return llm, token_counter


@lru_cache(maxsize=8)
def get_solr(core):
    return LocalSolrClient(
        core=resolve_solr_core(core),
        base_url=os.environ.get("SOLR_BASE_URL", "http://localhost:8983/solr"),
    )


@lru_cache(maxsize=1)
def get_retrieval_run_logger() -> RetrievalRunLogger:
    return RetrievalRunLogger(LOG_DIR / "retrieval_rankings.jsonl")


_retrieval_capture: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "lakegen_retrieval_capture", default=None
)
_retrieval_context: ContextVar[dict[str, Any]] = ContextVar(
    "lakegen_retrieval_context", default={}
)
_retrieval_attempt: ContextVar[int] = ContextVar(
    "lakegen_retrieval_attempt", default=0
)


@contextmanager
def capture_retrieval_runs(context: Mapping[str, Any] | None = None):
    """Collect retrieval JSON records produced by the current workflow."""

    captured: list[dict[str, Any]] = []
    token = _retrieval_capture.set(captured)
    context_token = _retrieval_context.set(dict(context or {}))
    attempt_token = _retrieval_attempt.set(0)
    try:
        yield captured
    finally:
        _retrieval_capture.reset(token)
        _retrieval_context.reset(context_token)
        _retrieval_attempt.reset(attempt_token)


def _log_and_capture_retrieval_run(run: RetrievalRun) -> None:
    context = _retrieval_context.get()
    attempt = _retrieval_attempt.get() + 1
    _retrieval_attempt.set(attempt)
    run.job_id = context.get("JOB_ID")
    run.source_path = context.get("SOURCE_PATH")
    run.source_id = context.get("SOURCE_ID")
    run.execution_attempt = context.get("EXECUTION_ATTEMPT")
    run.experiment_id = context.get("EXPERIMENT_ID")
    run.retrieval_attempt = attempt
    get_retrieval_run_logger()(run)
    captured = _retrieval_capture.get()
    if captured is not None:
        payload = run.to_log_dict()
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        captured.append(payload)


def make_retrieval_run_observer(
    context: Mapping[str, Any] | None,
    captured: list[dict[str, Any]],
) -> Callable[[RetrievalRun], None]:
    """Build a thread-safe observer with explicit workflow context.

    Agent tools may run in a worker context where ``ContextVar`` values are not
    propagated.  Binding the context and destination here keeps agentic
    retrieval telemetry attached to the correct job without changing retrieval.
    """

    bound_context = dict(context or {})
    lock = Lock()
    attempt = 0

    def observe(run: RetrievalRun) -> None:
        nonlocal attempt
        with lock:
            attempt += 1
            run.job_id = bound_context.get("JOB_ID")
            run.source_path = bound_context.get("SOURCE_PATH")
            run.source_id = bound_context.get("SOURCE_ID")
            run.execution_attempt = bound_context.get("EXECUTION_ATTEMPT")
            run.experiment_id = bound_context.get("EXPERIMENT_ID")
            run.retrieval_attempt = attempt
            get_retrieval_run_logger()(run)
            payload = run.to_log_dict()
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            captured.append(payload)

    return observe


def log_retrieval_decision(
    *,
    question: str,
    selected_tables: list[str],
    reason: str,
    context: Mapping[str, Any] | None = None,
    mode: str | None = None,
    keywords: Sequence[str] = (),
    retrieval_attempt: int | None = None,
) -> None:
    context = dict(context or {})
    get_retrieval_run_logger().log_payload(
        {
            "event_type": "table_selection",
            "question": question,
            "job_id": context.get("JOB_ID"),
            "source_path": context.get("SOURCE_PATH"),
            "source_id": context.get("SOURCE_ID"),
            "execution_attempt": context.get("EXECUTION_ATTEMPT"),
            "experiment_id": context.get("EXPERIMENT_ID"),
            "mode": mode,
            "keywords": list(keywords),
            "retrieval_attempt": retrieval_attempt,
            "selected_tables": list(selected_tables),
            "decision": "accepted" if selected_tables else "rejected",
            "reason": reason,
        }
    )


def get_table_retrieval_service(
    solr: LocalSolrClient,
    config: RetrievalConfig,
    table_dir: str | Path | None = None,
    *,
    observer: Callable[[RetrievalRun], None] | None = None,
) -> TableRetrievalService:
    return TableRetrievalService(
        solr,
        config,
        observer=observer or _log_and_capture_retrieval_run,
        table_dir=str(table_dir) if table_dir is not None else None,
    )


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    return PromptManager()


def get_all_table_files(table_dir):
    return list_table_files(table_dir)


# Backward-compatible name for callers outside this repository.
get_all_csv_files = get_all_table_files
