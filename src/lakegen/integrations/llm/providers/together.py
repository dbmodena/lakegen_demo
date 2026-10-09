"""Together AI through its OpenAI-compatible endpoint: retries on its rate limits, the reasoning kept, one tool call per turn."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from contextvars import ContextVar
from typing import Any

import llama_index.llms.openai.base as _openai_base
import openai
from llama_index.core.callbacks import CallbackManager

from lakegen.config.registry import ModelSpec
from lakegen.config.settings import Settings, get_settings
from lakegen.integrations.llm.providers import register
from lakegen.integrations.llm.providers.ollama import OllamaLLM
from lakegen.integrations.llm.tool_calls import (
    STREAM_END,
    keep_first_tool_call,
    recover_leaked_tool_call,
)

logger = logging.getLogger(__name__)

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
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def together_wait(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before retrying a Together call, or None when the error is not about load."""
    status = getattr(exc, "status_code", None)
    if status == 402:
        return TOGETHER_CREDIT_CHECK_SECONDS
    if not (isinstance(exc, openai.APIConnectionError) or status in RETRYABLE_STATUS):
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


def reasoning_of(raw: Any, delta: bool = False) -> str:
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


# Set while TogetherLLM builds a request, so the history fixes below
# apply to Together only.
TOGETHER_REQUEST: ContextVar[bool] = ContextVar("lakegen_together_request", default=False)


def with_reasoning(serialize):
    """Send each assistant turn's reasoning back with it, as Together asks for gpt-oss tool calling.

    The OpenAI client drops thinking from the history it sends. A message
    carries reasoning only when TogetherLLM put it there. For Together
    requests, an assistant turn that is only a tool call is also sent with
    content "" instead of null: Together answers null with "400 prompt cannot be
    empty". Other OpenAI-compatible backends send exactly what they did before.
    """
    def wrapper(messages, *args, **kwargs):
        dicts = serialize(messages, *args, **kwargs)
        if isinstance(dicts, list) and len(dicts) == len(messages):
            together = TOGETHER_REQUEST.get()
            for message, message_dict in zip(messages, dicts, strict=True):
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
    _openai_base.to_openai_message_dicts = with_reasoning(_openai_base.to_openai_message_dicts)


class TogetherLLM(OllamaLLM):
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
        parallel = get_settings().together_parallel_tool_calls
        return super()._prepare_chat_with_tools(
            tools, user_msg=user_msg, chat_history=chat_history, verbose=verbose,
            allow_parallel_tool_calls=parallel, **kwargs)

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True, **kwargs):
        if get_settings().together_recover_leaks:
            recover_leaked_tool_call(response)
        keep_first_tool_call(response)
        return super().get_tool_calls_from_response(
            response, error_on_no_tool_call=error_on_no_tool_call, **kwargs)

    def _retrying(self, call):
        started, attempt = time.monotonic(), 0
        while True:
            token = TOGETHER_REQUEST.set(True)
            try:
                return call()
            except Exception as exc:
                wait = together_wait(exc, attempt)
                window = TOGETHER_CREDIT_WAIT_SECONDS if getattr(exc, "status_code", None) == 402 else TOGETHER_RETRY_WINDOW_SECONDS
                if wait is None or time.monotonic() - started + wait > window:
                    raise
                logger.warning("Together %s; retry %d in %.0fs", type(exc).__name__, attempt + 1, wait)
                time.sleep(wait)
                attempt += 1
            finally:
                TOGETHER_REQUEST.reset(token)

    async def _aretrying(self, call):
        started, attempt = time.monotonic(), 0
        while True:
            token = TOGETHER_REQUEST.set(True)
            try:
                return await call()
            except Exception as exc:
                wait = together_wait(exc, attempt)
                window = TOGETHER_CREDIT_WAIT_SECONDS if getattr(exc, "status_code", None) == 402 else TOGETHER_RETRY_WINDOW_SECONDS
                if wait is None or time.monotonic() - started + wait > window:
                    raise
                logger.warning("Together %s; retry %d in %.0fs", type(exc).__name__, attempt + 1, wait)
                await asyncio.sleep(wait)
                attempt += 1
            finally:
                TOGETHER_REQUEST.reset(token)

    @staticmethod
    def _keep_reasoning(response, reasoning: str | None = None):
        """Put Together's reasoning on the reply's message, where the history serializer finds it."""
        if reasoning is None:
            reasoning = reasoning_of(getattr(response, "raw", None))
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
            return stream, next(stream, STREAM_END)

        stream, head = self._retrying(first)

        def generate():
            reasoning = ""
            for chunk in ([head] if head is not STREAM_END else []):
                reasoning += reasoning_of(getattr(chunk, "raw", None), delta=True)
                yield self._keep_reasoning(chunk, reasoning)
            for chunk in stream:
                reasoning += reasoning_of(getattr(chunk, "raw", None), delta=True)
                yield self._keep_reasoning(chunk, reasoning)

        return generate()

    async def astream_chat(self, messages, **kwargs):
        parent = super().astream_chat

        async def first():
            stream = await parent(messages, **kwargs)
            try:
                return stream, await stream.__anext__()
            except StopAsyncIteration:
                return stream, STREAM_END

        stream, head = await self._aretrying(first)

        async def generate():
            reasoning = ""
            if head is not STREAM_END:
                reasoning += reasoning_of(getattr(head, "raw", None), delta=True)
                yield self._keep_reasoning(head, reasoning)
            async for chunk in stream:
                reasoning += reasoning_of(getattr(chunk, "raw", None), delta=True)
                yield self._keep_reasoning(chunk, reasoning)

        return generate()


@register("together")
class TogetherProvider:
    name = "together"

    def build(self, spec: ModelSpec, settings: Settings, callback_manager: CallbackManager) -> TogetherLLM:
        api_key = settings.together_api_key
        if not api_key:
            raise RuntimeError(f"{spec.name} needs a Together AI key: set TOGETHERAI_API_KEY")
        return TogetherLLM(
            model=spec.provider_model,
            api_base=settings.together_base_url,
            api_key=api_key,
            context_window=spec.context_window,
            is_chat_model=True,
            is_function_calling_model=True,
            # LakeGen's 0.1 by default; Together recommends 1.0 for reasoning models.
            temperature=settings.together_temperature,
            max_tokens=spec.max_output_tokens,
            timeout=spec.timeout_seconds,
            max_retries=0,  # TogetherLLM retries as Together asks
            # Streams then end with a usage chunk, which the token accounting reads.
            additional_kwargs={"stream_options": {"include_usage": True}},
            callback_manager=callback_manager,
        )
