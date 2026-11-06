"""Token-usage helpers shared by all LakeGen LLM phases."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

import tiktoken

_TOTAL_KEYS = ("total_tokens", "totalTokens")
_PROMPT_KEYS = ("prompt_tokens", "promptTokens", "input_tokens", "inputTokens")
_COMPLETION_KEYS = (
    "completion_tokens",
    "completionTokens",
    "output_tokens",
    "outputTokens",
)
_NESTED_KEYS = (
    "usage",
    "data",
    "chat_response",
    "chatResponse",
    "inference_response",
    "inferenceResponse",
)


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return 0


def extract_total_tokens(value: Any, *, _depth: int = 0) -> int:
    """Extract a total from OCI, OpenAI-style, or Ollama-style usage data.

    A total field wins; else the prompt and completion counts are added; else
    the usage nested under ``usage``, ``data``, ``chat_response`` and their
    kin is searched. A string is read as JSON; a mapping is read by key,
    anything else by attribute.

    Args:
        value: A response, its raw payload, a usage object or JSON text.
        _depth: The nesting depth reached; the search stops below 8.

    Returns:
        The total tokens; 0 when none are found.
    """
    if value is None or _depth > 8:
        return 0

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0

    is_mapping = isinstance(value, Mapping)
    read = _reader(value)
    for key in _TOTAL_KEYS:
        total = _non_negative_int(read(key))
        if total:
            return total

    prompt = _first_count(read, _PROMPT_KEYS)
    completion = _first_count(read, _COMPLETION_KEYS)
    if is_mapping:
        # Ollama uses prompt_eval_count/eval_count. OCI and OpenAI use the
        # prompt/completion or input/output pairs above.
        prompt = prompt or _non_negative_int(read("prompt_eval_count"))
        completion = completion or _non_negative_int(read("eval_count"))
    if prompt or completion:
        return prompt + completion

    for key in _NESTED_KEYS:
        nested = read(key)
        if nested is not None:
            total = extract_total_tokens(nested, _depth=_depth + 1)
            if total:
                return total
    return 0


def _reader(value: Any) -> Callable[[str], Any]:
    """Read a field of usage data: a mapping's key, else an object's attribute; None when absent."""
    if isinstance(value, Mapping):
        return value.get
    return lambda key: getattr(value, key, None)


def _first_count(read: Callable[[str], Any], keys: tuple[str, ...]) -> int:
    """The first of ``keys`` that has a value, as a count; 0 when none has one."""
    return next((_non_negative_int(read(key)) for key in keys if read(key) is not None), 0)


def reset_llm_token_usage(llm: Any) -> None:
    """Reset the client's token count, when it keeps one."""
    reset = getattr(llm, "reset_token_usage", None)
    if callable(reset):
        reset()


def get_llm_token_usage(llm: Any) -> int:
    """The client's token count since its last reset; 0 when it keeps none."""
    return _non_negative_int(getattr(llm, "token_usage_total", 0))


def estimate_tokens(*values: Any) -> int:
    """Estimate request/response tokens when the provider omits usage metadata.

    OCI does not consistently include usage on agent tool-call responses.  The
    estimate is deliberately centralized so those calls are still accounted
    for instead of silently being logged as zero.

    Args:
        *values: The messages, options and reply text; non-strings are counted as JSON.

    Returns:
        The cl100k_base token count of the values joined by newlines.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    text_parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            text_parts.append(value)
            continue
        try:
            text_parts.append(json.dumps(value, default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            text_parts.append(str(value))
    return len(encoding.encode("\n".join(text_parts)))
