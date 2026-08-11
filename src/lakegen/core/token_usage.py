"""Token-usage helpers shared by all LakeGen LLM phases."""

from __future__ import annotations

import json
from collections.abc import Mapping
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
    """Extract a total from OCI, OpenAI-style, or Ollama-style usage data."""

    if value is None or _depth > 8:
        return 0

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0

    if isinstance(value, Mapping):
        for key in _TOTAL_KEYS:
            total = _non_negative_int(value.get(key))
            if total:
                return total

        # Ollama uses prompt_eval_count/eval_count. OCI and OpenAI use the
        # prompt/completion or input/output pairs above.
        prompt = next(
            (_non_negative_int(value.get(key)) for key in _PROMPT_KEYS if value.get(key) is not None),
            0,
        )
        completion = next(
            (
                _non_negative_int(value.get(key))
                for key in _COMPLETION_KEYS
                if value.get(key) is not None
            ),
            0,
        )
        prompt = prompt or _non_negative_int(value.get("prompt_eval_count"))
        completion = completion or _non_negative_int(value.get("eval_count"))
        if prompt or completion:
            return prompt + completion

        for key in _NESTED_KEYS:
            if key in value:
                total = extract_total_tokens(value[key], _depth=_depth + 1)
                if total:
                    return total
        return 0

    for key in _TOTAL_KEYS:
        total = _non_negative_int(getattr(value, key, None))
        if total:
            return total

    prompt = next(
        (
            _non_negative_int(getattr(value, key, None))
            for key in _PROMPT_KEYS
            if getattr(value, key, None) is not None
        ),
        0,
    )
    completion = next(
        (
            _non_negative_int(getattr(value, key, None))
            for key in _COMPLETION_KEYS
            if getattr(value, key, None) is not None
        ),
        0,
    )
    if prompt or completion:
        return prompt + completion

    for key in _NESTED_KEYS:
        nested = getattr(value, key, None)
        if nested is not None:
            total = extract_total_tokens(nested, _depth=_depth + 1)
            if total:
                return total
    return 0


def reset_llm_token_usage(llm: Any) -> None:
    reset = getattr(llm, "reset_token_usage", None)
    if callable(reset):
        reset()


def get_llm_token_usage(llm: Any) -> int:
    return _non_negative_int(getattr(llm, "token_usage_total", 0))


def estimate_tokens(*values: Any) -> int:
    """Estimate request/response tokens when the provider omits usage metadata.

    OCI does not consistently include usage on agent tool-call responses.  The
    estimate is deliberately centralized so those calls are still accounted
    for instead of silently being logged as zero.
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
