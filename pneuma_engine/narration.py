"""Summarizer helpers for Pneuma 0.0.4: concurrent column narration, capped row samples."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


def concurrent_prompt_openai_llm(workers: int) -> Callable[..., list[list[dict[str, str]]]]:
    """Build Pneuma's ``prompt_openai_llm`` with one call's conversations answered concurrently.

    Pneuma narrates a table's columns one request at a time. Each conversation
    here gets the same request, seed and retries, and its answer is appended to
    it in place, so the result is the sequential one; only wall time changes.

    Args:
        workers: Conversations answered at once.

    Returns:
        A drop-in replacement for ``pneuma.summarizer.summarizer.prompt_openai_llm``.
    """
    from openai import OpenAIError

    def prompt(
        llm: Any,
        conversations: list[list[dict[str, str]]],
        model: str = "gpt-4o-mini",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        retry_attempts: int = 5,
    ) -> list[list[dict[str, str]]]:
        def answer(conversation: list[dict[str, str]]) -> None:
            for attempt in range(retry_attempts):
                try:
                    response = llm.chat.completions.create(
                        model=model,
                        messages=conversation,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=42,
                    )
                except OpenAIError as error:
                    if attempt == retry_attempts - 1:
                        raise
                    logger.warning(
                        "OpenAI API error: %s. Retrying (%d/%d)...",
                        error,
                        attempt + 1,
                        retry_attempts,
                    )
                    continue
                conversation.append(
                    {"role": "assistant", "content": response.choices[0].message.content}
                )
                return

        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(conversations)))) as pool:
            list(pool.map(answer, conversations))
        return conversations

    return prompt


def cap_long_cells(frame: Any, limit: int) -> Any:
    """Cut every text cell longer than ``limit`` characters to that length.

    Pneuma writes a row sample as ``column: value | ...`` and embeds it whole.
    A map layer's geometry cell runs to millions of characters of coordinates,
    which buries the other columns, overflows the judge's context and is billed
    per character by the embedding service. Only such cells change, and the
    cut is marked with an ellipsis; row selection does not depend on values.

    Args:
        frame: The pandas DataFrame Pneuma samples rows from; it is not modified.
        limit: Longest text a cell keeps.

    Returns:
        A copy of ``frame`` with the long text cells cut.
    """
    from pandas.api.types import is_object_dtype, is_string_dtype

    capped = frame.copy()
    for column in capped.columns:
        if is_object_dtype(capped[column]) or is_string_dtype(capped[column]):
            capped[column] = capped[column].map(
                lambda value: (
                    value[:limit] + "…" if isinstance(value, str) and len(value) > limit else value
                )
            )
    return capped
