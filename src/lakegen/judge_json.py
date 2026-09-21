"""Shared JSON-extraction-with-repair helper for LLM judge modules.

`extract_json` is factored out of `semantic_code_judge._extract_json`
verbatim (that module now re-exports it under its original private name so
its own, already-tested call site and token accounting are untouched) so the
plan judge and code judge (`lakegen.plan_judge`, `lakegen.code_judge`) share
one implementation instead of duplicating it a second and third time.

`chat_json_with_repair` factors the "chat -> parse JSON -> on failure, ask
the LLM to repair its own malformed response once -> parse that instead"
loop that both new judges also duplicated verbatim. `semantic_code_judge`
keeps its own inline loop (it separately accumulates per-call token usage
across both chat calls for a tested accounting behavior); the new judges
don't need that, so this simpler shared version is used there instead.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from llama_index.core.llms import LLM, ChatMessage


def extract_json(text: str) -> Mapping[str, Any]:
    """Parse a JSON object out of `text`, tolerating a markdown code fence
    and leading/trailing prose around the object (scans for the first `{`
    that starts a valid JSON object via `json.JSONDecoder.raw_decode`)."""
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        loaded = None
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, Mapping):
                loaded = candidate
                break
        if loaded is None:
            raise
    if not isinstance(loaded, Mapping):
        raise ValueError("judge response must be a JSON object")
    return loaded


def chat_json_with_repair(
    llm: LLM, prompt: str, *, repair_fields: str, max_chars: int = 4000,
) -> dict[str, Any]:
    """Call the LLM, parse its response as JSON; on parse failure, ask it to
    repair its own malformed output once and parse that instead. Raises the
    parse error from the repair attempt if that also fails -- callers are
    expected to wrap this in their own fail-open `except Exception` handler,
    matching the validated scratchpad judges' behavior of never blocking a
    run on judge infrastructure failure."""
    response = llm.chat([ChatMessage(role="user", content=prompt)])
    raw = str(response.message.content or "").strip()
    try:
        return dict(extract_json(raw))
    except (json.JSONDecodeError, ValueError, TypeError):
        repair_prompt = (
            f"Return only one valid JSON object with fields {repair_fields}. "
            f"Invalid response:\n{raw[:max_chars]}"
        )
        response = llm.chat([ChatMessage(role="user", content=repair_prompt)])
        raw = str(response.message.content or "").strip()
        return dict(extract_json(raw))
