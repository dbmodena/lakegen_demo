"""LLM-assisted adjudication for executable results that differ from the gold."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from llama_index.core.llms import ChatMessage, LLM

from lakegen.core.token_usage import (
    extract_total_tokens,
    get_llm_token_usage,
    reset_llm_token_usage,
)
from prompts.prompt_manager import PromptManager


SEMANTIC_DISPOSITIONS = {
    "alternative_correct",
    "partially_correct",
    "incorrect",
    "indeterminate",
}


def _bounded_json(value: Any, *, max_chars: int = 12_000) -> str:
    if isinstance(value, list) and len(value) > 20:
        value = {
            "total_items": len(value),
            "first_20_items": value[:20],
            "truncated_for_judge": True,
        }
    encoded = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    if len(encoded) <= max_chars:
        return encoded
    return encoded[:max_chars] + "...<truncated>"


def _extract_json(text: str) -> Mapping[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    loaded = json.loads(stripped)
    if not isinstance(loaded, Mapping):
        raise ValueError("semantic judge response must be a JSON object")
    return loaded


def judge_semantic_code_result(
    *,
    question: str,
    expected_description: str,
    reference_result: Any,
    selected_tables: Sequence[str],
    selected_metadata: Mapping[str, Any],
    generated_code: str,
    generated_result: Any,
    llm: LLM,
    prompt_manager: PromptManager,
) -> tuple[dict[str, Any], int]:
    """Return a conservative semantic disposition and judge token usage."""

    reset_llm_token_usage(llm)
    try:
        prompt = prompt_manager.render(
            "code_semantic_judge",
            "prompt",
            question=question,
            expected_description=expected_description,
            reference_result=_bounded_json(reference_result),
            selected_tables=_bounded_json(list(selected_tables)),
            selected_metadata=_bounded_json(selected_metadata),
            generated_code=generated_code[:16_000],
            generated_result=_bounded_json(generated_result),
        )
        response = llm.chat([ChatMessage(role="user", content=prompt)])
        raw_content = str(response.message.content).strip()
        payload = dict(_extract_json(raw_content))
        disposition = str(payload.get("disposition", "")).casefold()
        if disposition not in SEMANTIC_DISPOSITIONS:
            raise ValueError(f"unsupported semantic disposition {disposition!r}")
        confidence = float(payload.get("confidence", 0.0))
        result = {
            "disposition": disposition,
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "rationale": str(payload.get("rationale") or "").strip(),
            "requirements_met": [
                str(item) for item in payload.get("requirements_met", [])
            ],
            "requirements_missing": [
                str(item) for item in payload.get("requirements_missing", [])
            ],
            "judge_error": "",
        }
        tokens = max(
            extract_total_tokens(response.raw),
            extract_total_tokens(response.message.additional_kwargs),
            get_llm_token_usage(llm),
        )
        return result, tokens
    except Exception as exc:
        return {
            "disposition": "indeterminate",
            "confidence": 0.0,
            "rationale": "Semantic adjudication could not be completed.",
            "requirements_met": [],
            "requirements_missing": [],
            "judge_error": f"{type(exc).__name__}: {exc}",
        }, get_llm_token_usage(llm)
