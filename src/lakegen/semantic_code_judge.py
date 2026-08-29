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


def _bounded_json(value: Any, *, max_chars: int = 8_000) -> str:
    encoded = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    if len(encoded) <= max_chars:
        return encoded
    return json.dumps({
        "serialized_prefix": encoded[:max_chars],
        "preview_is_truncated": True,
    }, ensure_ascii=False)


def _result_preview(value: Any, *, sample_size: int = 6) -> Mapping[str, Any]:
    """Describe an evaluator-created preview without changing result semantics."""

    if isinstance(value, list):
        truncated = len(value) > sample_size * 2
        items = value[:sample_size]
        if truncated:
            items = [*items, *value[-sample_size:]]
        return {
            "original_result_type": "list",
            "total_items": len(value),
            "preview_is_truncated": truncated,
            "sample_items": items,
        }
    return {
        "original_result_type": type(value).__name__,
        "preview_is_truncated": False,
        "value": value,
    }


def _comparison_facts(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    useful_keys = {
        "expected_result_type", "result_type_match", "exact_result_match",
        "representation_equivalent_match",
        "column_precision", "column_recall", "column_f1", "row_precision",
        "row_recall", "row_f1", "cell_accuracy", "item_precision",
        "item_recall", "item_f1", "numeric_absolute_error",
        "numeric_relative_error", "expected_row_count", "actual_row_count",
        "order_required", "order_correct", "column_aliases",
        "requirement_checks", "requirement_pass_rate", "key_columns",
    }
    return {key: evaluation[key] for key in useful_keys if key in evaluation}


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
    deterministic_evaluation: Mapping[str, Any],
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
            reference_result_preview=_bounded_json(_result_preview(reference_result)),
            selected_tables=_bounded_json(list(selected_tables)),
            selected_metadata=_bounded_json(selected_metadata),
            generated_code=generated_code[:10_000],
            generated_result_preview=_bounded_json(_result_preview(generated_result)),
            deterministic_comparison=_bounded_json(
                _comparison_facts(deterministic_evaluation)
            ),
        )
        response = llm.chat([ChatMessage(role="user", content=prompt)])
        raw_content = str(response.message.content).strip()
        payload = dict(_extract_json(raw_content))
        disposition = str(payload.get("disposition", "")).casefold()
        if disposition not in SEMANTIC_DISPOSITIONS:
            raise ValueError(f"unsupported semantic disposition {disposition!r}")
        confidence = float(payload.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
        all_requirements_verified = payload.get("all_requirements_verified") is True
        requested_disposition = disposition
        if disposition == "alternative_correct" and (
            confidence < 0.9 or not all_requirements_verified
        ):
            disposition = "indeterminate"
        result = {
            "disposition": disposition,
            "requested_disposition": requested_disposition,
            "confidence": round(confidence, 6),
            "all_requirements_verified": all_requirements_verified,
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
            "all_requirements_verified": False,
            "judge_error": f"{type(exc).__name__}: {exc}",
        }, get_llm_token_usage(llm)
