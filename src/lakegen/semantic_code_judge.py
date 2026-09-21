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
_ASSESSMENT_STATUSES = {"verified", "failed", "unknown"}
_FAILURE_BASES = {"observed_result", "necessary_code_contradiction"}
_EVIDENCE_SCOPES = {"source_identity", "row_operation", "computation", "output"}
_EVIDENCE_SOURCES = {
    "generated_code", "code_analysis", "result_comparison", "metadata",
    "gold_comparison",
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
    """Describe result content without exposing its serialization type."""

    if isinstance(value, list):
        truncated = len(value) > sample_size * 2
        items = value[:sample_size]
        if truncated:
            items = [*items, *value[-sample_size:]]
        return {
            "total_items": len(value),
            "preview_is_truncated": truncated,
            "sample_items": items,
        }
    return {
        "preview_is_truncated": False,
        "value": value,
    }


def _comparison_facts(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    useful_keys = {
        "column_precision", "column_recall", "column_f1", "row_precision",
        "row_recall", "row_f1", "cell_accuracy", "item_precision",
        "item_recall", "item_f1", "numeric_absolute_error",
        "numeric_relative_error", "expected_row_count", "actual_row_count",
        "order_required", "order_correct", "column_aliases",
        "requirement_checks", "requirement_pass_rate", "key_columns",
    }
    facts = {key: evaluation[key] for key in useful_keys if key in evaluation}
    checks = facts.get("requirement_checks")
    if isinstance(checks, Mapping):
        facts["requirement_checks"] = {
            key: value for key, value in checks.items() if key != "result_type"
        }
    return facts


def _extract_json(text: str) -> Mapping[str, Any]:
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
        raise ValueError("semantic judge response must be a JSON object")
    return loaded


def _chat_json(
    llm: LLM, prompt: str, stage: str
) -> tuple[dict[str, Any], list[Any]]:
    """Run one structured stage, with a single syntax-repair attempt."""

    response = llm.chat([ChatMessage(role="user", content=prompt)])
    responses = [response]
    raw_content = str(response.message.content).strip()
    try:
        return dict(_extract_json(raw_content)), responses
    except (json.JSONDecodeError, ValueError, TypeError):
        repair_prompt = (
            f"Your {stage} response was not one valid JSON object. Return only "
            "the same intended object as valid JSON, without commentary. Invalid "
            f"response:\n{raw_content[:6000]}"
        )
        response = llm.chat([ChatMessage(role="user", content=repair_prompt)])
        responses.append(response)
        return dict(_extract_json(str(response.message.content).strip())), responses


def _requirements(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("requirements")
    if not isinstance(raw, list) or not raw:
        raise ValueError("requirement extraction returned no requirements")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, 1):
        if not isinstance(item, Mapping):
            raise ValueError("each requirement must be an object")
        identifier = str(item.get("id") or f"R{index}").strip()
        description = str(item.get("description") or "").strip()
        if not identifier or identifier in seen or not description:
            raise ValueError("requirements need unique ids and descriptions")
        seen.add(identifier)
        evidence_scope = str(item.get("evidence_scope") or "").casefold()
        if evidence_scope not in _EVIDENCE_SCOPES:
            evidence_scope = "computation"
        result.append({
            "id": identifier,
            "type": str(item.get("type") or "other").strip(),
            "evidence_scope": evidence_scope,
            "description": description,
            "essential": item.get("essential") is not False,
        })
    return result


def _validated_assessments(
    value: Any, requirement_ids: set[str]
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("requirement_assessments must be a list")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        identifier = str(item.get("requirement_id") or "").strip()
        status = str(item.get("status") or "").casefold()
        source = str(item.get("evidence_source") or "").casefold()
        evidence = str(item.get("evidence") or "").strip()
        failure_basis = str(item.get("failure_basis") or "none").casefold()
        if identifier not in requirement_ids or identifier in seen:
            continue
        if status not in _ASSESSMENT_STATUSES:
            status = "unknown"
        if status == "verified" and (
            source not in _EVIDENCE_SOURCES or not evidence
        ):
            status = "unknown"
        if status == "failed" and (
            failure_basis not in _FAILURE_BASES or not evidence
        ):
            status = "unknown"
        seen.add(identifier)
        result.append({
            "requirement_id": identifier, "status": status,
            "failure_basis": failure_basis,
            "evidence_source": source, "evidence": evidence,
        })
    for identifier in sorted(requirement_ids - seen):
        result.append({
            "requirement_id": identifier, "status": "unknown",
            "failure_basis": "none",
            "evidence_source": "", "evidence": "No assessment was returned.",
        })
    return result


def _complete_alternative_proof(
    value: Any, selected_tables: Sequence[str], selected_metadata: Mapping[str, Any]
) -> bool:
    if not isinstance(value, Mapping) or not all(
        str(value.get(key) or "").strip()
        for key in ("source", "version_or_scope_difference", "causal_explanation")
    ):
        return False
    source = str(value["source"]).casefold()
    known_sources = [*map(str, selected_tables), *map(str, selected_metadata)]
    return any(
        source == known.casefold()
        or source in known.casefold()
        or known.casefold() in source
        for known in known_sources if known.strip()
    )


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
    requirements_override: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], int]:
    """Run a gold-blind, evidence-based, adversarial semantic adjudication."""

    reset_llm_token_usage(llm)
    responses: list[Any] = []
    try:
        if requirements_override is None:
            requirements_payload, stage_responses = _chat_json(
                llm, prompt_manager.render(
                    "code_semantic_judge", "requirements",
                    question=question, expected_description=expected_description,
                    selected_tables=_bounded_json(list(selected_tables)),
                    selected_metadata=_bounded_json(selected_metadata),
                ), "requirement extraction",
            )
            responses.extend(stage_responses)
            requirements = _requirements(requirements_payload)
        else:
            requirements = _requirements({"requirements": list(requirements_override)})
        requirement_ids = {item["id"] for item in requirements}
        requirements_json = _bounded_json(requirements)

        # This stage is deliberately gold-blind to reduce anchoring on the
        # reference value before the model explains what the code computes.
        code_analysis, stage_responses = _chat_json(
            llm, prompt_manager.render(
                "code_semantic_judge", "code_analysis",
                requirements=requirements_json,
                selected_tables=_bounded_json(list(selected_tables)),
                selected_metadata=_bounded_json(selected_metadata),
                generated_code=generated_code[:10_000],
            ), "gold-blind code analysis",
        )
        responses.extend(stage_responses)

        comparison_json = _bounded_json(_comparison_facts(deterministic_evaluation))
        proposed, stage_responses = _chat_json(
            llm, prompt_manager.render(
                "code_semantic_judge", "evidence_judgment",
                requirements=requirements_json,
                code_analysis=_bounded_json(code_analysis),
                reference_result_preview=_bounded_json(_result_preview(reference_result)),
                generated_result_preview=_bounded_json(_result_preview(generated_result)),
                deterministic_comparison=comparison_json,
                selected_metadata=_bounded_json(selected_metadata),
            ), "evidence judgment",
        )
        responses.extend(stage_responses)
        proposed["requirement_assessments"] = _validated_assessments(
            proposed.get("requirement_assessments"), requirement_ids
        )

        critique, stage_responses = _chat_json(
            llm, prompt_manager.render(
                "code_semantic_judge", "critic",
                requirements=requirements_json,
                code_analysis=_bounded_json(code_analysis),
                proposed_judgment=_bounded_json(proposed),
                deterministic_comparison=comparison_json,
            ), "adversarial critique",
        )
        responses.extend(stage_responses)

        requested_disposition = str(
            proposed.get("proposed_disposition") or ""
        ).casefold()
        if requested_disposition not in SEMANTIC_DISPOSITIONS:
            raise ValueError(
                f"unsupported semantic disposition {requested_disposition!r}"
            )
        assessments = list(proposed["requirement_assessments"])
        objections = critique.get("objections", [])
        if not isinstance(objections, list):
            objections = []
        blocking_objections = [
            item for item in objections
            if isinstance(item, Mapping)
            and str(item.get("severity") or "").casefold() in {"critical", "major"}
        ]
        assessment_by_id = {
            item["requirement_id"]: item for item in assessments
        }
        critic_downgrades: list[str] = []
        for objection in blocking_objections:
            identifier = str(objection.get("requirement_id") or "").strip()
            assessment = assessment_by_id.get(identifier)
            if assessment and assessment["status"] in {"verified", "failed"}:
                assessment["status"] = "unknown"
                critic_downgrades.append(identifier)
                assessment["evidence"] = (
                    assessment["evidence"] + " Critic objection: "
                    + str(objection.get("reason") or "unsupported evidence")
                ).strip()
        by_id = {item["requirement_id"]: item for item in assessments}
        essential_ids = {
            item["id"] for item in requirements if item["essential"]
        }
        all_requirements_verified = all(
            by_id[identifier]["status"] == "verified"
            for identifier in essential_ids
        )
        alternative_proof_required = (
            deterministic_evaluation.get("representation_equivalent_match") is not True
        )
        alternative_proof_complete = _complete_alternative_proof(
            proposed.get("alternative_source_proof"), selected_tables, selected_metadata
        )

        disposition = "indeterminate"
        downgrade_reasons: list[str] = []
        failed_essential_scopes = {
            by_id[identifier]["requirement_id"]: next(
                item["evidence_scope"] for item in requirements
                if item["id"] == identifier
            )
            for identifier in essential_ids
            if by_id[identifier]["status"] == "failed"
        }
        verified_essential = any(
            by_id[identifier]["status"] == "verified" for identifier in essential_ids
        )
        central_failure = any(
            scope in {"source_identity", "row_operation", "computation"}
            for scope in failed_essential_scopes.values()
        )
        unknown_essential = any(
            by_id[identifier]["status"] == "unknown" for identifier in essential_ids
        )
        if failed_essential_scopes:
            disposition = (
                "incorrect" if central_failure or not verified_essential
                else "partially_correct"
            )
        elif unknown_essential:
            downgrade_reasons.append("at least one essential requirement is unknown")
        elif blocking_objections:
            downgrade_reasons.append("the critic invalidated positive evidence")
        elif alternative_proof_required and not alternative_proof_complete:
            downgrade_reasons.append("alternative-source proof is incomplete")
        else:
            disposition = "alternative_correct"

        requirements_met = [
            item["requirement_id"] for item in assessments
            if item["status"] == "verified"
        ]
        requirements_missing = [
            item["requirement_id"] for item in assessments
            if item["status"] != "verified"
        ]
        rationale = str(proposed.get("rationale") or "").strip()
        if critic_downgrades:
            downgrade_reasons.append(
                "critic downgraded " + ", ".join(sorted(set(critic_downgrades)))
            )
        if downgrade_reasons:
            rationale = (
                (rationale + " ") if rationale else ""
            ) + "Downgraded: " + "; ".join(downgrade_reasons) + "."
        confidence = (
            len(requirements_met) / len(requirements) if requirements else 0.0
        )
        result = {
            "disposition": disposition,
            "requested_disposition": requested_disposition,
            "confidence": round(confidence, 6),
            "all_requirements_verified": all_requirements_verified,
            "rationale": rationale,
            "requirements_met": requirements_met,
            "requirements_missing": requirements_missing,
            "requirements": requirements,
            "requirement_assessments": assessments,
            "code_analysis": code_analysis,
            "proposed_judgment": proposed,
            "critique": critique,
            "judge_error": "",
        }
        response_tokens = sum(max(
            extract_total_tokens(item.raw),
            extract_total_tokens(item.message.additional_kwargs),
        ) for item in responses)
        return result, max(response_tokens, get_llm_token_usage(llm))
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
