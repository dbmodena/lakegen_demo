"""Post-execution, gold-free coding judge.

Scope: CODE FIDELITY TO THE PLAN, and nothing else. Given the approved
`AnalysisPlan` (the `plan` argument) it decides only whether the code
implements that plan: every filter, preparation, measure, grouping, output and
nothing extra. It is deliberately not shown the question. Whether the plan
answers the question is `lakegen.plan_judge`'s ground and was settled before
the code existed; a code judge that re-argues it, with less evidence than the
plan judge had, is how a correct `Body Name == 'TfGM'` was rejected three
times for "may not match the full entity name". Whether the result is empty or
degenerate is the deterministic validator's, not this judge's.

Without a plan the judge falls back to the older question-based prompt below,
kept for callers that have no planner stage.

Deliberately not a wrapper
around `lakegen.semantic_code_judge.judge_semantic_code_result`, which is
gold-comparison-based by design (needs a `reference_result`) and never fires
for a live production question at all (gated on
`evaluation_disposition == "pending_semantic_review"`, which requires a
benchmark reference -- confirmed by direct exploration of `service.py`).
This judge is gold-free by construction, which is the entire point.

Two votes, mirroring OrQa's `plan_compliance_approval` (including VALUE
GROUNDING sanity of coded literals as a semantic complement to the
deterministic `lakegen.value_grounding` check) and `present_result_approval`.
Structurally mirrors `lakegen.plan_judge` (render -> llm.chat -> JSON-repair
fallback via `lakegen.judge_json`), same schema-grounding fix applied: the
real column list is given directly in the prompt, with an explicit
instruction to only claim a column is missing if it's genuinely absent.

Fail-open by design, same as `lakegen.plan_judge`: a judge INFRASTRUCTURE
error returns both votes as approved rather than blocking the run.

Validated over four rounds of live 100-question scratchpad testing.
"""
from __future__ import annotations

import json
from typing import Any

from llama_index.core.llms import LLM

from lakegen.analysis_plan import plan_to_text
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.judge_json import chat_json_with_repair
from lakegen.value_grounding import GroundingViolation
from prompts.prompt_manager import PromptManager

_REPAIR_FIELDS = (
    "plan_compliance_approval, present_result_approval, violated_criteria, "
    "feedback, suggestions"
)


def judge_production_code_result(
    *,
    question: str = "",
    generated_code: str,
    generated_result: Any = None,
    grounding_violations: list[GroundingViolation] | None = None,
    real_columns: dict[str, list[str]] | None = None,
    plan: Any = None,
    alignment_notes: str = "",
    llm: LLM,
    prompt_manager: PromptManager,
) -> tuple[dict[str, Any], int]:
    reset_llm_token_usage(llm)
    grounding_note = (
        "\n\nDeterministic value-grounding check already found these potential issues "
        "(weigh them, don't just repeat them): "
        + "; ".join(f"{v.column}={v.literal!r} ({v.confidence})" for v in grounding_violations)
        if grounding_violations else ""
    )
    try:
        if plan is not None:
            prompt = prompt_manager.render(
                "code_judge",
                "plan_prompt",
                plan=plan if isinstance(plan, str) else plan_to_text(plan),
                real_columns=json.dumps(real_columns or {}, default=str)[:2000],
                generated_code=generated_code[:6000],
                alignment_notes=(
                    "\nMechanical hints (prompts to look closer, not verdicts): "
                    + alignment_notes if alignment_notes else ""
                ),
            )
        else:
            prompt = prompt_manager.render(
                "code_judge",
                "prompt",
                question=question,
                real_columns=json.dumps(real_columns or {}, default=str)[:2000],
                generated_code=generated_code[:6000],
                generated_result=json.dumps(generated_result, default=str)[:1500],
                grounding_note=grounding_note,
            )
        payload = chat_json_with_repair(llm, prompt, repair_fields=_REPAIR_FIELDS)
    except Exception as exc:  # noqa: BLE001
        return {
            "plan_compliance_approval": True, "present_result_approval": True,  # fail-open
            "violated_criteria": [], "feedback": f"code judge error: {exc}",
            "suggestions": [], "judge_error": str(exc), "approved": True,
        }, get_llm_token_usage(llm)

    result = {
        "plan_compliance_approval": payload.get("plan_compliance_approval") is True,
        # Against a plan this judge does not look at the result: an empty or
        # degenerate one is the deterministic validator's finding, made before
        # this stage runs, so the vote is not asked for and cannot fail here.
        "present_result_approval": (
            True if plan is not None else payload.get("present_result_approval") is True
        ),
        "violated_criteria": [str(x) for x in (payload.get("violated_criteria") or [])],
        "feedback": str(payload.get("feedback") or "").strip(),
        "suggestions": [str(x) for x in (payload.get("suggestions") or [])],
        "judge_error": "",
    }
    result["approved"] = result["plan_compliance_approval"] and result["present_result_approval"]
    return result, get_llm_token_usage(llm)
