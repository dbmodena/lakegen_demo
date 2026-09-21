"""Plan judge, scoped to QUESTION FIDELITY ONLY (the user's explicit
deviation from OrQa's six-dimension panel) -- one LLM call, one vote: does
the plan produce exactly what the question asks.

Given an `AnalysisPlan` (the `plan` argument) it judges that plan against the
question and the data facts, before any code exists, and never sees code: the
question-versus-plan ground is this judge's alone, and `lakegen.code_judge`
holds the plan-versus-code ground. Without a plan it falls back to judging the
generated code and its runtime trace, the older shape kept for callers that
have no planner stage. Structurally mirrors
the proven shape of `lakegen.semantic_code_judge.judge_semantic_code_result`
(render -> llm.chat -> JSON-repair fallback), sharing its JSON extraction via
`lakegen.judge_json`.

Ported from scratchpad validation (four rounds of live 100-question testing)
with two fixes already baked in, not aspirational:
  1. LakeGen has no reliable pre-code structured step plan (`coder_brief` is
     usually an empty "runtime_fallback" -- confirmed by direct smoke test),
     so when `derive_plan_from_operation_trace` produces no steps, the prompt
     stops pretending one was submitted and judges the actual generated code
     + its runtime trace directly instead -- never the misleading
     "(empty plan)" framing that caused ~45% of rejections to be a spurious
     "the plan has no steps" complaint rather than real judgment.
  2. A judge asserting a factual claim about the data ("this column doesn't
     exist") was caught being flatly wrong during validation (a real column
     was right there in the schema), causing a regression. The real column
     list for every involved table is given directly in the prompt, with an
     explicit instruction to only claim a column is missing if it's
     genuinely absent from that list.

Fail-open by design: a judge INFRASTRUCTURE error (LLM call/parse failure)
returns `approves_question_fidelity: True` rather than blocking the run --
that's a different failure mode from the judge actively reviewing and
rejecting, which is the mechanism that should gate the answer.
"""
from __future__ import annotations

import json
from typing import Any

from llama_index.core.llms import LLM

from lakegen.analysis_plan import plan_to_text
from lakegen.column_lineage import PlanStep
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.judge_json import chat_json_with_repair
from prompts.prompt_manager import PromptManager

_REPAIR_FIELDS = (
    "approves_question_fidelity, confidence, rationale, missing_requirements, "
    "unjustified_steps, suggestions"
)


def _format_steps(steps: list[PlanStep]) -> str | None:
    if not steps:
        return None
    lines = []
    for s in steps:
        produces = ", ".join(f"{dc.name}<-{dc.sources}" for dc in s.produces) or "(none)"
        lines.append(
            f"  [{s.order}] {s.op}: tables={s.tables} reads={s.reads} "
            f"produces={produces} -- {s.description}"
        )
    return "\n".join(lines)


def judge_plan_question_fidelity(
    *,
    question: str,
    steps: list[PlanStep] | None = None,
    selected_tables: list[str] | None = None,
    selected_metadata: dict | None = None,
    real_columns: dict[str, list[str]] | None = None,
    generated_code: str = "",
    operation_trace: dict | None = None,
    plan: Any = None,
    data_facts: str = "",
    mechanical_notes: str = "",
    llm: LLM,
    prompt_manager: PromptManager,
) -> tuple[dict[str, Any], int]:
    reset_llm_token_usage(llm)
    steps = steps or []
    selected_tables = selected_tables or []
    selected_metadata = selected_metadata or {}
    formatted_steps = _format_steps(steps)
    if plan is not None:
        plan_section = ""
    elif formatted_steps is not None:
        plan_section = f"Plan steps (in order):\n{formatted_steps}"
    else:
        # No structured pre-code plan exists for this run -- judge fidelity
        # from the ACTUAL generated code and its deterministic runtime trace
        # instead of a misleading "(empty plan)" framing.
        plan_section = (
            "No structured step-by-step plan was produced ahead of coding (this pipeline "
            "doesn't always have one) -- judge fidelity from the ACTUAL generated code and its "
            f"runtime trace below instead:\n\nGenerated code:\n{generated_code[:3000]}\n\n"
            f"Runtime trace (deterministically extracted from the executed code): "
            f"{json.dumps(operation_trace or {}, default=str)[:1500]}"
        )
    try:
        if plan is not None:
            prompt = prompt_manager.render(
                "plan_judge",
                "plan_prompt",
                question=question,
                data_facts=data_facts,
                real_columns=json.dumps(real_columns or {}, default=str)[:2000],
                plan=plan_to_text(plan),
                mechanical_notes=(
                    "\nMechanical hints (prompts to look closer, not verdicts): "
                    + mechanical_notes if mechanical_notes else ""
                ),
            )
        else:
            prompt = prompt_manager.render(
                "plan_judge",
                "prompt",
                question=question,
                selected_tables=json.dumps(selected_tables, default=str),
                selected_metadata=json.dumps(selected_metadata, default=str)[:2000],
                real_columns=json.dumps(real_columns or {}, default=str)[:2000],
                plan_section=plan_section,
            )
        payload = chat_json_with_repair(llm, prompt, repair_fields=_REPAIR_FIELDS)
    except Exception as exc:  # noqa: BLE001
        return {
            "approves_question_fidelity": True,  # fail-open: judge errors don't block a run
            "confidence": 0.0,
            "rationale": f"plan judge error: {exc}",
            "missing_requirements": [],
            "unjustified_steps": [],
            "suggestions": [],
            "judge_error": str(exc),
        }, get_llm_token_usage(llm)

    result = {
        "approves_question_fidelity": payload.get("approves_question_fidelity") is True,
        "confidence": max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))),
        "rationale": str(payload.get("rationale") or "").strip(),
        "missing_requirements": [str(x) for x in (payload.get("missing_requirements") or [])],
        "unjustified_steps": list(payload.get("unjustified_steps") or []),
        "suggestions": [str(x) for x in (payload.get("suggestions") or [])],
        "judge_error": "",
    }
    return result, get_llm_token_usage(llm)
