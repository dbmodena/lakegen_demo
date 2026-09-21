"""The planner: turns a question and facts about the selected tables into an
`AnalysisPlan`, before any code is written.

Planning is its own stage with its own author so that the plan is written from
the question and the data, not rationalised after the fact from code that
already exists. The discovery agent that picks the tables does not own the
plan either: it has already argued the tables are right, and that same
argument once talked the amount threshold out of a question ("the tables list
invoices meeting the 500 threshold") on the strength of a dataset description
the data itself contradicted.

Returns what the stage loop needs to decide its next move. A model that
answered but gave an unusable plan (bad JSON, wrong shape) is the planner's
own failure and is retried with that as feedback; a model call that could not
be made at all is infrastructure, and the caller proceeds without a plan
rather than declining the run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llama_index.core.llms import LLM
from pydantic import ValidationError

from lakegen.analysis_plan import AnalysisPlan, plan_to_text
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.judge_json import chat_json_with_repair
from prompts.prompt_manager import PromptManager

_REPAIR_FIELDS = (
    "summary, table_roles, combination, joins, filters, preparation, dimensions, "
    "measures, ordering, limit, output_columns, question_coverage"
)


@dataclass
class PlannerResult:
    plan: AnalysisPlan | None
    error: str = ""
    infrastructure_error: bool = False
    tokens: int = 0


def _shape_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'plan'}: {error['msg']}"
        for error in exc.errors()[:6]
    )


def plan_analysis(
    *,
    question: str,
    data_facts: str,
    previous_plan: AnalysisPlan | dict[str, Any] | None = None,
    feedback: str = "",
    llm: LLM,
    prompt_manager: PromptManager,
) -> PlannerResult:
    reset_llm_token_usage(llm)
    try:
        # Rendering is part of invoking the planner too.  A missing prompt
        # configuration must use the same fail-open path as an unavailable
        # model, rather than taking down the entire coding run before the
        # reviewed pipeline can make that choice.
        prompt = prompt_manager.render(
            "analysis_planner",
            "prompt",
            question=question,
            data_facts=data_facts,
            previous_plan=plan_to_text(previous_plan) if previous_plan else "",
            feedback=feedback,
        )
        payload = chat_json_with_repair(llm, prompt, repair_fields=_REPAIR_FIELDS)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return PlannerResult(
            None, f"the plan was not valid JSON: {exc}", tokens=get_llm_token_usage(llm)
        )
    except Exception as exc:  # noqa: BLE001
        return PlannerResult(
            None, f"planner call failed: {exc}", infrastructure_error=True,
            tokens=get_llm_token_usage(llm),
        )
    try:
        plan = AnalysisPlan.model_validate(payload)
    except ValidationError as exc:
        return PlannerResult(
            None, "the plan did not have the required shape: " + _shape_errors(exc),
            tokens=get_llm_token_usage(llm),
        )
    return PlannerResult(plan, tokens=get_llm_token_usage(llm))
