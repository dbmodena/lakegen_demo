"""The reviewed coder pipeline, OrQa-style: a plan is written and checked
against the question BEFORE any code exists, then the code is checked against
that plan. Each judge has one concern and does not reach into the other's.

  Stage A -- plan, up to `stage_max_retries` tries. A dedicated planner
  (`lakegen.analysis_planner`) writes an `AnalysisPlan` from the question and
  facts about the selected tables (`lakegen.table_facts`). The plan is checked
  deterministically (`validate_analysis_plan`: tables and columns exist,
  filter values are observed in the data, text columns used as numbers are
  converted first) and then by the PLAN JUDGE, which decides one thing: is this
  plan faithful to the QUESTION. It never sees code. A rejected plan goes back
  to the planner with the feedback. Gated on `enable_plan_review`.

  Stage B -- code and validator, up to `stage_max_retries` tries. The coder
  writes and executes code with the approved plan pinned into its prompt. The
  value-grounding / date-parse / empty-zero validator runs on the result; a
  flag regenerates the code, and the plan stays fixed.

  Stage C -- code judge, up to `stage_max_retries` tries. The CODE JUDGE
  decides one thing: does the code implement the approved PLAN. It is not shown
  the question. A failure regenerates the code; the plan stays fixed.

If the code implements the plan and the result is still empty or zero, the
plan is the likeliest culprit (a wrong value, threshold, conversion or
period), so the pipeline goes back to Stage A once with that evidence instead
of retrying code that already does what the plan says. As in OrQa, that
escalation happens once.

Without `enable_plan_review` there is no plan of this pipeline's own, and the
code judge falls back to the selection stage's coder brief as the plan; with
neither there is nothing to judge the code against and that stage is skipped,
said so in the trace rather than silently approved.

Never confidently returns an unapproved answer: if a stage's retry budget is
exhausted without approval, the returned `Phase3Result` has `rejected_reason`
set and no `structured_result`/`raw_result` -- exactly generalizing how
`reject_data`/`reject_tables` already signal "cannot proceed" for a different
reason (data insufficiency) to "the plan/code never passed automated review".
The plan that was written is returned either way (`analysis_plan`), so a caller
that overrides a decline codes against it instead of without one.

Two DIFFERENT things both end up setting `Phase3Result.rejected_reason`, and
callers (`service.py`/`ui/workflow.py`) need to tell them apart: (1) the
underlying coder's OWN reject_tables/reject_data verdict (data/tables
genuinely insufficient -- an existing signal this module passes through
untouched, see `_coder_declined`), which should keep going through today's
table-banning/rediscovery handling; and (2) a review-stage decline from this
module's own stage loops (see `_fail`), which should NOT be treated as a
table problem. The two are disambiguated via `Phase3Result.finalization_mode
== "review_declined"`, set only by `_fail`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from llama_index.core.llms import LLM

from lakegen.analysis_plan import (
    AnalysisPlan,
    alignment_notes,
    plan_to_text,
    question_numbers_missing_from_plan,
    validate_analysis_plan,
)
from lakegen.analysis_planner import plan_analysis
from lakegen.code_judge import judge_production_code_result
from lakegen.core.table_io import read_table
from lakegen.core.types import SolrMetadata
from lakegen.experiment_config import CoderContextLevel
from lakegen.phases.phase3 import Phase3Result, phase3_generate_and_execute
from lakegen.plan_judge import judge_plan_question_fidelity
from lakegen.query_validator import ValidationOutcome, looks_empty_or_zero, validate_query
from lakegen.table_facts import build_table_facts
from lakegen.value_grounding import GroundingViolation
from prompts.prompt_manager import PromptManager

MAX_PLAN_REVISIONS_AFTER_CODE = 1
MAX_FALLBACK_PLAN_CHARS = 6000


def _read_tables(tables: list[str], csv_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for table in tables:
        try:
            frames[table] = read_table(csv_dir / table)
        except Exception:  # noqa: BLE001
            continue
    return frames


def _dedup_violations(violations: list[GroundingViolation]) -> list[GroundingViolation]:
    seen: set[tuple] = set()
    deduped: list[GroundingViolation] = []
    for v in violations:
        key = (v.column, v.operator, repr(v.literal), v.confidence)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


def _validate_across_tables(code: str, frames: dict[str, pd.DataFrame], result_value: Any) -> ValidationOutcome:
    """Compose grounding + validator checks across every table actually
    read -- each individual check already no-ops for a column/table it
    doesn't apply to (`check_value_grounding` skips any comparison whose
    column isn't in the given df), so merging per-table outcomes is safe
    even when more than one table is in play. When no table could be read
    at all, this is a validator infrastructure gap, not a code problem --
    fail open (not flagged), consistent with the judges' own fail-open
    policy on infrastructure failure."""
    if not frames:
        return ValidationOutcome()
    grounding_violations: list[GroundingViolation] = []
    flagged = False
    reasons: list[str] = []
    empty_or_zero = False
    for df in frames.values():
        outcome = validate_query(code, df, result_value)
        grounding_violations.extend(outcome.grounding_violations)
        empty_or_zero = outcome.empty_or_zero_result  # result-derived only, same for every df
        if outcome.flagged:
            flagged = True
            reasons.append(outcome.flag_reason)
    return ValidationOutcome(
        grounding_violations=_dedup_violations(grounding_violations),
        empty_or_zero_result=empty_or_zero,
        flagged=flagged,
        flag_reason=" | ".join(dict.fromkeys(reasons)),
    )


def phase3_generate_and_execute_reviewed(
    query: str,
    tables: list[str],
    candidates: list[str],
    solr_meta: SolrMetadata,
    reasoning: str,
    llm: LLM,
    pm: PromptManager,
    csv_dir: Path,
    *,
    stage_max_retries: int = 3,
    enable_plan_review: bool = True,
    enable_code_review: bool = True,
    on_stage_event: Callable[[dict[str, Any]], None] | None = None,
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
    cancel_check: Callable[[], None] | None = None,
    run_dir: Path | None = None,
    seed: int = 0,
    seed_instruction_recorder: Callable[[], None] | None = None,
    coder_context_level: CoderContextLevel = CoderContextLevel.FULL,
    evaluation_result_type: str | None = None,
    max_run_calls: int = 3,
    selection_plan: dict[str, object] | None = None,
    source_field_names: list[str] | None = None,
    require_semantic_plan: bool = True,
) -> Phase3Result:
    started = time.monotonic()
    frames = _read_tables(tables, csv_dir)
    real_columns = {table: list(df.columns) for table, df in frames.items()}
    trace: list[dict[str, Any]] = []
    error_msg = ""
    previous_code = ""
    result: Phase3Result | None = None
    total_calls = 0
    tokens_spent = 0  # every model call of the pipeline, not only the last coder run
    plan: AnalysisPlan | None = None
    plan_approved = False
    data_facts = build_table_facts(frames, solr_meta) if enable_plan_review and frames else ""

    forwarded_kwargs = dict(
        stream_placeholder=stream_placeholder,
        reasoning_placeholder=reasoning_placeholder,
        stream_reasoning=stream_reasoning,
        cancel_check=cancel_check,
        run_dir=run_dir,
        seed=seed,
        seed_instruction_recorder=seed_instruction_recorder,
        coder_context_level=coder_context_level,
        evaluation_result_type=evaluation_result_type,
        max_run_calls=max_run_calls,
        selection_plan=selection_plan,
        source_field_names=source_field_names,
        require_semantic_plan=require_semantic_plan,
    )

    def _emit(event: dict[str, Any]) -> None:
        trace.append(event)
        if on_stage_event is not None:
            try:
                on_stage_event(event)
            except Exception:  # noqa: BLE001
                pass  # a UI callback failure must never break the pipeline

    def _finalize(base: Phase3Result) -> Phase3Result:
        base.review_trace = trace
        base.analysis_plan = plan.model_dump(exclude_none=True) if plan is not None else None
        base.plan_approved = plan_approved
        base.tokens = tokens_spent
        return base

    def _generate(fresh: bool = False) -> Phase3Result:
        """One coder run against the current plan. `fresh` starts a round from
        the plan alone; otherwise the run is a correction of `previous_code`."""
        nonlocal result, total_calls, tokens_spent
        result = phase3_generate_and_execute(
            query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
            retries=0 if fresh else total_calls,
            error_msg="" if fresh else error_msg,
            previous_code="" if fresh else previous_code,
            approved_plan_text=plan_to_text(plan) if plan is not None else "",
            **forwarded_kwargs,
        )
        total_calls += 1
        tokens_spent += result.tokens
        return result

    def _current_answer():
        assert result is not None
        return result.structured_result if result.structured_result is not None else result.raw_result

    def _answer_is_empty() -> bool:
        """An executed result that is empty or zero. No result at all means the
        coder failed to produce one, which is not evidence about the plan."""
        answer = _current_answer()
        return answer is not None and looks_empty_or_zero(answer)

    def _execution_failure() -> str:
        """Return an execution/preflight failure that needs code repair.

        A failed `run_analysis` returns a normal `Phase3Result`, rather than
        raising from `_generate`.  Do not let that result drift into the
        validator or code judge: neither can repair an unexecuted program and
        the latter used to spend its full review budget on it.
        """
        assert result is not None
        if _current_answer() is not None:
            return ""
        if result.error:
            return str(result.error)
        details = result.execution_error or {}
        if isinstance(details, dict):
            message = details.get("message") or details.get("repair_hint")
            if message:
                return str(message)
        return ""

    def _coder_declined() -> Phase3Result | None:
        """If the underlying coder itself already declined (its own
        reject_tables/reject_data verdict, unrelated to review -- e.g. the
        selected tables are genuinely insufficient), short-circuit
        immediately and return that Phase3Result AS-IS. There is no code
        left to review, and critically this preserves the original
        rejected_reason/rejection_details/rejection_keep_tables/
        rejection_skip_tables shape untouched, unmarked by
        `finalization_mode` below, so the EXISTING table-rejection handling
        in service.py/workflow.py keeps working exactly as it does today for
        a plain (non-reviewed) run -- only `_fail` below (a review-stage
        decline, a materially different situation) is tagged distinctly."""
        assert result is not None
        if result.rejected_reason:
            return _finalize(result)
        return None

    def _fail(reason: str) -> Phase3Result:
        base = result or Phase3Result(code_raw="", tokens=0)
        base.rejected_reason = reason
        base.structured_result = None
        base.raw_result = None
        # Distinguishes a review-stage decline from the coder's own
        # reject_tables/reject_data verdict (see `_coder_declined` above) --
        # both populate `rejected_reason`, but only this one should be kept
        # OUT of the existing table-banning/rediscovery machinery, which
        # assumes any `rejected_reason` is about data/table insufficiency.
        base.finalization_mode = "review_declined"
        _emit({
            "stage": "summary", "outcome": "declined", "reason": reason,
            "generation_calls": total_calls, "wall_time": round(time.monotonic() - started, 1),
        })
        return _finalize(base)

    def _plan_stage(feedback: str) -> str:
        """Write the plan and get it approved. Returns "approved", "declined"
        (the budget ran out; `plan` is the last plan written) or "unplanned"
        (the planner could not be reached, so review continues without one)."""
        nonlocal plan, plan_approved, tokens_spent
        plan_approved = False
        for attempt in range(1, stage_max_retries + 1):
            planned = plan_analysis(
                question=query, data_facts=data_facts, previous_plan=plan,
                feedback=feedback, llm=llm, prompt_manager=pm,
            )
            tokens_spent += planned.tokens
            if planned.infrastructure_error:
                plan = None
                _emit({
                    "stage": "plan", "attempt": attempt, "max_attempts": stage_max_retries,
                    "note": f"The planner could not be reached ({planned.error}); "
                            "continuing without a plan.",
                })
                return "unplanned"
            if planned.plan is None:
                feedback = "Your last answer could not be used: " + planned.error
                _emit({
                    "stage": "plan", "attempt": attempt, "approved": False,
                    "max_attempts": stage_max_retries, "plan": None,
                    "planner_error": planned.error,
                })
                continue
            plan = planned.plan
            validation = validate_analysis_plan(plan, frames)
            missing_numbers = question_numbers_missing_from_plan(query, plan)
            judgment, judge_tokens = judge_plan_question_fidelity(
                question=query, plan=plan, data_facts=data_facts,
                mechanical_notes=(
                    "the question states " + ", ".join(missing_numbers)
                    + " but no plan element uses it" if missing_numbers else ""
                ),
                real_columns=real_columns, llm=llm, prompt_manager=pm,
            )
            tokens_spent += judge_tokens
            plan_approved = validation.ok and judgment["approves_question_fidelity"]
            _emit({
                "stage": "plan", "attempt": attempt, "approved": plan_approved,
                "max_attempts": stage_max_retries,
                "plan": plan.model_dump(exclude_none=True),
                "lineage_violations": validation.violations,
                "plan_diagnostics": validation.diagnostics,
                "plan_judgment": judgment,
            })
            if plan_approved:
                return "approved"
            reasons = []
            if not validation.ok:
                reasons.append("Deterministic plan check: " + validation.feedback())
            if not judgment["approves_question_fidelity"]:
                reasons.append(
                    "Plan judge (question fidelity): " + judgment["rationale"]
                    + " Missing: " + str(judgment["missing_requirements"])
                    + (" Unjustified: " + str(judgment["unjustified_steps"])
                       if judgment.get("unjustified_steps") else "")
                    + (" Suggestions: " + str(judgment["suggestions"])
                       if judgment.get("suggestions") else "")
                )
            feedback = " | ".join(reasons)
        return "declined"

    def _judge_plan() -> AnalysisPlan | str | None:
        """What the code judge holds the code to: this pipeline's approved
        plan, else the selection stage's coder brief, else nothing."""
        if plan is not None:
            return plan
        brief = (selection_plan or {}).get("coder_brief") or (selection_plan or {}).get("semantic_plan")
        if brief:
            return json.dumps(brief, ensure_ascii=False, default=str)[:MAX_FALLBACK_PLAN_CHARS]
        return None

    plan_feedback = ""
    plan_revisions = 0
    while True:
        # === Stage A: plan ===
        if enable_plan_review and not frames:
            _emit({
                "stage": "plan",
                "note": "No selected table could be read, so no plan could be checked against the data.",
            })
        elif enable_plan_review:
            outcome = _plan_stage(plan_feedback)
            if outcome == "declined":
                return _fail(
                    f"Plan stage never approved after {stage_max_retries} tries: "
                    f"{_last_plan_feedback(trace)}"
                )

        # === Stage B: code, then the validator loop (the plan stays fixed) ===
        try:
            _generate(fresh=True)
        except Exception as exc:  # noqa: BLE001
            _emit({"stage": "validator", "event": f"generation_error: {exc}"})
            return _fail(f"Generation failed: {exc}")
        declined = _coder_declined()
        if declined is not None:
            return declined

        validator_ok = False
        validation: ValidationOutcome | None = None
        for stage_attempt in range(stage_max_retries):
            code = result.clean_code or ""
            execution_failure = _execution_failure()
            if execution_failure:
                _emit({
                    "stage": "validator", "attempt": stage_attempt + 1, "approved": False,
                    "grounding_violations": [], "validator_flagged": True,
                    "validator_reason": "Execution/preflight failed: " + execution_failure,
                    "execution_failure": execution_failure,
                    "max_attempts": stage_max_retries,
                })
                if stage_attempt < stage_max_retries - 1:
                    error_msg = "Execution/preflight error: " + execution_failure
                    previous_code = code or result.code_raw or ""
                    try:
                        _generate()
                    except Exception as exc:  # noqa: BLE001
                        _emit({"stage": "validator", "event": f"generation_error: {exc}"})
                        return _fail(f"Generation failed during execution retries: {exc}")
                    declined = _coder_declined()
                    if declined is not None:
                        return declined
                    continue
                return _fail(
                    f"Execution never succeeded after {stage_max_retries} tries: {execution_failure}"
                )
            current_answer = _current_answer()
            validation = _validate_across_tables(code, frames, current_answer)
            validator_ok = not validation.flagged
            _emit({
                "stage": "validator", "attempt": stage_attempt + 1, "approved": validator_ok,
                "grounding_violations": [
                    {"column": v.column, "literal": v.literal, "confidence": v.confidence, "message": v.message}
                    for v in validation.grounding_violations
                ],
                "validator_flagged": validation.flagged, "validator_reason": validation.flag_reason,
                "max_attempts": stage_max_retries,
            })
            if validator_ok:
                break
            if stage_attempt < stage_max_retries - 1:
                error_msg = "Query validator: " + validation.flag_reason
                previous_code = code
                try:
                    _generate()
                except Exception as exc:  # noqa: BLE001
                    _emit({"stage": "validator", "event": f"generation_error: {exc}"})
                    return _fail(f"Generation failed during validator retries: {exc}")
                declined = _coder_declined()
                if declined is not None:
                    return declined
        if not validator_ok:
            return _fail(
                f"Validator never cleared after {stage_max_retries} tries: {validation.flag_reason}"
            )

        # === Stage C: the code judge holds the code to the plan ===
        code_plan = _judge_plan() if enable_code_review else None
        if enable_code_review and code_plan is None:
            _emit({
                "stage": "code_judge",
                "note": "Skipped: there is no plan to judge the code against.",
            })
        elif enable_code_review:
            code_ok = False
            code_judgment: dict[str, Any] = {}
            for stage_attempt in range(stage_max_retries):
                code = result.clean_code or ""
                code_judgment, judge_tokens = judge_production_code_result(
                    generated_code=code, plan=code_plan,
                    alignment_notes=alignment_notes(plan, code) if plan is not None else "",
                    real_columns=real_columns, llm=llm, prompt_manager=pm,
                )
                tokens_spent += judge_tokens
                code_ok = code_judgment["approved"]
                _emit({
                    "stage": "code_judge", "attempt": stage_attempt + 1, "approved": code_ok,
                    "code_judgment": code_judgment, "max_attempts": stage_max_retries,
                    "judged_against": "approved plan" if plan is not None else "selection brief",
                })
                if code_ok:
                    break
                if stage_attempt < stage_max_retries - 1:
                    error_msg = "Code judge: " + code_judgment["feedback"]
                    previous_code = code
                    try:
                        _generate()
                    except Exception as exc:  # noqa: BLE001
                        _emit({"stage": "code_judge", "event": f"generation_error: {exc}"})
                        return _fail(f"Generation failed during code-judge retries: {exc}")
                    declined = _coder_declined()
                    if declined is not None:
                        return declined
            if not code_ok:
                return _fail(
                    f"Code judge never approved after {stage_max_retries} tries: "
                    f"{code_judgment.get('feedback', '')}"
                )

        # === Compliant code, empty result: the plan is the suspect ===
        if (
            enable_plan_review and plan is not None
            and plan_revisions < MAX_PLAN_REVISIONS_AFTER_CODE
            and _answer_is_empty()
        ):
            plan_revisions += 1
            plan_feedback = (
                "Code that implements this plan produced an empty or zero result "
                f"({_current_answer()!r}). Something in the plan is probably wrong for this "
                "data: a filter value or threshold, a conversion, a period, or a table. Check "
                "each against the data facts and revise the plan."
            )
            _emit({
                "stage": "plan",
                "note": "The code implements the plan but the result is empty or zero; "
                        "revising the plan once.",
            })
            error_msg = previous_code = ""
            continue
        break

    _emit({
        "stage": "summary", "outcome": "validated", "reason": "",
        "generation_calls": total_calls, "wall_time": round(time.monotonic() - started, 1),
        "plan_approved": plan_approved, "plan_revisions": plan_revisions,
        "advisory": (
            "The result is empty or zero even after the plan was revised."
            if _answer_is_empty() else ""
        ),
    })
    result.rejected_reason = ""
    return _finalize(result)


def _last_plan_feedback(trace: list[dict[str, Any]]) -> str:
    """Why the last plan attempt was rejected, for the decline reason."""
    for event in reversed(trace):
        if event.get("stage") != "plan" or event.get("approved") is not False:
            continue
        judgment = event.get("plan_judgment") or {}
        parts = [str(event["planner_error"])] if event.get("planner_error") else []
        parts += list(event.get("plan_diagnostics") or [])
        parts += [f"{v.column}: {v.suggestion}" for v in event.get("lineage_violations") or []]
        if judgment and not judgment.get("approves_question_fidelity"):
            parts.append(f"{judgment.get('rationale', '')} Missing: {judgment.get('missing_requirements')}")
        return " | ".join(part for part in parts if part)
    return ""
