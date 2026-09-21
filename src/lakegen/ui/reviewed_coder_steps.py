"""Chainlit rendering for `lakegen.reviewed_coder`'s per-stage trace events,
styled after OrQa's `pipeline_logger.py` (`orqa/src/orqa/utils/
pipeline_logger.py`): a consistent visual vocabulary of badges (a step's
`name`, standing in for `_badge`'s bold-on-color terminal text), ✔/✖/⚡/⚠
markers for approved/rejected/errored/warned, indented bullet lists for
feedback/suggestions under each verdict, and a per-judge vote tally line
(`panel_votes`'s `question 3/3 · plan 2/3` style, adapted here to the plan
judge's single vote and the code judge's two votes). Markdown stands in for
ANSI color -- Chainlit step output renders Markdown, not a terminal.

Deliberately NOT a literal one-for-one port of `query_plan`'s step/`produces`
rendering: LakeGen's `operation_trace` doesn't carry OrQa's ordered-step/
`produces` lineage structure (see `lakegen.column_lineage`'s module
docstring -- `coder_brief` is unreliable, so the plan stage judges
`operation_trace` + the actual generated code post-hoc instead of a real
pre-code plan). What's rendered here is the trace-based analogue: which
tables/columns the executed code actually touched, not a declared plan.

Each `format_*` function takes one trace event dict exactly as
`lakegen.reviewed_coder`'s `_emit` produces it (see that module's stage
loops) and returns Markdown step output. Pure string building -- no I/O, no
Chainlit imports here, so these are trivially unit-testable; the Chainlit
wiring itself lives in `emit_review_step` at the bottom, which is the only
function that touches `ThreadSafeStepEmitter`.
"""
from __future__ import annotations

from typing import Any

CHECK = "✔"
CROSS = "✖"
BOLT = "⚡"
WARN = "⚠"


def _mark(ok: bool) -> str:
    return CHECK if ok else CROSS


def step_name_for_event(event: dict[str, Any]) -> str:
    """The step's badge -- Chainlit renders `cl.Step.name` prominently, so
    this stands in for OrQa's `_badge(text, bg)`."""
    stage = event.get("stage")
    attempt = event.get("attempt")
    max_attempts = event.get("max_attempts")
    tag = f" · attempt {attempt}/{max_attempts}" if attempt and max_attempts else ""
    if stage == "plan":
        return f"🧭 Plan Review{tag}"
    if stage == "validator":
        return f"🔍 Query Validator{tag}"
    if stage == "code_judge":
        return f"⚖️ Code Judge{tag}"
    if stage == "summary":
        return "🏁 Pipeline Summary"
    return f"Review stage: {stage}"


def _format_generation_error(event: dict[str, Any]) -> str:
    return f"{BOLT} **Generation error**  \n{event.get('event', '')}"


def format_plan_step(event: dict[str, Any]) -> str:
    """A `Violation` (`lakegen.column_lineage`) carries `.column`/
    `.category`/`.step_order`/`.suggestion`; `plan_judgment` is the raw dict
    `lakegen.plan_judge.judge_plan_question_fidelity` returns."""
    if "approved" not in event:
        return _format_generation_error(event)

    approved = bool(event.get("approved"))
    lines = [f"{_mark(approved)} **{'Approved' if approved else 'Rejected'}**"]

    lineage_violations = event.get("lineage_violations") or []
    if lineage_violations:
        lines.append("")
        lines.append(f"{WARN} **Lineage check** -- columns not in the real schema:")
        for v in lineage_violations:
            suggestion = f" -- {v.suggestion}" if getattr(v, "suggestion", None) else ""
            lines.append(f"- `{v.column}` ({v.category}){suggestion}")
    else:
        lines.append(f"\n{CHECK} Lineage check: every referenced column resolves against the real schema.")

    judgment = event.get("plan_judgment") or {}
    if judgment:
        judge_ok = judgment.get("approves_question_fidelity") is True
        lines.append("")
        lines.append(
            f"{_mark(judge_ok)} **Plan judge (question fidelity)**  \n"
            f"{judgment.get('rationale', '').strip() or '_(no rationale given)_'}"
        )
        missing = judgment.get("missing_requirements") or []
        if missing:
            lines.append("  \n**Missing:** " + ", ".join(f"`{m}`" for m in missing))
        unjustified = judgment.get("unjustified_steps") or []
        if unjustified:
            lines.append("  \n**Unjustified steps:** " + ", ".join(str(s) for s in unjustified))
        if judgment.get("judge_error"):
            lines.append(f"  \n{BOLT} judge error (failed open, treated as approved): {judgment['judge_error']}")

    return "\n".join(lines)


def format_validator_step(event: dict[str, Any]) -> str:
    if "approved" not in event:
        return _format_generation_error(event)

    approved = bool(event.get("approved"))
    lines = [f"{_mark(approved)} **{'Cleared' if approved else 'Flagged'}**"]

    violations = event.get("grounding_violations") or []
    if violations:
        lines.append("")
        lines.append("**Value-grounding checks:**")
        for v in violations:
            marker = CROSS if v.get("confidence") == "HIGH" else WARN
            lines.append(
                f"- {marker} `{v['column']}` = `{v['literal']!r}` "
                f"({v.get('confidence', '?')})  \n  {v.get('message', '')}"
            )
    else:
        lines.append(f"\n{CHECK} No value-grounding or date-parse issues found.")

    if event.get("validator_flagged") and event.get("validator_reason"):
        lines.append(f"\n{CROSS} **Composed validator verdict:**  \n{event['validator_reason']}")

    return "\n".join(lines)


# The code judge's two votes, mirroring `panel_votes`'s per-layer tally
# (`question 3/3 · plan 2/3`), adapted to a single judge casting two named
# votes instead of a multi-judge panel casting one vote each.
_CODE_JUDGE_VOTES = (
    ("plan_compliance_approval", "plan compliance"),
    ("present_result_approval", "present result"),
)


def format_code_judge_step(event: dict[str, Any]) -> str:
    if "approved" not in event:
        return _format_generation_error(event)

    approved = bool(event.get("approved"))
    judgment = event.get("code_judgment") or {}
    lines = [f"{_mark(approved)} **{'Approved' if approved else 'Rejected'}**"]

    vote_bits = [
        f"{label} {_mark(bool(judgment.get(field)))}"
        for field, label in _CODE_JUDGE_VOTES
        if field in judgment
    ]
    if vote_bits:
        lines.append("  \n" + "  ·  ".join(vote_bits))

    violated = judgment.get("violated_criteria") or []
    if violated:
        lines.append("\n**Violated criteria:** " + ", ".join(f"`{c}`" for c in violated))

    if judgment.get("feedback"):
        lines.append(f"\n**Feedback:**  \n{judgment['feedback']}")

    suggestions = judgment.get("suggestions") or []
    if suggestions:
        lines.append("\n**Suggestions:**")
        for s in suggestions:
            lines.append(f"- {s}")

    if judgment.get("judge_error"):
        lines.append(f"\n{BOLT} judge error (failed open, treated as approved): {judgment['judge_error']}")

    return "\n".join(lines)


def format_pipeline_summary(event: dict[str, Any]) -> str:
    outcome = event.get("outcome")
    validated = outcome == "validated"
    lines = [
        f"{_mark(validated)} **{'Validated' if validated else 'Declined'}**",
        f"generation calls: {event.get('generation_calls', '?')}  ·  "
        f"elapsed: {event.get('wall_time', '?')}s",
    ]
    if not validated and event.get("reason"):
        lines.append(f"\n**Reason:**  \n{event['reason']}")
    return "\n".join(lines)


_FORMATTERS = {
    "plan": format_plan_step,
    "validator": format_validator_step,
    "code_judge": format_code_judge_step,
    "summary": format_pipeline_summary,
}


def format_stage_event(event: dict[str, Any]) -> str:
    """Dispatch a raw `lakegen.reviewed_coder` trace event to its formatter."""
    formatter = _FORMATTERS.get(event.get("stage"))
    if formatter is None:
        return str(event)
    return formatter(event)


def emit_review_step(emitter: "ThreadSafeStepEmitter", event: dict[str, Any], step_counter: dict[str, int]) -> None:  # noqa: F821
    """The `on_stage_event` callback `ui/workflow.py` passes to
    `phase3_generate_and_execute_reviewed` -- renders one trace event as one
    Chainlit step, nested under the same live run the user is already
    watching (`emitter` is already constructed with `parent_step=step`).
    `step_counter` (a plain dict the caller owns, e.g. `{"n": 0}`) keeps
    every step_id unique across the whole reviewed-pipeline call, since
    `ThreadSafeStepEmitter.instant` needs one per step."""
    step_counter["n"] = step_counter.get("n", 0) + 1
    step_id = f"review-{step_counter['n']}"
    emitter.instant(step_id, step_name_for_event(event), format_stage_event(event))
