from __future__ import annotations
import uuid
import os

import asyncio
from dataclasses import dataclass
from typing import Any

import chainlit as cl

from lakegen.ui.sections import (
    build_phase1_summary,
    build_phase2_summary,
    build_phase3_summary,
    build_phase4_summary,
)
from lakegen.ui.i18n import t
from lakegen.ui.state import (
    LakeGenSession,
    WorkflowCancelled,
    apply_phase2_keyword_rejection,
    get_runtime_settings,
    get_session,
)
from lakegen.ui.streaming import (
    CumulativeMarkdownEmitter,
    StepStreamBridge,
)
from lakegen.phases import (
    phase1_generate_keywords,
    phase2_select_tables,
    phase1_updated_agent,
    phase3_generate_and_execute,
    phase4_synthesize,
)
from lakegen.resources import (
    get_all_csv_files,
    get_llm,
    get_prompt_manager,
    get_solr,
)
from lakegen.utils import save_experiment_log, BASE_DIR


WORKFLOW_LOCK = asyncio.Lock()
MAX_RETRIES = 3


@dataclass
class ExecutionOutcome:
    status: str
    reason: str = ""


def _fenced_text(content: str) -> str:
    fence = "```"
    while fence in content:
        fence += "`"
    return f"{fence}text\n{content}\n{fence}"


def _format_phase3_attempt_block(
    session: LakeGenSession,
    attempt: dict[str, Any],
    output: str | None,
) -> str:
    status = attempt.get("status", "generated")
    status_label = t(f"status.{str(status).lower().replace(' ', '_')}", default=str(status))
    rendered_output = output or session.text("phase3.success")
    return (
        f"### {session.text('summary.attempt')} {attempt.get('attempt')} - {status_label}\n\n"
        f"{_fenced_text(rendered_output)}\n\n"
        f"- {session.text('summary.tokens')}: `{attempt.get('tokens', 0)}`"
    )


def _action_value(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        payload = response.get("payload") or {}
        return str(payload.get("value") or "")
    payload = getattr(response, "payload", {}) or {}
    return str(payload.get("value") or "")


async def _ask_choice(
    content: str,
    choices: list[tuple[str, str, str]],
    *,
    remove_after_answer: bool = False,
) -> str:
    message = cl.AskActionMessage(
        content=content,
        actions=[
            cl.Action(name=name, payload={"value": value}, label=label)
            for name, value, label in choices
        ],
        timeout=24 * 60 * 60,
        raise_on_timeout=False,
    )
    response = await message.send()
    if remove_after_answer:
        await message.remove()
    return _action_value(response)


async def _ask_hint(content: str, *, remove_after_answer: bool = False) -> str:
    message = cl.AskUserMessage(
        content=f"{content}\n\n{t('hint.skip_suffix')}",
        timeout=10 * 60,
        raise_on_timeout=False,
    )
    response = await message.send()
    if remove_after_answer:
        await message.remove()
    if not response:
        return ""
    hint = str(response.get("output") or "").strip()
    return "" if hint.lower() in {"", "skip", "none", "no"} else hint


def _keyword_list(keywords: list[str]) -> str:
    return ", ".join(f"`{kw}`" for kw in keywords) or t("summary.none")


async def _generate_keywords(
    session: LakeGenSession,
    llm,
    pm,
    hint: str,
    label: str,
) -> cl.Step:
    # Gather previously generated keywords in the session to avoid repeating them
    avoid_kws = []
    for run in session.phase1_runs:
        avoid_kws.extend(run.get("keywords", []))
    avoid_kws = list(dict.fromkeys(avoid_kws))

    async with cl.Step(name=session.text("phase1.step"), type="llm", default_open=True) as step:
        async with StepStreamBridge(step) as bridge:
            stream_box = CumulativeMarkdownEmitter(
                bridge.emit,
                session.text("phase1.keyword_stream"),
            )
            reasoning_box = CumulativeMarkdownEmitter(
                bridge.emit,
                session.text("phase1.model_reasoning"),
            )
            kws, raw, tok, reasoning = await cl.make_async(phase1_generate_keywords)(
                session.query,
                llm,
                pm,
                hint=hint,
                portal_name=session.runtime.portal_name,
                stream_placeholder=stream_box,
                reasoning_placeholder=reasoning_box,
                avoid_keywords=avoid_kws,
            )
        session.keywords = kws
        session.raw_keywords = raw
        session.tokens["p1"] += tok
        session.record_phase1_run(label, hint, kws, raw, tok, reasoning)
        step.output = (
            f"{t('summary.keywords').title()}: "
            f"{_keyword_list(kws)}\n\n"
            f"{t('summary.tokens').title()}: `{tok}`"
        )
    return step


async def _run_keyword_gate(session: LakeGenSession, llm, pm, initial_hint: str) -> None:
    hint = initial_hint
    label = (
        session.text("phase1.fallback_regeneration")
        if hint
        else session.text("phase1.initial_generation")
    )
    while True:
        phase1_step = await _generate_keywords(session, llm, pm, hint, label)
        session.check_cancelled()
        action = await _ask_choice(
            session.text(
                "phase1.review_keywords",
                keywords=_keyword_list(session.keywords),
            ),
            [
                ("approve_keywords", "approve", session.text("phase1.approve")),
                ("recalculate_keywords", "recalculate", session.text("phase1.recalculate")),
            ],
            remove_after_answer=True,
        )
        if action == "approve":
            phase1_step.output = build_phase1_summary(session, hint)
            await phase1_step.update()
            return
        session.check_cancelled()
        hint = await _ask_hint(
            session.text("phase1.change_hint"),
            remove_after_answer=True,
        )
        label = session.text("phase1.recalculation")


async def _select_tables_once(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_csv: list[str],
    *,
    hint: str,
    accumulate_tokens: bool,
) -> tuple[bool, cl.Step]:
    async with cl.Step(
        name=session.text("phase2.step"),
        type="run",
        default_open=True,
        auto_collapse=True,
    ) as step:
        async with StepStreamBridge(step) as bridge:
            result = await cl.make_async(phase2_select_tables)(
                query=session.query,
                llm=llm,
                pm=pm,
                all_files=all_csv,
                keywords=session.keywords,
                solr_client=solr,
                csv_dir=session.runtime.csv_dir,
                hint=hint,
                portal_name=session.runtime.portal_name,
                stream_callback=bridge.emit,
                cancel_check=session.check_cancelled,
            )

        sel, cands, smeta, reasoning, trace, tok2 = result
        if apply_phase2_keyword_rejection(
            session,
            cands,
            smeta,
            reasoning,
            trace,
            tok2,
            accumulate_tokens=accumulate_tokens,
        ):
            step.output = session.text(
                "phase2.keywords_rejected",
                reason=session.fallback_reason,
            )
            return False, step

        session.tables = sel
        session.candidates = cands
        session.solr_metadata_map = smeta
        session.architect_reasoning = reasoning
        session.full_trace = trace
        if accumulate_tokens:
            session.tokens["p2"] += tok2
        else:
            session.tokens["p2"] = tok2
        step.output = build_phase2_summary(session, hint)
        return True, step


async def _run_table_gate(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_csv: list[str],
    *,
    initial_hint: str = "",
) -> str:
    hint = initial_hint
    first = True

    while True:
        session.check_cancelled()
        ok, phase2_step = await _select_tables_once(
            session,
            llm,
            pm,
            solr,
            all_csv,
            hint=hint,
            accumulate_tokens=not first,
        )

        first = False

        if not ok:
            import chainlit as cl
            await cl.Message(
                content=session.text(
                    "phase2.architect_rejected",
                    feedback=session.fallback_reason,
                ) + "\n\n🔄 **Auto-correcting:** Sending feedback to Phase 1 for new keywords..."
            ).send()
            
            phase2_step.default_open = False
            await phase2_step.update()
            return "keywords_rejected"

        action = await _ask_choice(
            session.text(
                "phase2.review_tables",
                tables="\n".join(f"- `{table}`" for table in session.tables)
                    + f"\n\n**Reasoning:**\n{session.architect_reasoning}",
            ),
            [
                ("approve_tables", "approve", session.text("phase2.approve")),
                ("recalculate_tables", "recalculate", session.text("phase2.recalculate")),
            ],
            remove_after_answer=True,
        )

        phase2_step.default_open = False
        if action == "approve":
            phase2_step.output = build_phase2_summary(session, hint)
            await phase2_step.update()
            return "approved"
        await phase2_step.update()

        session.check_cancelled()
        hint = await _ask_hint(
            session.text("phase2.change_hint"),
            remove_after_answer=True,
        )


# ── Unified Gate (phase1_updated) — kept for A/B testing ─────────────
# Uncomment this block and comment the two-phase flow below to use the
# unified single-agent approach instead.

async def _run_unified_gate(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_csv: list[str],
    initial_hint: str = "",
) -> str:
    hint = initial_hint
    first = True

    while True:
        session.check_cancelled()
        async with cl.Step(
            name="Phase 1 & 2 (Unified Architect & Search)",
            type="run",
            default_open=True,
            auto_collapse=True
        ) as step:
            async with StepStreamBridge(step) as bridge:
                selected, keywords, smeta, reasoning, trace, tokens = await cl.make_async(phase1_updated_agent)(
                    query=session.query,
                    llm=llm,
                    pm=pm,
                    all_files=all_csv,
                    solr_client=solr,
                    csv_dir=session.runtime.csv_dir,
                    hint=hint,
                    portal_name=session.runtime.portal_name,
                    stream_callback=bridge.emit,
                    cancel_check=session.check_cancelled,
                )

            session.tables = selected
            session.keywords = keywords
            session.candidates = selected
            session.solr_metadata_map = smeta
            session.architect_reasoning = reasoning
            session.full_trace = trace
            if first:
                session.tokens["p1"] = tokens
                session.tokens["p2"] = 0
            else:
                session.tokens["p1"] += tokens

            step.output = (
                f"**Keywords used:** {_keyword_list(keywords)}\n\n"
                f"**Tables selected:** " + ", ".join(f"`{t}`" for t in selected) + "\n\n"
                f"**Reasoning:**\n{reasoning}\n\n"
                f"- Tokens: `{tokens}`\n\n"
                f"***Full agent activity log:***\n\n"
                f"{trace}\n\n"
            )

        first = False

        action = await _ask_choice(
            session.text(
                "phase2.review_tables",
                tables=f"**Keywords:** {_keyword_list(keywords)}\n\n**Tables:**\n" + "\n".join(f"- `{table}`" for table in session.tables) + f"\n\n**Reasoning:**\n{reasoning}",
            ),
            [
                ("approve_selection", "approve", "Approve Selection"),
                ("recalculate_selection", "recalculate", "Recalculate (change hint)"),
            ],
            remove_after_answer=True,
        )

        step.default_open = False
        if action == "approve":
            await step.update()
            return "approved"

        await step.update()

        session.check_cancelled()
        hint = await _ask_hint(
            "What should the agent change? (e.g., use different keywords, or look for different tables)",
            remove_after_answer=True,
        )


async def _run_execution(session: LakeGenSession, llm, pm) -> ExecutionOutcome:
    retries = 0
    error_msg = ""
    final_code = ""
    raw_result = None
    err = None
    code_attempts: list[dict[str, Any]] = []
    attempt_blocks: list[str] = []

    async with cl.Step(name=session.text("phase3.step"), type="run", default_open=True) as step:
        while retries < MAX_RETRIES:
            session.check_cancelled()
            attempt_no = retries + 1
            async with StepStreamBridge(step) as bridge:
                bridge.emit(f"\n\n## {session.text('summary.attempt')} {attempt_no}\n")
                code_box = CumulativeMarkdownEmitter(
                    bridge.emit,
                    session.text("phase3.code_stream"),
                )
                reasoning_box = CumulativeMarkdownEmitter(
                    bridge.emit,
                    session.text("phase3.model_reasoning"),
                )
                phase3_result = await cl.make_async(phase3_generate_and_execute)(
                    session.query,
                    session.tables,
                    session.candidates,
                    session.solr_metadata_map,
                    session.architect_reasoning,
                    llm,
                    pm,
                    session.runtime.csv_dir,
                    retries=retries,
                    error_msg=error_msg,
                    previous_code=final_code,
                    force_execution=session.force_execution,
                    stream_placeholder=code_box,
                    reasoning_placeholder=reasoning_box,
                    cancel_check=session.check_cancelled,
                    run_dir=session.run_dir,
                )

            session.tokens["p3"] += phase3_result.tokens
            final_code = phase3_result.clean_code
            raw_result = phase3_result.raw_result
            err = phase3_result.error
            generation_attempt = {
                "attempt": attempt_no,
                "correction_feedback": error_msg,
                "error": phase3_result.error,
                "raw_response": phase3_result.code_raw,
                "clean_code": phase3_result.clean_code,
                "tokens": phase3_result.tokens,
                "status": "success" if phase3_result.error is None else "error",
            }

            if phase3_result.rejected_reason:
                reason = phase3_result.rejected_reason
                session.fallback_reason = reason
                generation_attempt["status"] = "rejected tables"
                code_attempts.append(generation_attempt)
                attempt_blocks.append(
                    _format_phase3_attempt_block(
                        session,
                        generation_attempt,
                        phase3_result.code_raw,
                    )
                )
                step.output = "\n\n".join(attempt_blocks)
                await step.update()
                await cl.Message(content=build_phase3_summary(session, code_attempts)).send()
                return ExecutionOutcome(status="tables_rejected", reason=reason)

            code_attempts.append(generation_attempt)
            if err is None:
                attempt_blocks.append(
                    _format_phase3_attempt_block(
                        session,
                        generation_attempt,
                        raw_result or session.text("phase3.success"),
                    )
                )
                step.output = "\n\n".join(attempt_blocks)
                break

            attempt_blocks.append(
                _format_phase3_attempt_block(
                    session,
                    generation_attempt,
                    err,
                )
            )
            step.output = "\n\n".join(attempt_blocks)
            await step.update()
            error_msg = err
            retries += 1

    if raw_result is None:
        raw_result = f"Execution failed after {MAX_RETRIES} attempts. Last error: {error_msg}"

    async with cl.Step(name=session.text("phase4.step"), type="llm", default_open=True) as step:
        answer, tok4 = await cl.make_async(phase4_synthesize)(
            session.query,
            raw_result,
            llm,
            pm,
        )
        session.tokens["p4"] = tok4
        step.output = answer

    elements = [
        cl.Text(
            name="generated_code.py",
            content=final_code or "# No executable code captured.",
            language="python",
            display="side",
        ),
        cl.Text(
            name="execution_output.txt",
            content=str(raw_result or ""),
            language="text",
            display="side",
        ),
    ]
    await cl.Message(
        content=(
            f"### {session.text('result.final')}\n{answer}\n\n"
            f"{build_phase3_summary(session, code_attempts)}\n\n"
            f"{build_phase4_summary(session, answer)}"
        ),
        elements=elements,
    ).send()

    code_history_parts = []
    for att in code_attempts:
        code_history_parts.append(f"--- Attempt {att['attempt']} ({att['status']}) ---")
        code_history_parts.append(f"Code:\n{att['clean_code']}")
        if att['error']:
            code_history_parts.append(f"Error:\n{att['error']}\n")
        else:
            code_history_parts.append("Status: Success\n")
    full_code_history = "\n".join(code_history_parts)

    save_experiment_log(
        question=session.query,
        code=full_code_history,
        result=raw_result if raw_result else "",
        retries=retries,
        reasoning=session.architect_reasoning,
        tables=session.tables,
        raw_keywords=session.raw_keywords,
        final_keywords=session.keywords,
        debug_raw="",
        final_result=answer,
        full_trace=session.full_trace,
        tokens_phase1=session.tokens["p1"],
        tokens_phase2=session.tokens["p2"],
        tokens_phase3=session.tokens["p3"],
        tokens_phase4=session.tokens["p4"],
        error=err if err is not None else "",
    )
    return ExecutionOutcome(status="done")


async def _run_locked_workflow(question: str) -> None:
    session = get_session()
    runtime = get_runtime_settings()
    session.runtime = runtime

    llm, _token_counter = get_llm(runtime.model_name, runtime.ollama_url)
    solr = get_solr(runtime.solr_core)
    pm = get_prompt_manager()
    all_csv = get_all_csv_files(runtime.csv_dir)

    keyword_hint = ""
    while True:
        if session.runtime.use_unified_agent:
            table_status = await _run_unified_gate(
                session,
                llm,
                pm,
                solr,
                all_csv,
                initial_hint=keyword_hint,
            )
            if table_status != "approved":
                await cl.Message(content=session.text("workflow.cancelled")).send()
                return
        else:
            # ── Two-phase flow: Phase 1 (keywords) → Phase 2 (search + judge) ──
            await _run_keyword_gate(session, llm, pm, keyword_hint)
            table_status = await _run_table_gate(
                session,
                llm,
                pm,
                solr,
                all_csv,
            )
            if table_status == "keywords_rejected":
                keyword_hint = (
                    "The previous keywords led to bad tables. "
                    f"Architect feedback: {session.fallback_reason}. "
                    "Generate completely different keywords."
                )
                continue
            if table_status != "approved":
                await cl.Message(content=session.text("workflow.cancelled")).send()
                return

        session.force_execution = False
        while True:
            outcome = await _run_execution(session, llm, pm)
            if outcome.status == "done":
                return

            action = await _ask_choice(
                session.text(
                    "workflow.tables_rejected",
                    feedback=outcome.reason,
                ),
                [
                    (
                        "reevaluate_tables",
                        "reevaluate",
                        session.text("workflow.reevaluate_tables"),
                    ),
                    ("force_execution", "force", session.text("workflow.force_execution")),
                ],
            )
            if action == "force":
                session.force_execution = True
                continue

            session.force_execution = False

            # Re-run Phase 1/2 with feedback from coder
            hint_msg = (
                "Previous selection rejected by Code Generator. "
                f"Coder feedback: {outcome.reason}"
            )
            if session.runtime.use_unified_agent:
                table_status = await _run_unified_gate(
                    session, llm, pm, solr, all_csv, initial_hint=hint_msg
                )
                if table_status != "approved":
                    await cl.Message(content=session.text("workflow.cancelled")).send()
                    return
            else:
                table_status = await _run_table_gate(
                    session,
                    llm,
                    pm,
                    solr,
                    all_csv,
                    initial_hint=hint_msg,
                )
                if table_status == "keywords_rejected":
                    keyword_hint = (
                        "The previous keywords led to bad tables. "
                        f"Architect feedback: {session.fallback_reason}. "
                        "Generate completely different keywords."
                    )
                    break
                if table_status != "approved":
                    await cl.Message(content=session.text("workflow.cancelled")).send()
                    return

        # If we broke out due to keywords_rejected, loop back to Phase 1
        if keyword_hint:
            continue


async def run_lakegen_workflow(question: str) -> None:
    if not question.strip():
        await cl.Message(content=t("workflow.empty_question")).send()
        return

    if WORKFLOW_LOCK.locked():
        await cl.Message(
            content=t("workflow.locked")
        ).send()

    async with WORKFLOW_LOCK:
        try:
            await _run_locked_workflow(question.strip())
        except WorkflowCancelled:
            raise
