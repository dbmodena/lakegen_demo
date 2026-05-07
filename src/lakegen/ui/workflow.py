from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import chainlit as cl

from lakegen.ui.sections import (
    build_phase1_summary,
    build_phase2_summary,
    build_phase3_summary,
    build_phase4_summary,
    build_phase5_summary,
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
    phase1_retrieve_candidates,
    phase2_select_tables,
    phase3_generate_code,
    phase4_execute,
    phase5_synthesize,
)
from lakegen.resources import (
    get_all_csv_files,
    get_llm,
    get_prompt_manager,
    get_solr,
)
from src.utils import save_experiment_log


WORKFLOW_LOCK = asyncio.Lock()
MAX_RETRIES = 3


@dataclass
class ExecutionOutcome:
    status: str
    reason: str = ""


def _action_value(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        payload = response.get("payload") or {}
        return str(payload.get("value") or "")
    payload = getattr(response, "payload", {}) or {}
    return str(payload.get("value") or "")


async def _ask_choice(content: str, choices: list[tuple[str, str, str]]) -> str:
    response = await cl.AskActionMessage(
        content=content,
        actions=[
            cl.Action(name=name, payload={"value": value}, label=label)
            for name, value, label in choices
        ],
        timeout=24 * 60 * 60,
        raise_on_timeout=False,
    ).send()
    return _action_value(response)


async def _ask_hint(content: str) -> str:
    response = await cl.AskUserMessage(
        content=f"{content}\n\n{t('hint.skip_suffix')}",
        timeout=10 * 60,
        raise_on_timeout=False,
    ).send()
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
) -> None:
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


async def _run_keyword_gate(session: LakeGenSession, llm, pm, initial_hint: str) -> None:
    hint = initial_hint
    label = (
        session.text("phase1.fallback_regeneration")
        if hint
        else session.text("phase1.initial_generation")
    )
    while True:
        await _generate_keywords(session, llm, pm, hint, label)
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
        )
        if action == "approve":
            await cl.Message(content=build_phase1_summary(session, hint)).send()
            return
        session.check_cancelled()
        hint = await _ask_hint(
            session.text("phase1.change_hint"),
        )
        label = session.text("phase1.recalculation")


async def _select_tables_once(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_csv: list[str],
    *,
    initial_retrieval: bool,
    hint: str,
    accumulate_tokens: bool,
) -> bool:
    async with cl.Step(name=session.text("phase2.step"), type="run", default_open=True) as step:
        async with StepStreamBridge(step) as bridge:
            if initial_retrieval:
                result = await cl.make_async(_retrieve_and_select_tables)(
                    session,
                    llm,
                    pm,
                    solr,
                    all_csv,
                    hint,
                    bridge.emit,
                )
            else:
                result = await cl.make_async(phase2_select_tables)(
                    session.query,
                    llm,
                    pm,
                    all_csv,
                    session.candidates,
                    session.solr_metadata_map,
                    session.runtime.csv_dir,
                    session.runtime.db_path,
                    hint=hint,
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
            return False

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
        return True


def _retrieve_and_select_tables(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_csv: list[str],
    hint: str,
    stream_callback,
):
    cands, smeta, activity_log_parts = phase1_retrieve_candidates(
        session.keywords,
        solr,
        all_csv,
        stream_callback=stream_callback,
    )
    return phase2_select_tables(
        session.query,
        llm,
        pm,
        all_csv,
        cands,
        smeta,
        session.runtime.csv_dir,
        session.runtime.db_path,
        activity_log_parts=activity_log_parts,
        hint=hint,
        stream_callback=stream_callback,
        cancel_check=session.check_cancelled,
    )


async def _run_table_gate(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_csv: list[str],
    *,
    initial_retrieval: bool,
    initial_hint: str = "",
) -> str:
    hint = initial_hint
    rerun_retrieval = initial_retrieval
    first = True

    while True:
        session.check_cancelled()
        ok = await _select_tables_once(
            session,
            llm,
            pm,
            solr,
            all_csv,
            initial_retrieval=rerun_retrieval,
            hint=hint,
            accumulate_tokens=not first,
        )

        first = False
        rerun_retrieval = False

        if not ok:
            action = await _ask_choice(
                session.text(
                    "phase2.architect_rejected",
                    feedback=session.fallback_reason,
                ),
                [
                    (
                        "regenerate_keywords",
                        "regenerate",
                        session.text("phase2.generate_keywords"),
                    )
                ],
            )
            return "keywords_rejected" if action == "regenerate" else "cancelled"

        action = await _ask_choice(
            session.text(
                "phase2.review_tables",
                tables="\n".join(f"- `{table}`" for table in session.tables),
            ),
            [
                ("approve_tables", "approve", session.text("phase2.approve")),
                ("recalculate_tables", "recalculate", session.text("phase2.recalculate")),
            ],
        )

        if action == "approve":
            await cl.Message(content=build_phase2_summary(session, hint)).send()
            return "approved"

        session.check_cancelled()
        hint = await _ask_hint(
            session.text("phase2.change_hint"),
        )
        rerun_retrieval = True


async def _run_execution(session: LakeGenSession, llm, pm) -> ExecutionOutcome:
    retries = 0
    error_msg = ""
    final_code = ""
    raw_result = None
    err = None
    code_attempts: list[dict[str, Any]] = []
    execution_attempts: list[dict[str, Any]] = []

    while retries < MAX_RETRIES:
        session.check_cancelled()
        async with cl.Step(name=session.text("phase3.step"), type="llm", default_open=True) as step:
            async with StepStreamBridge(step) as bridge:
                code_box = CumulativeMarkdownEmitter(
                    bridge.emit,
                    session.text("phase3.code_stream"),
                )
                reasoning_box = CumulativeMarkdownEmitter(
                    bridge.emit,
                    session.text("phase3.model_reasoning"),
                )
                code_raw, tok3 = await cl.make_async(phase3_generate_code)(
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
                    force_execution=session.force_execution,
                    stream_placeholder=code_box,
                    reasoning_placeholder=reasoning_box,
                )
            session.tokens["p3"] += tok3
            generation_attempt = {
                "attempt": retries + 1,
                "feedback": error_msg,
                "raw_response": code_raw,
                "tokens": tok3,
                "status": "generated",
            }
            step.output = code_raw

        if "REJECT_TABLES" in code_raw and not session.force_execution:
            reason = code_raw.replace("REJECT_TABLES:", "").replace("REJECT_TABLES", "").strip()
            session.fallback_reason = reason
            generation_attempt["status"] = "rejected tables"
            code_attempts.append(generation_attempt)
            await cl.Message(content=build_phase3_summary(session, code_attempts)).send()
            return ExecutionOutcome(status="tables_rejected", reason=reason)

        async with cl.Step(name=session.text("phase4.step"), type="tool", default_open=True) as step:
            raw_result, err, clean_code = await cl.make_async(phase4_execute)(
                code_raw,
                run_dir=session.run_dir,
            )
            final_code = clean_code
            generation_attempt["clean_code"] = clean_code
            code_attempts.append(generation_attempt)
            if err is None:
                execution_attempts.append({
                    "attempt": retries + 1,
                    "status": "success",
                    "output": raw_result or "",
                })
                step.output = raw_result or session.text("phase4.success")
                break

            execution_attempts.append({
                "attempt": retries + 1,
                "status": "error",
                "error": err,
            })
            step.output = err
            error_msg = err
            retries += 1

    if raw_result is None:
        raw_result = f"Execution failed after {MAX_RETRIES} attempts. Last error: {error_msg}"

    async with cl.Step(name=session.text("phase5.step"), type="llm", default_open=True) as step:
        answer, tok5 = await cl.make_async(phase5_synthesize)(
            session.query,
            raw_result,
            llm,
            pm,
        )
        session.tokens["p5"] = tok5
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
            f"{build_phase4_summary(execution_attempts)}\n\n"
            f"{build_phase5_summary(session, answer)}"
        ),
        elements=elements,
    ).send()

    save_experiment_log(
        question=session.query,
        code=final_code,
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
        tokens_phase5=session.tokens["p5"],
        error=err if err is not None else "",
    )
    return ExecutionOutcome(status="done")


async def _run_locked_workflow(question: str) -> None:
    session = get_session()
    runtime = get_runtime_settings()
    session.runtime = runtime
    session.reset_for_query(question)

    llm, _token_counter = get_llm(runtime.model_name, runtime.ollama_url)
    solr = get_solr(runtime.solr_core)
    pm = get_prompt_manager()
    all_csv = get_all_csv_files(runtime.csv_dir)

    keyword_hint = ""
    while True:
        await _run_keyword_gate(session, llm, pm, keyword_hint)
        table_status = await _run_table_gate(
            session,
            llm,
            pm,
            solr,
            all_csv,
            initial_retrieval=True,
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
            table_status = await _run_table_gate(
                session,
                llm,
                pm,
                solr,
                all_csv,
                initial_retrieval=False,
                initial_hint=(
                    "Previous selection rejected by Code Generator. "
                    f"Coder feedback: {outcome.reason}"
                ),
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
