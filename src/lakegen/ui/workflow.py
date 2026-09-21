from __future__ import annotations
import uuid
import os
import time
import logging

import asyncio
from dataclasses import dataclass
from typing import Any

import chainlit as cl

from lakegen.ui.sections import (
    build_phase1_summary,
    build_phase2_summary,
    build_phase3_summary,
    build_phase4_summary,
    format_retrieval_keywords,
)
from lakegen.ui.i18n import t
from lakegen.ui.state import (
    LakeGenSession,
    WorkflowCancelled,
    WorkflowTimedOut,
    apply_phase2_keyword_rejection,
    get_runtime_settings,
    get_session,
)
from lakegen.ui.streaming import (
    CumulativeMarkdownEmitter,
    StepStreamBridge,
)
from lakegen.ui.live_steps import ThreadSafeStepEmitter, retrieval_observer_for
from lakegen.ui.reviewed_coder_steps import emit_review_step
from lakegen.phases import (
    phase1_generate_keywords,
    phase2_select_tables,
    phase12_agent,
    phase3_generate_and_execute,
    phase4_synthesize,
)
from lakegen.agent_tools.tools_p12 import P12State
from lakegen.core.resources import (
    get_all_table_files,
    get_llm,
    get_prompt_manager,
    get_solr,
)
from lakegen.core.logger import save_experiment_log
from lakegen.core.config import BASE_DIR, LOG_DIR
from lakegen.manifest import create_manifest, persist_manifest
from lakegen.reproducibility import initialize_reproducibility
from lakegen.tracing import (
    HumanGate,
    PhaseName,
    build_llm_phase_records,
    summarize_tool_calls,
    normalize_hint,
)
from lakegen.experiment_config import ToolAccess
from lakegen.orchestrated_context import prepare_discovery_context
from lakegen.retrieval.intent import intent_entities
from lakegen.phases.orchestrated_discovery import (
    OrchestratedContextPreparationError,
    OrchestratedSelectorError,
    RetrievalRequestProtocolError,
    run_unified_orchestrated_discovery,
    selector_retry_reason,
    select_from_prepared_context,
)

WORKFLOW_LOCK = asyncio.Lock()
MAX_RETRIES = 3
logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    status: str
    reason: str = ""


def _record_orchestrated_telemetry(session: LakeGenSession, prepared, invocations: int) -> None:
    telemetry = session.tool_access_telemetry
    telemetry["llm_invocations"] += invocations
    mode = session.runtime.retrieval.mode.value
    calls = telemetry["orchestrator_retrieval_calls"]
    calls[mode] = calls.get(mode, 0) + 1
    telemetry["prepared_candidate_count"] = prepared.prepared_candidate_count
    telemetry["retrieved_hit_count"] = prepared.retrieved_hit_count
    telemetry["prepared_context_utf8_bytes"] = len(prepared.stable_json().encode("utf-8"))


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


# A pending AskActionMessage waits on ``sio.call(..., to=sid)``, where ``sid`` is
# the websocket id captured when it was sent (chainlit/socket.py). A reconnect
# rebinds the session's emitter but cannot rebind that already-awaiting call, so
# its acknowledgement never arrives. The buttons stay on screen, and clicking one
# falls through to the HTTP action endpoint, which 404s because nothing is
# registered there. Registering real callbacks gives those clicks somewhere to
# land: the websocket reply and the HTTP reply resolve the same future, whichever
# arrives first.
_PENDING_CHOICES: dict[str, tuple[str, asyncio.Future[str]]] = {}

_CHOICE_ACTIONS = (
    "approve_keywords",
    "recalculate_keywords",
    "approve_tables",
    "recalculate_tables",
    "approve_selection",
    "recalculate_selection",
    "force_execution",
)


def _choice_session_key() -> str:
    session = getattr(cl.context, "session", None)
    return str(getattr(session, "id", "") or "")


async def _on_choice_action(action: Any) -> str:
    """Answer a choice whose websocket acknowledgement can no longer arrive."""
    payload = getattr(action, "payload", None) or {}
    pending = _PENDING_CHOICES.get(_choice_session_key())
    if pending is None:
        return "This prompt is no longer active."
    nonce, future = pending
    # The nonce stops a click on a superseded prompt from answering whichever
    # question happens to be open now.
    if payload.get("gate") != nonce or future.done():
        return "This prompt is no longer active."
    future.set_result(str(payload.get("value") or ""))
    return ""


for _choice_action in _CHOICE_ACTIONS:
    cl.action_callback(_choice_action)(_on_choice_action)


async def _ask_choice(
    content: str,
    choices: list[tuple[str, str, str]],
    *,
    phase: PhaseName,
    gate: HumanGate,
    approved_value: str,
    remove_after_answer: bool = False,
) -> str:
    nonce = uuid.uuid4().hex
    message = cl.AskActionMessage(
        content=content,
        actions=[
            cl.Action(name=name, payload={"value": value, "gate": nonce}, label=label)
            for name, value, label in choices
        ],
        timeout=24 * 60 * 60,
        raise_on_timeout=False,
    )
    started = time.monotonic()
    fallback: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    key = _choice_session_key()
    _PENDING_CHOICES[key] = (nonce, fallback)
    ask = asyncio.ensure_future(message.send())
    try:
        done, _ = await asyncio.wait(
            {ask, fallback}, return_when=asyncio.FIRST_COMPLETED
        )
        if ask in done:
            response = ask.result()
            if response is None:
                raise WorkflowTimedOut(f"Interaction timed out at {gate.value}")
            value = _action_value(response)
        else:
            value = fallback.result()
    finally:
        _PENDING_CHOICES.pop(key, None)
        if not ask.done():
            ask.cancel()
    if remove_after_answer:
        await message.remove()
    get_session().intervention_recorder.record_approval(
        phase=phase,
        gate=gate,
        approved=value == approved_value,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return value


async def _ask_hint(
    content: str,
    *,
    phase: PhaseName,
    gate: HumanGate,
    remove_after_answer: bool = False,
) -> str:
    message = cl.AskUserMessage(
        content=f"{content}\n\n{t('hint.skip_suffix')}",
        timeout=10 * 60,
        raise_on_timeout=False,
    )
    started = time.monotonic()
    response = await message.send()
    if remove_after_answer:
        await message.remove()
    if response is None:
        raise WorkflowTimedOut(f"Interaction timed out at {gate.value}")
    hint = normalize_hint(response.get("output") or "")
    get_session().intervention_recorder.record_hint(
        phase=phase,
        gate=gate,
        provided=bool(hint),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return hint


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
            phase_started = time.monotonic()
            kws, raw, tok, reasoning = await cl.make_async(phase1_generate_keywords)(
                session.query,
                llm,
                pm,
                hint=hint,
                portal_name=session.runtime.portal_name,
                stream_placeholder=stream_box,
                reasoning_placeholder=reasoning_box,
                avoid_keywords=avoid_kws,
                value_search=session.runtime.retrieval.mode.value_keywords,
                verbatim_entities=session.runtime.retrieval.mode.verbatim_entities,
            )
            session.phase_seconds["discovery"] += time.monotonic() - phase_started
            session.llm_call_counts["discovery"] += 1
        session.keywords = kws
        session.raw_keywords = raw
        session.tokens["p1"] += tok
        session.record_phase1_run(label, hint, kws, raw, tok, reasoning)
        step.output = (
            f"{t('summary.keywords').title()}: "
            f"{format_retrieval_keywords(session, kws)}\n\n"
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
                keywords=format_retrieval_keywords(session, session.keywords),
            ),
            [
                ("approve_keywords", "approve", session.text("phase1.approve")),
                ("recalculate_keywords", "recalculate", session.text("phase1.recalculate")),
            ],
            phase="discovery",
            gate=HumanGate.KEYWORD_APPROVAL,
            approved_value="approve",
            remove_after_answer=True,
        )
        if action == "approve":
            phase1_step.output = build_phase1_summary(session, hint)
            await phase1_step.update()
            return
        session.check_cancelled()
        hint = await _ask_hint(
            session.text("phase1.change_hint"),
            phase="discovery",
            gate=HumanGate.KEYWORD_HINT,
            remove_after_answer=True,
        )
        label = session.text("phase1.recalculation")


async def _select_tables_once(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_files: list[str],
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
            phase_started = time.monotonic()
            if session.runtime.experiment.tool_access == ToolAccess.ORCHESTRATED_CONTEXT:
                try:
                    prepared, smeta = await cl.make_async(prepare_discovery_context)(
                        query=session.query, keywords=session.keywords,
                        solr_client=solr, all_files=all_files,
                        retrieval_config=session.runtime.retrieval,
                        table_dir=session.runtime.csv_dir,
                        entities=intent_entities(session.raw_keywords or ""),
                    )
                except WorkflowCancelled:
                    raise
                except Exception as exc:
                    wrapped = OrchestratedContextPreparationError(str(exc))
                    session.tool_access_telemetry["preparation_error"] = f"{type(wrapped).__name__}: {wrapped}"
                    raise wrapped from exc
                cands = [candidate.dataset for candidate in prepared.candidates]
                if cands:
                    try:
                        sel, reasoning, trace, tok2 = await cl.make_async(
                            select_from_prepared_context
                        )(
                            query=session.query, llm=llm, context=prepared,
                            all_files=all_files,
                            architecture=session.runtime.experiment.discovery_architecture,
                            hint=hint, stream_callback=bridge.emit,
                            cancel_check=session.check_cancelled,
                        )
                    except WorkflowCancelled:
                        raise
                    except OrchestratedSelectorError as exc:
                        session.tool_access_telemetry["selector_error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    selector_calls = 1
                else:
                    sel, reasoning, trace, tok2 = [], "REJECT_KEYWORDS: No datasets found in the prepared context", "", 0
                    selector_calls = 0
                    session.tool_access_telemetry["empty_context_retries"] += 1
                _record_orchestrated_telemetry(session, prepared, 1 + selector_calls)
                result = (sel, cands, smeta, reasoning, trace, tok2)
                retry_reason = selector_retry_reason(sel, reasoning)
                if retry_reason is not None:
                    result = ([], cands, smeta, retry_reason, trace, tok2)
                    if cands:
                        session.tool_access_telemetry["empty_context_retries"] += 1
            else:
                # A fresh state each round mirrors the unified gate: it keeps
                # this a genuinely new judge turn, but seeding excluded/
                # carried tables from the session's cross-round memory stops
                # a table already proven insufficient from resurfacing, and
                # lets a partially-useful one carry forward.
                divided_state = P12State()
                result = await cl.make_async(phase2_select_tables)(
                    query=session.query,
                    llm=llm,
                    pm=pm,
                    all_files=all_files,
                    keywords=session.keywords,
                    solr_client=solr,
                    csv_dir=session.runtime.csv_dir,
                    hint=hint,
                    portal_name=session.runtime.portal_name,
                    stream_callback=bridge.emit,
                    cancel_check=session.check_cancelled,
                    retrieval_config=session.runtime.retrieval,
                    entities=intent_entities(session.raw_keywords or ""),
                    selection_state=divided_state,
                    excluded_tables=set(session.excluded_tables),
                    carried_tables=list(session.carried_tables),
                    carried_metadata=dict(session.carried_metadata),
                )
            session.phase_seconds["discovery"] += time.monotonic() - phase_started
            session.llm_call_counts["discovery"] += (
                selector_calls
                if session.runtime.experiment.tool_access == ToolAccess.ORCHESTRATED_CONTEXT
                else 1
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
            if session.runtime.experiment.tool_access != ToolAccess.ORCHESTRATED_CONTEXT:
                for table in divided_state.rejection_skip_tables:
                    session.excluded_tables.add(table.casefold())
                for table in divided_state.rejection_keep_tables:
                    if table.casefold() in session.excluded_tables:
                        continue
                    if table not in session.carried_tables:
                        session.carried_tables.append(table)
                    if table in smeta:
                        session.carried_metadata[table] = smeta[table]
                session.carried_tables = [
                    table for table in session.carried_tables
                    if table.casefold() not in session.excluded_tables
                ]
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
    all_files: list[str],
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
            all_files,
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
            phase="discovery",
            gate=HumanGate.DATASET_APPROVAL,
            approved_value="approve",
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
            phase="discovery",
            gate=HumanGate.DATASET_HINT,
            remove_after_answer=True,
        )


# ── Unified Gate (phase12) — kept for A/B testing ─────────────
# Uncomment this block and comment the two-phase flow below to use the
# unified single-agent approach instead.

async def _run_unified_gate(
    session: LakeGenSession,
    llm,
    pm,
    solr,
    all_files: list[str],
    initial_hint: str = "",
) -> str:
    hint = initial_hint
    first = True
    empty_context_retries = 0

    while True:
        session.check_cancelled()
        async with cl.Step(
            name="Phase 1 & 2 (Unified Architect & Search)",
            type="run",
            default_open=True,
            auto_collapse=True
        ) as step:
            async with StepStreamBridge(step) as bridge:
                phase_started = time.monotonic()
                if session.runtime.experiment.tool_access == ToolAccess.ORCHESTRATED_CONTEXT:
                    try:
                        discovery = await cl.make_async(run_unified_orchestrated_discovery)(
                            query=session.query, llm=llm, solr_client=solr,
                            all_files=all_files,
                            retrieval_config=session.runtime.retrieval, hint=hint,
                            table_dir=session.runtime.csv_dir,
                            stream_callback=bridge.emit,
                            cancel_check=session.check_cancelled,
                        )
                    except WorkflowCancelled:
                        raise
                    except RetrievalRequestProtocolError as exc:
                        session.tool_access_telemetry["request_protocol_error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    except OrchestratedContextPreparationError as exc:
                        session.tool_access_telemetry["preparation_error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    except OrchestratedSelectorError as exc:
                        session.tool_access_telemetry["selector_error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    selected, keywords, smeta = discovery.selected_datasets, discovery.keywords, discovery.metadata
                    reasoning, trace, tokens = discovery.reasoning, discovery.trace, discovery.tokens
                    unified_calls = discovery.llm_invocations
                    _record_orchestrated_telemetry(
                        session, discovery.prepared_context, discovery.llm_invocations
                    )
                else:
                    # A fresh P12State each round keeps this a genuinely new
                    # retrieval turn (reusing it would block new searches
                    # after the first inspected candidate), but seeding it
                    # with the session's accumulated cross-round memory keeps
                    # a table already proven insufficient from silently
                    # resurfacing (mirrors DIVIDED's excluded_tables/
                    # carried_tables in service.py).
                    unified_state = P12State()
                    unified_state.excluded_tables = set(session.excluded_tables)
                    unified_state.carried_tables = list(session.carried_tables)
                    unified_state.carried_metadata = dict(session.carried_metadata)
                    unified_state.inspection_cache = dict(session.carried_inspection)
                    # One step per Solr query, showing exactly what was sent
                    # (keywords, q_op, entities, configured vs. actual fetch
                    # width) -- not just _emit_notice's one-line summary.
                    step_emitter = ThreadSafeStepEmitter(parent_step=step)
                    try:
                        selected, keywords, smeta, reasoning, trace, tokens = await cl.make_async(phase12_agent)(
                            query=session.query,
                            llm=llm,
                            pm=pm,
                            all_files=all_files,
                            solr_client=solr,
                            csv_dir=session.runtime.csv_dir,
                            hint=hint,
                            portal_name=session.runtime.portal_name,
                            stream_callback=bridge.emit,
                            cancel_check=session.check_cancelled,
                            retrieval_config=session.runtime.retrieval,
                            retrieval_observer=retrieval_observer_for(step_emitter),
                            discovery_config=session.runtime.discovery,
                            state=unified_state,
                        )
                    finally:
                        await step_emitter.aclose()
                    unified_calls = 1
                    if reasoning.startswith("REJECT_KEYWORDS:"):
                        for table in unified_state.rejection_skip_tables:
                            session.excluded_tables.add(table.casefold())
                        for table in unified_state.rejection_keep_tables:
                            if table.casefold() in session.excluded_tables:
                                continue
                            if table not in session.carried_tables:
                                session.carried_tables.append(table)
                            if table in smeta:
                                session.carried_metadata[table] = smeta[table]
                            cached_inspection = unified_state.inspection_cache.get(
                                table.casefold()
                            )
                            if cached_inspection:
                                session.carried_inspection[table.casefold()] = cached_inspection
                        session.carried_tables = [
                            table for table in session.carried_tables
                            if table.casefold() not in session.excluded_tables
                        ]
                        session.carried_inspection = {
                            key: value for key, value in session.carried_inspection.items()
                            if key not in session.excluded_tables
                        }
                session.phase_seconds["discovery"] += time.monotonic() - phase_started
                session.llm_call_counts["discovery"] += unified_calls

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
                f"**Keywords used:** {format_retrieval_keywords(session, keywords)}\n\n"
                f"**Tables selected:** " + ", ".join(f"`{t}`" for t in selected) + "\n\n"
                f"**Reasoning:**\n{reasoning}\n\n"
                f"- Tokens: `{tokens}`\n\n"
                f"***Full agent activity log:***\n\n"
                f"{trace}\n\n"
            )

        if (
            session.runtime.experiment.tool_access == ToolAccess.ORCHESTRATED_CONTEXT
            and not selected
        ):
            empty_context_retries += 1
            session.tool_access_telemetry["empty_context_retries"] += 1
            if empty_context_retries >= MAX_RETRIES:
                session.execution_error = reasoning
                return "failed"
            hint = reasoning
            first = False
            continue

        first = False

        action = await _ask_choice(
            session.text(
                "phase2.review_tables",
                tables=f"**Keywords:** {format_retrieval_keywords(session, keywords)}\n\n**Tables:**\n" + "\n".join(f"- `{table}`" for table in session.tables) + f"\n\n**Reasoning:**\n{reasoning}",
            ),
            [
                ("approve_selection", "approve", "Approve Selection"),
                ("recalculate_selection", "recalculate", "Recalculate (change hint)"),
            ],
            phase="discovery",
            gate=HumanGate.DATASET_APPROVAL,
            approved_value="approve",
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
            phase="discovery",
            gate=HumanGate.DATASET_HINT,
            remove_after_answer=True,
        )


async def _run_execution(session: LakeGenSession, llm, pm) -> ExecutionOutcome:
    session.phase = "code"
    retries = 0
    error_msg = ""
    final_code = ""
    raw_result = None
    err = None
    code_attempts: list[dict[str, Any]] = []
    attempt_blocks: list[str] = []

    reviewers = session.runtime.experiment.reviewers
    # Same reasoning as service.py's outer loop: the reviewed pipeline
    # (lakegen.reviewed_coder) embeds its own bounded plan/validator/code-
    # judge retry loops internally, so this outer attempt loop runs exactly
    # once when either reviewer is enabled -- unless the user already chose
    # "Force execution" on a prior review decline, in which case this run
    # falls back to the plain, unreviewed path (matching what force_execution
    # already means for a plain table rejection).
    use_reviewed_pipeline = (reviewers.plan or reviewers.code) and not session.force_execution
    effective_max_retries = 1 if use_reviewed_pipeline else MAX_RETRIES

    async with cl.Step(name=session.text("phase3.step"), type="run", default_open=True) as step:
        while retries < effective_max_retries:
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
                phase_started = time.monotonic()
                if use_reviewed_pipeline:
                    from lakegen.reviewed_coder import phase3_generate_and_execute_reviewed

                    # Renders each plan/validator/code-judge stage attempt
                    # as its own live step nested under this run, styled
                    # after OrQa's pipeline_logger (see
                    # lakegen.ui.reviewed_coder_steps). on_stage_event fires
                    # synchronously from the make_async worker thread, so it
                    # needs the thread-safe emitter, not a bare cl.Step call.
                    review_step_emitter = ThreadSafeStepEmitter(parent_step=step)
                    review_step_counter = {"n": 0}

                    def _on_stage_event(event, _emitter=review_step_emitter, _counter=review_step_counter):
                        emit_review_step(_emitter, event, _counter)

                    phase3_result = await cl.make_async(phase3_generate_and_execute_reviewed)(
                        session.query,
                        session.tables,
                        session.candidates,
                        session.solr_metadata_map,
                        session.architect_reasoning,
                        llm,
                        pm,
                        session.runtime.csv_dir,
                        stage_max_retries=reviewers.stage_max_retries,
                        enable_plan_review=reviewers.plan,
                        enable_code_review=reviewers.code,
                        on_stage_event=_on_stage_event,
                        stream_placeholder=code_box,
                        reasoning_placeholder=reasoning_box,
                        cancel_check=session.check_cancelled,
                        run_dir=session.run_dir,
                        seed=session.runtime.experiment.seed,
                        seed_instruction_recorder=lambda: setattr(
                            session, "generated_code_seed_instruction_provided", True
                        ),
                        coder_context_level=session.runtime.experiment.coder_context_level,
                    )
                    # Wait for every review step's own cl.Step() to finish
                    # rendering before moving on (e.g. to phase 4 synthesis),
                    # so they appear in order rather than racing with it.
                    await review_step_emitter.aclose()
                else:
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
                        seed=session.runtime.experiment.seed,
                        seed_instruction_recorder=lambda: setattr(
                            session, "generated_code_seed_instruction_provided", True
                        ),
                        coder_context_level=session.runtime.experiment.coder_context_level,
                    )
                session.phase_seconds["code"] += time.monotonic() - phase_started
                session.llm_call_counts["code"] += 1

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

            if getattr(phase3_result, "finalization_mode", "") == "review_declined":
                # A review-stage decline (code executed fine but never
                # passed plan/validator/code review) is NOT a table/data
                # insufficiency verdict -- unlike the block below, it must
                # not be treated as a table rejection (no table banning, no
                # "the tables were wrong" framing). See
                # lakegen.reviewed_coder's module docstring.
                reason = phase3_result.rejected_reason
                session.fallback_reason = reason
                generation_attempt["status"] = "declined by review"
                code_attempts.append(generation_attempt)
                attempt_blocks.append(
                    _format_phase3_attempt_block(
                        session, generation_attempt, phase3_result.code_raw,
                    )
                )
                step.output = "\n\n".join(attempt_blocks)
                await step.update()
                await cl.Message(content=build_phase3_summary(session, code_attempts)).send()
                return ExecutionOutcome(status="review_declined", reason=reason)

            if phase3_result.rejected_reason:
                reason = phase3_result.rejected_reason
                session.fallback_reason = reason
                # The coder's own per-table judgment feeds the same
                # cross-round memory a discovery rejection does: a table it
                # proved unusable is banned outright, and one it found
                # partially useful carries forward instead of being lost
                # with the rest of the selected set.
                for table in phase3_result.rejection_skip_tables:
                    session.excluded_tables.add(table.casefold())
                for table in phase3_result.rejection_keep_tables:
                    if table.casefold() in session.excluded_tables:
                        continue
                    if table not in session.carried_tables:
                        session.carried_tables.append(table)
                    if table in session.solr_metadata_map:
                        session.carried_metadata[table] = session.solr_metadata_map[table]
                session.carried_tables = [
                    table for table in session.carried_tables
                    if table.casefold() not in session.excluded_tables
                ]
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
        session.phase = "result"
        phase_started = time.monotonic()
        answer, tok4 = await cl.make_async(phase4_synthesize)(
            session.query,
            raw_result,
            llm,
            pm,
        )
        session.phase_seconds["result"] += time.monotonic() - phase_started
        session.llm_call_counts["result"] += 1
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
            content=str(raw_result) if raw_result else "No output generated.",
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
    session.final_code = full_code_history
    session.raw_result = raw_result
    session.final_answer = answer
    session.retries = retries
    session.execution_error = str(err or "")
    return ExecutionOutcome(status="done")


def _finalize_run(session: LakeGenSession, status: str, error: str = "") -> None:
    """Persist exactly one terminal Chainlit record for the current run."""

    if session.finalized:
        return
    session.finalized = True
    safe_error = str(error or session.execution_error).replace("\n", " ")[:500]
    elapsed = round(time.monotonic() - session.started_at, 3)
    if session.runtime.experiment.tool_access == ToolAccess.AGENTIC:
        session.tool_access_telemetry.setdefault(
            "configured_tool_access", ToolAccess.AGENTIC.value
        )
        session.tool_access_telemetry["llm_invocations"] = (
            session.llm_call_counts["discovery"]
        )
        session.tool_access_telemetry["agent_direct_tools"] = [
            str(item["type"]) for item in summarize_tool_calls(session.full_trace)
        ]
    searched_keywords, unused_concepts = session.runtime.retrieval.mode.split_keywords(
        session.keywords
    )
    trace = {
        "status": status,
        "phase_reached": session.phase,
        "discovery": {
            "keywords": searched_keywords,
            "unused_concepts": unused_concepts,
            "selected_datasets": list(session.tables),
        },
        "tool_access": session.tool_access_telemetry,
        "llm_calls": build_llm_phase_records(
            total_tokens={
                "discovery": session.tokens["p1"] + session.tokens["p2"],
                "code": session.tokens["p3"],
                "result": session.tokens["p4"],
            },
            phase_invocations=session.llm_call_counts,
        ),
        "phase_metrics": {
            **{
                phase: {"latency_seconds": round(seconds, 6)}
                for phase, seconds in session.phase_seconds.items()
            },
            "total": {"latency_seconds": elapsed},
        },
        "tool_calls": summarize_tool_calls(session.full_trace),
        "retries": session.retries,
        "errors": [safe_error] if safe_error else [],
        "code": session.final_code or None,
        "execution_outcome": {
            "status": status,
            "raw_result": session.raw_result,
            "error": safe_error or None,
        },
        "human_interventions": session.intervention_recorder.to_list(),
        "configuration": session.manifest.get("resolved_config", {}),
        "reproducibility": initialize_reproducibility(
            session.runtime.experiment.seed
        ).telemetry(
            generated_code_seed_instruction_provided=(
                session.generated_code_seed_instruction_provided
            )
        ),
    }
    save_experiment_log(
        question=session.query,
        code=session.final_code,
        result=session.raw_result if session.raw_result is not None else "",
        retries=session.retries,
        reasoning=session.architect_reasoning,
        tables=session.tables,
        raw_keywords=session.raw_keywords,
        final_keywords=searched_keywords,
        unused_concepts=unused_concepts,
        final_result=session.final_answer,
        full_trace=session.full_trace,
        tokens_phase1=session.tokens["p1"],
        tokens_phase2=session.tokens["p2"],
        tokens_phase3=session.tokens["p3"],
        tokens_phase4=session.tokens["p4"],
        error=safe_error,
        model=session.runtime.model_name,
        architecture=session.runtime.experiment.architecture_name,
        status=status,
        elapsed_seconds=elapsed,
        extra_fields={"MANIFEST_JSON": session.manifest, "RUN_TRACE_JSON": trace},
    )


async def _run_locked_workflow(question: str) -> str:
    session = get_session()
    runtime = get_runtime_settings()
    session.runtime = runtime
    # The session object persists across multiple questions in one Chainlit
    # conversation; this discovery-attempt memory must not leak between them.
    session.excluded_tables = set()
    session.carried_tables = []
    session.carried_metadata = {}
    session.carried_inspection = {}
    session.tool_access_telemetry = {
        "configured_tool_access": runtime.experiment.tool_access.value,
        "execution_path": runtime.experiment.tool_access.value,
        "discovery_architecture": runtime.experiment.discovery_architecture.value,
        "agent_count": 1 if runtime.use_unified_agent else 2,
        "llm_invocations": 0,
        "retrieval_mode": runtime.retrieval.mode.value,
        "prepared_candidate_count": 0,
        "retrieved_hit_count": 0,
        "prepared_context_utf8_bytes": 0,
        "agent_direct_tools": [],
        "orchestrator_retrieval_calls": {},
        "empty_context_retries": 0,
        "request_protocol_error": None,
        "preparation_error": None,
        "selector_error": None,
    }
    session.phase = "initialization"
    manifest = create_manifest(
        runtime.experiment,
        base_dir=BASE_DIR,
        question=question,
        run_id=session.run_id,
    )
    persist_manifest(manifest, LOG_DIR / "manifests")
    session.manifest = manifest.model_dump(mode="json")

    llm, _token_counter = get_llm(runtime.model_name)
    solr = get_solr(runtime.solr_core)
    pm = get_prompt_manager()
    all_files = get_all_table_files(runtime.csv_dir)
    if not all_files:
        session.execution_error = f"No local tables found in {runtime.csv_dir}"
        await cl.Message(
            content=(
                "No CSV or Parquet files were found in "
                f"`{runtime.csv_dir}`."
            )
        ).send()
        return "failed"

    keyword_hint = ""
    session.phase = "discovery"
    while True:
        if session.runtime.use_unified_agent:
            table_status = await _run_unified_gate(
                session,
                llm,
                pm,
                solr,
                all_files,
                initial_hint=keyword_hint,
            )
            if table_status != "approved":
                if table_status == "failed":
                    return "failed"
                await cl.Message(content=session.text("workflow.cancelled")).send()
                return "cancelled"
        else:
            # ── Two-phase flow: Phase 1 (keywords) → Phase 2 (search + judge) ──
            await _run_keyword_gate(session, llm, pm, keyword_hint)
            table_status = await _run_table_gate(
                session,
                llm,
                pm,
                solr,
                all_files,
            )
            if table_status == "keywords_rejected":
                if (
                    session.runtime.experiment.tool_access
                    == ToolAccess.ORCHESTRATED_CONTEXT
                    and session.tool_access_telemetry["empty_context_retries"]
                    >= MAX_RETRIES
                ):
                    session.execution_error = session.fallback_reason
                    return "failed"
                keyword_hint = (
                    "The previous keywords led to bad tables. "
                    f"Architect feedback: {session.fallback_reason}. "
                    "Generate completely different keywords."
                )
                continue
            if table_status != "approved":
                await cl.Message(content=session.text("workflow.cancelled")).send()
                return "cancelled"

        session.force_execution = False
        while True:
            outcome = await _run_execution(session, llm, pm)
            if outcome.status == "done":
                return "completed"

            if outcome.status == "review_declined":
                prompt_text = session.text(
                    "workflow.review_declined",
                    feedback=outcome.reason,
                    stage_max_retries=session.runtime.experiment.reviewers.stage_max_retries,
                )
            else:
                prompt_text = session.text(
                    "workflow.tables_rejected",
                    feedback=outcome.reason,
                )

            action = await _ask_choice(
                prompt_text,
                [
                    (
                        "reevaluate_tables",
                        "reevaluate",
                        session.text("workflow.reevaluate_tables"),
                    ),
                    ("force_execution", "force", session.text("workflow.force_execution")),
                ],
                phase="code",
                gate=HumanGate.FORCE_EXECUTION_CONFIRMATION,
                approved_value="force",
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
                    session, llm, pm, solr, all_files, initial_hint=hint_msg
                )
                if table_status != "approved":
                    await cl.Message(content=session.text("workflow.cancelled")).send()
                    return "cancelled"
            else:
                table_status = await _run_table_gate(
                    session,
                    llm,
                    pm,
                    solr,
                    all_files,
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
                    return "cancelled"

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
        return

    async with WORKFLOW_LOCK:
        session = get_session()
        status = "failed"
        error = ""
        try:
            status = await _run_locked_workflow(question.strip())
        except WorkflowTimedOut as exc:
            status = "timed_out"
            error = f"{type(exc).__name__}: {exc}"
        except (asyncio.CancelledError, WorkflowCancelled):
            status = "cancelled"
            raise
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                _finalize_run(session, status, error)
            except Exception:
                logger.exception("Could not persist the Chainlit experiment record")
