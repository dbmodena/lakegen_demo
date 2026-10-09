"""The Chainlit workflow: one question through the engine, driven in the chat.

``run_lakegen_workflow`` takes the workflow lock and runs the question on
``lakegen.interfaces.chainlit.driver.ChainlitDriver``; ``_ask_choice`` and
``_ask_hint`` are the chat's two kinds of question, with the reconnect
fallback that lets a click on a button land after the websocket that sent
it is gone. The phases the engine calls are imported here by name so that a
test patching ``workflow.<name>`` patches what the engine calls.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from typing import Any

import chainlit as cl

from lakegen.discovery.orchestrated import (  # noqa: F401
    run_unified_orchestrated_discovery,
    select_from_prepared_context,
)
from lakegen.interfaces.chainlit.driver import ChainlitDriver
from lakegen.lakes import get_all_table_files  # noqa: F401
from lakegen.llm import get_llm  # noqa: F401

# The phases, resources and sinks the engine calls; pipeline_deps() binds
# them when a run starts, so a test patching a name here patches the engine.
from lakegen.observability.csv_log import save_experiment_log  # noqa: F401
from lakegen.observability.manifest import persist_manifest  # noqa: F401
from lakegen.observability.tracing import HumanGate, PhaseName, normalize_hint
from lakegen.orchestrated_context import prepare_discovery_context  # noqa: F401
from lakegen.phases import (  # noqa: F401
    phase1_generate_keywords,
    phase2_select_tables,
    phase3_generate_and_execute,
    phase4_synthesize,
    phase12_agent,
)
from lakegen.pipeline.context import MemoryScope, RunContext, SurfaceOptions
from lakegen.pipeline.deps import PipelineDeps
from lakegen.pipeline.engine import Pipeline
from lakegen.prompts import get_prompt_manager  # noqa: F401
from lakegen.retrieval.factory import get_solr  # noqa: F401
from lakegen.retrieval.run_log import log_retrieval_decision  # noqa: F401
from lakegen.semantic_code_judge import judge_semantic_code_result  # noqa: F401
from lakegen.ui.i18n import t
from lakegen.ui.state import (
    WorkflowCancelled,
    WorkflowTimedOut,
    get_runtime_settings,
    get_session,
)
from lakegen.ui.streaming import StepStreamBridge

WORKFLOW_LOCK = asyncio.Lock()
logger = logging.getLogger(__name__)


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


def pipeline_deps() -> PipelineDeps:
    """The phases, resources and sinks this surface runs the engine with.

    Built from this module's own names at the start of a run, so a test that
    replaces one here replaces what the engine calls.
    """
    return PipelineDeps(
        get_llm=get_llm,
        get_solr=get_solr,
        get_prompt_manager=get_prompt_manager,
        get_all_table_files=get_all_table_files,
        phase1_generate_keywords=phase1_generate_keywords,
        phase2_select_tables=phase2_select_tables,
        phase12_agent=phase12_agent,
        phase3_generate_and_execute=phase3_generate_and_execute,
        phase4_synthesize=phase4_synthesize,
        run_unified_orchestrated_discovery=run_unified_orchestrated_discovery,
        prepare_discovery_context=prepare_discovery_context,
        select_from_prepared_context=select_from_prepared_context,
        judge_semantic_code_result=judge_semantic_code_result,
        save_experiment_log=save_experiment_log,
        log_retrieval_decision=log_retrieval_decision,
        persist_manifest=persist_manifest,
    )


async def _run_locked_workflow(question: str) -> str:
    """One question on the engine, driven in the chat; the status the run ended with."""
    session = get_session()
    runtime = get_runtime_settings()
    session.runtime = runtime
    session.phase = "discovery"
    ctx = RunContext.from_runtime(
        runtime,
        SurfaceOptions(human_gated=True, memory_scope=MemoryScope.SESSION, answer_on_failure=True),
    )
    pipeline = Pipeline(ctx, pipeline_deps())
    driver = ChainlitDriver(
        pipeline, ctx, session, ask_choice=_ask_choice, ask_hint=_ask_hint, bridge_class=StepStreamBridge,
    )
    try:
        result = await driver.run(question)
    except RuntimeError as exc:
        # The engine refuses a lake without tables before it starts.
        session.execution_error = str(exc)
        await cl.Message(content=f"No CSV or Parquet files were found in `{runtime.csv_dir}`.").send()
        return "failed"
    session.manifest = dict(result.manifest)
    session.execution_error = result.error
    return result.status


async def run_lakegen_workflow(question: str) -> None:
    """One question in the chat, one at a time; the engine records how it ended."""
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
        try:
            await _run_locked_workflow(question.strip())
        except WorkflowTimedOut as exc:
            logger.info("Workflow timed out: %s", exc)
        finally:
            session.finalized = True
