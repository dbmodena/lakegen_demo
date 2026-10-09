"""The Chainlit driver: the engine's steps shown as Chainlit steps, its gates as buttons.

The engine runs on one worker thread of its own (``_EngineThread``), so
the context its phases set up (the retrieval capture) stays whole from the
first step to the last; the driver, on the event loop, opens a ``cl.Step``
with a stream bridge for each stage, fills the session the summaries read,
sends the messages the chat always showed, and answers each request with
the same buttons and prompts. Stop reaches a running phase through the
session's cancel check; at a gate the driver throws the cancellation into
the engine so the run is recorded. The engine's logic underneath is the
batch service's.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from typing import Any

import chainlit as cl

from lakegen.domain.errors import WorkflowCancelled, WorkflowTimedOut
from lakegen.domain.results import QueryResult
from lakegen.observability.tracing import HumanGate
from lakegen.pipeline.context import RunContext
from lakegen.pipeline.engine import Pipeline
from lakegen.pipeline.events import (
    Answer,
    Approve,
    CoderRejected,
    Force,
    Rediscover,
    Regenerate,
    Reject,
    ReviewKeywords,
    ReviewTables,
    RunStarted,
    StageFinished,
    StageStarted,
)
from lakegen.pipeline.hooks import StageHooks
from lakegen.ui.i18n import t
from lakegen.ui.sections import (
    build_phase1_summary,
    build_phase2_summary,
    build_phase3_summary,
    build_phase4_summary,
    format_retrieval_keywords,
)
from lakegen.ui.state import LakeGenSession
from lakegen.ui.streaming import CumulativeMarkdownEmitter, StepStreamBridge

Asker = Callable[..., Any]

UNIFIED_STEP = "Phase 1 & 2 (Unified Architect & Search)"
NO_CODE = "# No executable code captured."
NO_OUTPUT = "No output generated."


def _fenced_text(content: str) -> str:
    fence = "```"
    while fence in content:
        fence += "`"
    return f"{fence}text\n{content}\n{fence}"


class _EngineThread:
    """The engine's generator on one thread; the loop sends answers in and awaits what comes out."""

    def __init__(self, steps, loop: asyncio.AbstractEventLoop) -> None:
        self.steps = steps
        self.loop = loop
        self.inbox: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="lakegen-engine", daemon=True)
        self.finished = False

    def start(self) -> None:
        self.thread.start()

    async def send(self, answer: Answer | None) -> tuple[str, Any]:
        """Run the engine to its next yield: ``("event", event)`` or ``("done", result)``."""
        future = self.loop.create_future()
        self.inbox.put(("send", answer, future))
        return await future

    async def throw(self, exc: BaseException) -> None:
        """Raise ``exc`` inside the engine at its current yield; it records the run and ends."""
        future = self.loop.create_future()
        self.inbox.put(("throw", exc, future))
        try:
            await future
        except BaseException:  # noqa: BLE001  (the engine re-raises what it was thrown; the record is written)
            pass

    def close(self) -> None:
        if not self.finished:
            self.inbox.put(("close", None, None))

    def _run(self) -> None:
        while True:
            command, payload, future = self.inbox.get()
            try:
                if command == "close":
                    self.steps.close()
                    return
                event = self.steps.send(payload) if command == "send" else self.steps.throw(payload)
                outcome: tuple[str, Any] = ("event", event)
            except StopIteration as stop:
                outcome = ("done", stop.value)
            except BaseException as exc:  # noqa: BLE001  (the exception travels to the loop)
                outcome = ("error", exc)
            self.finished = outcome[0] != "event"
            if future is not None:
                self.loop.call_soon_threadsafe(self._resolve, future, outcome)
            if self.finished:
                return

    @staticmethod
    def _resolve(future: asyncio.Future, outcome: tuple[str, Any]) -> None:
        if future.cancelled():
            return
        kind, payload = outcome
        if kind == "error":
            future.set_exception(payload)
        else:
            future.set_result(outcome)


class _OpenStep:
    """A Chainlit step with its stream bridge, entered now and left on request."""

    def __init__(self, step: Any, bridge: Any = None) -> None:
        self.step = step
        self.bridge = bridge

    @classmethod
    async def enter(cls, step: Any, *, bridge: bool, bridge_class: type) -> _OpenStep:
        await step.__aenter__()
        opened = cls(step, bridge_class(step) if bridge else None)
        if opened.bridge is not None:
            await opened.bridge.__aenter__()
        return opened

    async def leave(self) -> None:
        if self.bridge is not None:
            await self.bridge.__aexit__(None, None, None)
            self.bridge = None
        await self.step.__aexit__(None, None, None)

    def emit(self, delta: str) -> None:
        if self.bridge is not None:
            self.bridge.emit(delta)


class ChainlitDriver:
    def __init__(
        self,
        pipeline: Pipeline,
        ctx: RunContext,
        session: LakeGenSession,
        *,
        ask_choice: Asker,
        ask_hint: Asker,
        bridge_class: type = StepStreamBridge,
    ) -> None:
        self.pipeline = pipeline
        self.ctx = ctx
        self.session = session
        self.ask_choice = ask_choice
        self.ask_hint = ask_hint
        self.bridge_class = bridge_class
        self.unified = ctx.experiment.use_unified_agent
        self.hooks = StageHooks(cancel_check=session.check_cancelled)
        self.discovery: _OpenStep | None = None
        self.coding: _OpenStep | None = None
        self.attempt: _OpenStep | None = None
        self.answer_step: _OpenStep | None = None
        self.code_attempts: list[dict[str, Any]] = []
        self.attempt_blocks: list[str] = []
        self.user_hint = ""
        """What the person asked of the round in progress; the summaries show it as the approval hint."""
        self.pending_hint = ""
        self.recalculated = False
        self.reselecting = False
        """The next round re-selects with the same keywords: it opens at phase 2."""

    # --- the loop -------------------------------------------------------------

    async def run(self, question: str) -> QueryResult:
        steps = self.pipeline.steps(
            question, interventions=self.session.intervention_recorder, hooks=self.hooks,
        )
        engine = _EngineThread(steps, asyncio.get_running_loop())
        engine.start()
        answer: Answer | None = None
        try:
            while True:
                kind, payload = await engine.send(answer)
                if kind == "done":
                    return payload
                try:
                    answer = await self.handle(payload)
                except (WorkflowTimedOut, WorkflowCancelled, asyncio.CancelledError) as exc:
                    # The engine waits at a gate: it records the run with the
                    # cancellation or the timeout and ends on its own thread.
                    thrown = exc if isinstance(exc, WorkflowTimedOut | WorkflowCancelled) else WorkflowCancelled("Workflow stopped by user.")
                    await engine.throw(thrown)
                    raise
        finally:
            engine.close()

    async def handle(self, event) -> Answer | None:
        if isinstance(event, RunStarted):
            return None
        if isinstance(event, StageStarted):
            await self.stage_started(event)
        elif isinstance(event, StageFinished):
            await self.stage_finished(event)
        elif isinstance(event, ReviewKeywords):
            return await self.review_keywords(event)
        elif isinstance(event, ReviewTables):
            return await self.review_tables(event)
        elif isinstance(event, CoderRejected):
            return await self.coder_rejected(event)
        return None

    # --- steps ----------------------------------------------------------------

    async def _open(self, name: str, *, type: str, bridge: bool = True, **kwargs) -> _OpenStep:
        step = cl.Step(name=name, type=type, default_open=True, **kwargs)
        return await _OpenStep.enter(step, bridge=bridge, bridge_class=self.bridge_class)

    def _boxes(self, opened: _OpenStep, stream_title: str, reasoning_title: str) -> None:
        self.hooks.stream = None
        self.hooks.stream_placeholder = CumulativeMarkdownEmitter(opened.emit, stream_title)
        self.hooks.reasoning_placeholder = CumulativeMarkdownEmitter(opened.emit, reasoning_title)

    async def stage_started(self, event: StageStarted) -> None:
        session = self.session
        if event.stage == "discovery":
            self.user_hint, self.pending_hint = self.pending_hint, ""
            self.recalculated = False
            if self.unified:
                self.discovery = await self._open(UNIFIED_STEP, type="run", auto_collapse=True)
                self.hooks.stream = self.discovery.emit
                self.hooks.stream_placeholder = self.hooks.reasoning_placeholder = None
            elif self.reselecting:
                self.reselecting = False
                await self._open_selection_step()
            else:
                await self._open_keyword_step()
        elif event.stage == "coding":
            self.coding = await self._open(session.text("phase3.step"), type="run", bridge=False)
            self.code_attempts, self.attempt_blocks = [], []
        elif event.stage == "code":
            await self._open_attempt(event.attempt)
        elif event.stage == "answer":
            if self.coding is not None:
                await self.coding.leave()
                self.coding = None
            self.answer_step = await self._open(session.text("phase4.step"), type="llm", bridge=False)

    async def _open_selection_step(self) -> None:
        self.discovery = await self._open(self.session.text("phase2.step"), type="run", auto_collapse=True)
        self.hooks.stream = self.discovery.emit
        self.hooks.stream_placeholder = self.hooks.reasoning_placeholder = None

    async def _open_keyword_step(self) -> None:
        self.discovery = await self._open(self.session.text("phase1.step"), type="llm")
        self._boxes(self.discovery, self.session.text("phase1.keyword_stream"), self.session.text("phase1.model_reasoning"))

    async def _open_attempt(self, number: int) -> None:
        assert self.coding is not None
        self.attempt = _OpenStep(self.coding.step, self.bridge_class(self.coding.step))
        await self.attempt.bridge.__aenter__()
        self.attempt.emit(f"\n\n## {self.session.text('summary.attempt')} {number}\n")
        self._boxes(self.attempt, self.session.text("phase3.code_stream"), self.session.text("phase3.model_reasoning"))

    async def stage_finished(self, event: StageFinished) -> None:
        if event.stage == "discovery":
            await self.discovery_finished(event)
        elif event.stage == "code":
            await self.code_finished(event)
        elif event.stage == "answer":
            await self.answer_finished(event)

    # --- discovery ------------------------------------------------------------

    async def discovery_finished(self, event: StageFinished) -> None:
        session, d = self.session, event.details
        session.tables = list(d["selected"])
        session.keywords = list(d["keywords"])
        session.candidates = list(d["selected"] if self.unified else d["candidates"])
        session.solr_metadata_map = dict(d["metadata"])
        session.architect_reasoning = d["reasoning"]
        session.full_trace = d["trace"] if self.unified else d["selection_trace"]
        opened = self.discovery
        assert opened is not None
        if self.unified:
            session.tokens["p1"] += d["tokens"]
            opened.step.output = (
                f"**Keywords used:** {format_retrieval_keywords(session, session.keywords)}\n\n"
                f"**Tables selected:** " + ", ".join(f"`{table}`" for table in session.tables) + "\n\n"
                f"**Reasoning:**\n{session.architect_reasoning}\n\n"
                f"- Tokens: `{d['tokens']}`\n\n"
                f"***Full agent activity log:***\n\n"
                f"{session.full_trace}\n\n"
            )
            await opened.leave()
            return
        session.tokens["p2"] += d["phase_tokens"][1]
        if event.detail == "keywords_rejected":
            reason = d["reasoning"].replace("REJECT_KEYWORDS:", "", 1).replace("REJECT_KEYWORDS", "", 1).strip()
            session.fallback_reason = reason or "The candidate tables did not match the generated keywords."
            opened.step.output = session.text("phase2.keywords_rejected", reason=session.fallback_reason)
            await opened.leave()
            await cl.Message(
                content=session.text("phase2.architect_rejected", feedback=session.fallback_reason)
                + "\n\n🔄 **Auto-correcting:** Sending feedback to Phase 1 for new keywords..."
            ).send()
            opened.step.default_open = False
            await opened.step.update()
            return
        opened.step.output = build_phase2_summary(session, self.user_hint)
        await opened.leave()

    async def review_keywords(self, event: ReviewKeywords) -> Answer:
        session = self.session
        label = (
            session.text("phase1.recalculation") if self.recalculated
            else session.text("phase1.fallback_regeneration") if event.hint
            else session.text("phase1.initial_generation")
        )
        session.keywords = list(event.keywords)
        session.raw_keywords = event.raw
        session.tokens["p1"] += event.tokens
        session.record_phase1_run(label, event.hint, list(event.keywords), event.raw, event.tokens, event.reasoning)
        opened = self.discovery
        assert opened is not None
        opened.step.output = (
            f"{t('summary.keywords').title()}: {format_retrieval_keywords(session, event.keywords)}\n\n"
            f"{t('summary.tokens').title()}: `{event.tokens}`"
        )
        await opened.leave()
        session.check_cancelled()
        action = await self.ask_choice(
            session.text("phase1.review_keywords", keywords=format_retrieval_keywords(session, session.keywords)),
            [
                ("approve_keywords", "approve", session.text("phase1.approve")),
                ("recalculate_keywords", "recalculate", session.text("phase1.recalculate")),
            ],
            phase="discovery", gate=HumanGate.KEYWORD_APPROVAL, approved_value="approve", remove_after_answer=True,
        )
        if action == "approve":
            opened.step.output = build_phase1_summary(session, event.hint)
            await opened.step.update()
            await self._open_selection_step()
            return Approve()
        session.check_cancelled()
        hint = await self.ask_hint(
            session.text("phase1.change_hint"), phase="discovery", gate=HumanGate.KEYWORD_HINT, remove_after_answer=True,
        )
        self.recalculated = True
        await self._open_keyword_step()
        return Regenerate(hint)

    async def review_tables(self, event: ReviewTables) -> Answer:
        session, opened = self.session, self.discovery
        assert opened is not None
        if self.unified:
            prompt = session.text(
                "phase2.review_tables",
                tables=f"**Keywords:** {format_retrieval_keywords(session, event.keywords)}\n\n**Tables:**\n"
                + "\n".join(f"- `{table}`" for table in event.selected) + f"\n\n**Reasoning:**\n{event.reasoning}",
            )
            choices = [("approve_selection", "approve", "Approve Selection"), ("recalculate_selection", "recalculate", "Recalculate (change hint)")]
        else:
            prompt = session.text(
                "phase2.review_tables",
                tables="\n".join(f"- `{table}`" for table in event.selected) + f"\n\n**Reasoning:**\n{event.reasoning}",
            )
            choices = [("approve_tables", "approve", session.text("phase2.approve")), ("recalculate_tables", "recalculate", session.text("phase2.recalculate"))]
        action = await self.ask_choice(
            prompt, choices, phase="discovery", gate=HumanGate.DATASET_APPROVAL, approved_value="approve", remove_after_answer=True,
        )
        opened.step.default_open = False
        if action == "approve":
            if not self.unified:
                opened.step.output = build_phase2_summary(session, self.user_hint)
            await opened.step.update()
            return Approve()
        await opened.step.update()
        session.check_cancelled()
        if self.unified:
            prompt = "What should the agent change? (e.g., use different keywords, or look for different tables)"
            if event.architect_feedback:
                prompt += (
                    "\n\nThe architect's own feedback from this round is passed on "
                    "automatically; add anything else it should know."
                )
        else:
            prompt = session.text("phase2.change_hint")
        hint = await self.ask_hint(prompt, phase="discovery", gate=HumanGate.DATASET_HINT, remove_after_answer=True)
        self.pending_hint = hint
        # The chat re-selects with the same keywords; the CLI writes new ones.
        self.reselecting = not self.unified
        return Reject(hint, regenerate_keywords=self.unified)

    # --- coding ---------------------------------------------------------------

    async def code_finished(self, event: StageFinished) -> None:
        session, d, coding = self.session, event.details, self.coding
        assert coding is not None
        if self.attempt is not None:
            await self.attempt.bridge.__aexit__(None, None, None)
            self.attempt = None
        status = {"succeeded": "success", "tables_rejected": "rejected tables"}.get(event.detail, "error")
        attempt = {
            "attempt": event.attempt,
            "correction_feedback": d["error_msg"],
            "error": d["error"] if event.detail != "succeeded" else None,
            "raw_response": d["code_raw"],
            "clean_code": d["clean_code"],
            "tokens": d["tokens"],
            "status": status,
        }
        session.tokens["p3"] += d["tokens"]
        session.final_code = d["clean_code"]
        session.raw_result = d["raw_result"]
        session.execution_error = str(d["error"] or "")
        self.code_attempts.append(attempt)
        shown = d["code_raw"] if status == "rejected tables" else (d["raw_result"] or session.text("phase3.success")) if status == "success" else event.message
        self.attempt_blocks.append(self._attempt_block(attempt, shown))
        coding.step.output = "\n\n".join(self.attempt_blocks)
        if status == "rejected tables":
            await coding.step.update()
            await cl.Message(content=build_phase3_summary(session, self.code_attempts)).send()
            await coding.leave()
            self.coding = None
        elif status == "error":
            await coding.step.update()
            if not event.more:
                await coding.leave()
                self.coding = None
        else:
            await coding.leave()
            self.coding = None

    def _attempt_block(self, attempt: dict[str, Any], output: str | None) -> str:
        status = str(attempt.get("status", "generated"))
        label = t(f"status.{status.lower().replace(' ', '_')}", default=status)
        return (
            f"### {self.session.text('summary.attempt')} {attempt.get('attempt')} - {label}\n\n"
            f"{_fenced_text(output or self.session.text('phase3.success'))}\n\n"
            f"- {self.session.text('summary.tokens')}: `{attempt.get('tokens', 0)}`"
        )

    async def coder_rejected(self, event: CoderRejected) -> Answer:
        session = self.session
        session.fallback_reason = event.reason
        action = await self.ask_choice(
            session.text("workflow.tables_rejected", feedback=event.reason),
            [
                ("reevaluate_tables", "reevaluate", session.text("workflow.reevaluate_tables")),
                ("force_execution", "force", session.text("workflow.force_execution")),
            ],
            phase="code", gate=HumanGate.FORCE_EXECUTION_CONFIRMATION, approved_value="force",
        )
        if action == "force":
            self.coding = await self._open(session.text("phase3.step"), type="run", bridge=False)
            self.code_attempts, self.attempt_blocks = [], []
            return Force()
        return Rediscover()

    # --- the answer -----------------------------------------------------------

    async def answer_finished(self, event: StageFinished) -> None:
        session, opened = self.session, self.answer_step
        assert opened is not None
        answer = event.details["answer"]
        session.final_answer = answer
        session.tokens["p4"] = event.details["tokens"]
        opened.step.output = answer
        await opened.leave()
        self.answer_step = None
        raw_result = session.raw_result
        elements = [
            cl.Text(name="generated_code.py", content=session.final_code or NO_CODE, language="python", display="side"),
            cl.Text(name="execution_output.txt", content=str(raw_result) if raw_result else NO_OUTPUT, language="text", display="side"),
        ]
        await cl.Message(
            content=(
                f"### {session.text('result.final')}\n{answer}\n\n"
                f"{build_phase3_summary(session, self.code_attempts)}\n\n"
                f"{build_phase4_summary(session, answer)}"
            ),
            elements=elements,
        ).send()
