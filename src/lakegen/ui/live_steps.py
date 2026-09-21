"""Bridges background threads (e.g. `cl.make_async`'s worker thread, or a
`ThreadPoolExecutor` running inside it) to Chainlit's async `cl.Step` API.

Generalizes `StepStreamBridge`'s `call_soon_threadsafe` pattern (streaming
tokens into one fixed step) to opening, updating, and closing multiple
independently-lifecycled NAMED child steps from any thread -- the primitive
`retrieval_observer` wiring uses to show exactly what was sent to the
retriever on each search, as its own step.
"""
from __future__ import annotations

import asyncio
import contextvars
from typing import Callable

import chainlit as cl

from lakegen.retrieval.models import RetrievalRun


class ThreadSafeStepEmitter:
    """One instance per parent operation (e.g. per phase12_agent call).
    Every public method is safe to call from any thread; each call schedules
    its actual `cl.Step` work back onto the event loop this instance was
    constructed on, via `call_soon_threadsafe` -- the same technique
    `StepStreamBridge.emit` already uses.

    `call_soon_threadsafe(callback, *args)` with no explicit `context=`
    captures whatever contextvars.Context is active on the CALLING thread at
    the moment it's invoked -- not the context of the loop thread this
    instance was constructed on. A background/executor thread's context is
    blank (contextvars are thread-local; they don't cross OS threads), so a
    task created from inside that blank context -- like the cl.Step() this
    class opens -- can't find Chainlit's session context and raises
    ChainlitContextException. StepStreamBridge doesn't hit this because its
    one task is created up front from the correct context and only ever
    reads off a plain, context-independent asyncio.Queue afterwards; this
    class creates a NEW task per step, at arbitrary later times, from
    arbitrary threads, so each one needs the right context pinned explicitly
    -- captured once here, on the correct thread, and passed to every
    call_soon_threadsafe call below."""

    def __init__(self, parent_step: "cl.Step | None" = None) -> None:
        self._loop = asyncio.get_running_loop()
        self._parent_step = parent_step
        self._context = contextvars.copy_context()
        self._pending: dict[str, "asyncio.Queue"] = {}
        self._tasks: dict[str, "asyncio.Task"] = {}

    def instant(self, step_id: str, name: str, output: str, step_type: str = "tool") -> None:
        """Thread-safe: create, populate, and close a step in one shot --
        for events whose full result is already known when reported (e.g. a
        retrieval call, whose observer only fires after it completes)."""
        self._loop.call_soon_threadsafe(
            self._start_instant, step_id, name, output, step_type, context=self._context
        )

    def start(self, step_id: str, name: str, step_type: str = "tool") -> None:
        """Thread-safe: open a step whose result isn't known yet. Must be
        paired with a later `finish(step_id, ...)` call."""
        self._loop.call_soon_threadsafe(
            self._start_pending, step_id, name, step_type, context=self._context
        )

    def finish(self, step_id: str, output: str = "", *, is_error: bool = False) -> None:
        """Thread-safe: close a step previously opened with `start`."""
        self._loop.call_soon_threadsafe(self._finish_pending, step_id, output, is_error)

    # --- event-loop-thread-only from here down ---

    def _start_instant(self, step_id: str, name: str, output: str, step_type: str) -> None:
        self._tasks[step_id] = asyncio.create_task(self._run_instant(name, output, step_type))

    async def _run_instant(self, name: str, output: str, step_type: str) -> None:
        step = cl.Step(name=name, type=step_type)
        if self._parent_step is not None:
            step.parent_id = self._parent_step.id
        async with step:
            step.output = output

    def _start_pending(self, step_id: str, name: str, step_type: str) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        self._pending[step_id] = queue
        self._tasks[step_id] = asyncio.create_task(self._run_pending(name, step_type, queue))

    def _finish_pending(self, step_id: str, output: str, is_error: bool) -> None:
        queue = self._pending.get(step_id)
        if queue is not None:
            queue.put_nowait((output, is_error))

    async def _run_pending(self, name: str, step_type: str, queue: "asyncio.Queue") -> None:
        step = cl.Step(name=name, type=step_type)
        if self._parent_step is not None:
            step.parent_id = self._parent_step.id
        async with step:
            output, is_error = await queue.get()
            step.output = output
            step.is_error = is_error

    async def aclose(self) -> None:
        """Wait for every step this instance started to actually finish --
        call once at the end of the parent operation so nothing is left
        dangling if a caller forgot a matching `finish`.

        `start`/`instant`/`finish` only SCHEDULE their work via
        call_soon_threadsafe; a caller on another thread (e.g. joined just
        before this call) is guaranteed its calls are enqueued, but not yet
        run. Yielding to the loop repeatedly first drains that queue before
        this method inspects `_pending`/`_tasks`, so a start()/instant()
        issued immediately beforehand is never missed.
        """
        for _ in range(50):
            await asyncio.sleep(0)
        for step_id, queue in list(self._pending.items()):
            if queue.empty():
                queue.put_nowait(("(no result reported)", False))
        for task in list(self._tasks.values()):
            await task
        self._pending.clear()
        self._tasks.clear()


def _format_retrieval_run(run: RetrievalRun) -> str:
    """The literal parameters sent to the retriever for one search -- not
    the one-line human summary `_emit_notice` already prints, the actual
    call: keywords, q_op, entities, the real configured top_k vs. the wider
    fetch width this specific call used, hit count, and latency."""
    lines = [
        f"mode: {run.mode}",
        f"keywords: {run.keywords!r}",
    ]
    if run.entities:
        lines.append(f"entities: {run.entities!r}")
    if run.q_op:
        fallback_note = " (OR fallback after an empty AND search)" if run.q_op == "OR" else ""
        lines.append(f"q_op: {run.q_op}{fallback_note}")
    lines.append(f"configured top_k: {run.configured_top_k}  ·  this call's fetch width: {run.top_k}")
    lines.append(f"status: {run.status}" + (f"  ({run.error})" if run.error else ""))
    lines.append(f"hits returned: {len(run.hits)}")
    if run.duration_seconds is not None:
        lines.append(f"latency: {run.duration_seconds:.2f}s")
    return "\n".join(lines)


def retrieval_observer_for(emitter: ThreadSafeStepEmitter) -> "Callable[[RetrievalRun], None]":
    """Build a retrieval_observer (the Callable[[RetrievalRun], None]
    protocol phase12_agent already accepts) that renders each search as its
    own step via `emitter`, instead of the JSONL-only sink the batch path
    uses (`make_retrieval_run_observer`). Safe to call from any thread --
    the observer fires from inside _search, which runs off the event loop
    thread under cl.make_async."""
    counter = {"n": 0}

    def _observe(run: RetrievalRun) -> None:
        counter["n"] += 1
        name = f"Solr query #{counter['n']}"
        emitter.instant(f"retrieval-{counter['n']}", name, _format_retrieval_run(run))

    return _observe
