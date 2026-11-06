"""Streaming a worker thread's output into the app's Chainlit steps."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import chainlit as cl
from chainlit.utils import utc_now

from lakegen.datastructures.types import ToolCall


class StepStreamBridge:
    """Carries a worker thread's output into a step: text is streamed into it, a tool call becomes a step under it.

    Both go through one queue, so the steps appear in the order the calls were made, among the streamed text.

    Args:
        step: The step the text streams into and the tool calls nest under.
    """

    def __init__(self, step: cl.Step) -> None:
        self._step = step
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[str | ToolCall | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> StepStreamBridge:
        self._task = asyncio.create_task(self._drain())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.emit(None)
        if self._task is not None:
            await self._task

    def emit(self, delta: str | ToolCall | None) -> None:
        """Queue text or a tool call from any thread; ``None`` ends the stream."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, delta)

    async def _drain(self) -> None:
        while True:
            delta = await self._queue.get()
            if delta is None:
                return
            if isinstance(delta, ToolCall):
                await self._tool_step(delta)
            elif delta:
                await self._step.stream_token(delta)

    async def _tool_step(self, call: ToolCall) -> None:
        """The call as a collapsed step of its own: the title, and the result in the step's body."""
        step = cl.Step(name=call.title, type="tool", parent_id=self._step.id, show_input=False)
        step.start = step.end = utc_now()
        step.output = call.output
        step.is_error = call.failed
        await step.send()


class CumulativeMarkdownEmitter:
    """A stream placeholder that is handed the whole text so far and emits only what is new.

    The first call emits a bold header with the title. A text that extends the
    previous one emits the added part; any other text starts a new paragraph.

    Args:
        emit: Where the text goes, such as ``StepStreamBridge.emit``.
        title: The header shown above the text.
    """

    def __init__(self, emit: Callable[[str | None], None], title: str) -> None:
        self._emit = emit
        self._title = title
        self._current = ""
        self._header_sent = False

    def markdown(self, text: str) -> None:
        """Emit the part of ``text`` not emitted yet, after the header on the first call."""
        if not self._header_sent:
            self._emit(f"\n\n**{self._title}**\n")
            self._header_sent = True

        if text == self._current:
            return
        if text.startswith(self._current):
            delta = text[len(self._current) :]
        else:
            delta = f"\n\n{text}"
        self._current = text
        self._emit(delta)
