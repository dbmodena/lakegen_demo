from __future__ import annotations

import asyncio
from collections.abc import Callable

import chainlit as cl


class StepStreamBridge:
    def __init__(self, step: cl.Step) -> None:
        self._step = step
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> "StepStreamBridge":
        self._task = asyncio.create_task(self._drain())
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.emit(None)
        if self._task is not None:
            await self._task

    def emit(self, delta: str | None) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, delta)

    async def _drain(self) -> None:
        while True:
            delta = await self._queue.get()
            if delta is None:
                return
            if delta:
                await self._step.stream_token(delta)


class CumulativeMarkdownEmitter:
    def __init__(self, emit: Callable[[str | None], None], title: str) -> None:
        self._emit = emit
        self._title = title
        self._current = ""
        self._header_sent = False

    def markdown(self, text: str) -> None:
        if not self._header_sent:
            self._emit(f"\n\n**{self._title}**\n")
            self._header_sent = True

        if text == self._current:
            return
        if text.startswith(self._current):
            delta = text[len(self._current):]
        else:
            delta = f"\n\n{text}"
        self._current = text
        self._emit(delta)
