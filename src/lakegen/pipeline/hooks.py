"""What a surface attaches to the phases while they run: streams and the stop button.

The Chainlit app streams a phase's output into its step and stops a phase
when the person presses Stop; the service and the CLI attach nothing. A
driver fills the hooks before each stage runs; the stages pass only what
is set, so a run without hooks calls the phases exactly as before.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class StageHooks:
    stream: Callable[[str], None] | None = None
    """Receives the text a phase streams (the architect's turns, the selector)."""
    stream_placeholder: Any = None
    """Receives the keyword writer's or the coder's streamed output, as markdown."""
    reasoning_placeholder: Any = None
    """Receives the model's reasoning, as markdown."""
    cancel_check: Callable[[], None] | None = None
    """Raises when the person stopped the run; the phases call it between steps."""

    def agent_kwargs(self) -> dict[str, Any]:
        """For the phases that stream text: the unified agent, the selector, the judge."""
        kwargs: dict[str, Any] = {}
        if self.stream is not None:
            kwargs["stream_callback"] = self.stream
        if self.cancel_check is not None:
            kwargs["cancel_check"] = self.cancel_check
        return kwargs

    def placeholder_kwargs(self, *, cancel: bool) -> dict[str, Any]:
        """For the phases that stream into placeholders: the keyword writer, the coder."""
        kwargs: dict[str, Any] = {}
        if self.stream_placeholder is not None:
            kwargs["stream_placeholder"] = self.stream_placeholder
        if self.reasoning_placeholder is not None:
            kwargs["reasoning_placeholder"] = self.reasoning_placeholder
        if cancel and self.cancel_check is not None:
            kwargs["cancel_check"] = self.cancel_check
        return kwargs
