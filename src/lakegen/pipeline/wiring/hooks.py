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
    """The streams and the stop check a driver attaches to the phases.

    Attributes:
        stream: Receives the text a phase streams (the architect's turns, the selector).
        stream_placeholder: Receives the keyword writer's or the coder's streamed output,
            as markdown.
        reasoning_placeholder: Receives the model's reasoning, as markdown.
        cancel_check: Raises when the person stopped the run; the phases call it between steps.
        tool_call: Receives each tool call of the table selector (``domain.types.ToolCall``),
            to show as a step.
    """

    stream: Callable[[str], None] | None = None
    stream_placeholder: Any = None
    reasoning_placeholder: Any = None
    cancel_check: Callable[[], None] | None = None
    tool_call: Callable[[Any], None] | None = None

    def agent_kwargs(self) -> dict[str, Any]:
        """For the phases that stream text: the unified agent and the table selectors."""
        kwargs: dict[str, Any] = {}
        if self.stream is not None:
            kwargs["stream_callback"] = self.stream
        if self.cancel_check is not None:
            kwargs["cancel_check"] = self.cancel_check
        return kwargs

    def selector_kwargs(self) -> dict[str, Any]:
        """For the divided table selection: the streams, the stop button and the tool calls' channel."""
        kwargs = self.agent_kwargs()
        if self.tool_call is not None:
            kwargs["tool_callback"] = self.tool_call
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
