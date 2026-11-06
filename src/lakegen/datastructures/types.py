"""Type aliases and the tool-call record the phases and the interfaces share."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

SolrMetadata = dict[str, dict[str, Any]]
Phase2SelectionResult = tuple[list[str], list[str], SolrMetadata, str, str, int]
StreamCallback = Callable[[str], None]


@dataclass(frozen=True)
class ToolCall:
    """One tool call of an agent as a person reads it: the webapp shows it as a step of its own."""

    name: str
    """The tool (``inspect``)."""
    title: str
    """What the call did, in one line ("Inspect #1 Vivienda libre, #2 ...")."""
    output: str
    """What came back, as markdown."""
    summary: str = ""
    """The result in one line, for a terminal."""
    failed: bool = False
    """The call was refused or broke."""


ToolCallback = Callable[[ToolCall], None]
