"""What a stage of the pipeline concluded, as a type the engine compares.

The phases still return what they always did (a ``CoderResult``, the coder
sweep's variant dict, the discovery tuples); ``lakegen.pipeline.wiring.adapters``
turns each into one of these once, at the boundary, so the engine compares
statuses and never sniffs a string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DiscoveryStatus(StrEnum):
    """How a discovery round ended."""

    SELECTED = "selected"
    REJECT_KEYWORDS = "keywords_rejected"
    """The architect asked for new keywords (its reasoning starts with ``REJECT_KEYWORDS:``)."""


class SelectorKind(StrEnum):
    """Who chose the tables of a discovery round."""

    AGENT = "agent"
    """The unified agent, which searches and selects in one conversation."""
    MEMGPT = "memgpt"
    """The divided architecture's MemGPT selector, an agent with a system-kept working memory."""
    NONE = "none"
    """No selector ran: retrieval returned nothing to choose from."""
    STORED = "stored"
    """A replay: the selection of an earlier run, taken as it was recorded."""


@dataclass(frozen=True)
class DiscoveryOutcome:
    """What a discovery round concluded: the selected tables, or a request for new keywords.

    Attributes:
        status: Whether the round selected tables or rejected its keywords.
        selected: The selected table ids.
        candidates: This round's retrieval list, for the last attempt's forced selection.
        keywords: The keywords the round searched with.
        metadata: The catalogue metadata of the tables, by table id.
        reasoning: The architect's reasoning, verbatim (the prompts ask for
            ``REJECT_KEYWORDS:``).
        trace: The round's trace; the divided architecture's composes the keyword and
            selection rounds.
        tokens: The tokens the round spent.
        phase_tokens: (keywords, selection) tokens of the divided architecture; the unified
            agent's all count as keywords.
        selector: Who chose the tables.
        keep_tables: Tables the architect found useful when rejecting a round; they carry
            into the next.
        skip_tables: Tables the architect proved unusable; excluded from later rounds.
        selection_trace: The selector's own trace, before the divided architecture's
            composition with the keyword round.
    """

    status: DiscoveryStatus
    selected: list[str]
    candidates: list[str]
    keywords: list[str]
    metadata: dict[str, Any]
    reasoning: str
    trace: str
    tokens: int
    phase_tokens: tuple[int, int] = (0, 0)
    selector: SelectorKind = SelectorKind.AGENT
    keep_tables: list[str] = field(default_factory=list)
    skip_tables: list[str] = field(default_factory=list)
    selection_trace: str = ""

    @property
    def rejected(self) -> bool:
        """Whether the round asked for new keywords."""
        return self.status is DiscoveryStatus.REJECT_KEYWORDS


class CodeStatus(StrEnum):
    """How a coding attempt ended."""

    SUCCESS = "succeeded"
    TABLES_REJECTED = "tables_rejected"
    BLOCKED_BY_SELECTION = "blocked_by_selection"
    EXECUTION_ERROR = "failed"


@dataclass(frozen=True)
class CodeOutcome:
    """What a coding attempt concluded, from a ``CoderResult`` or a coder-sweep variant.

    Attributes:
        status: How the attempt ended.
        generated: The phase's own result object, for the evaluator, the judge and the
            telemetry.
        code: The code to carry into the next attempt: the clean code, else the raw output.
        tokens: The tokens the attempt spent.
        raw_result: The executed code's raw result; None when it produced none.
        error: The message the next attempt or the final result carries.
        raw_status: The sweep's own status word (``completed``, ``rejected``, ...), for the
            stages it records.
        rejection_details: Why the coder rejected the tables, when it did.
        keep_tables: Tables the coder kept when rejecting the selection.
        skip_tables: Tables the coder found unusable when rejecting the selection.
        audit: The coder's context audit (the semantic plan and brief), when it produced one.
        execution_error: The structured execution error, when the attempt failed.
        retryable: Whether the execution error says the attempt can be retried.
        coder_runs: How many coder runs the attempt took.
        coder_review: The coder's review of its result, when it made one.
    """

    status: CodeStatus
    generated: Any
    code: str
    tokens: int
    raw_result: str | None
    error: str
    raw_status: str = ""
    rejection_details: dict[str, Any] = field(default_factory=dict)
    keep_tables: list[str] = field(default_factory=list)
    skip_tables: list[str] = field(default_factory=list)
    audit: dict[str, Any] | None = None
    execution_error: dict[str, Any] | None = None
    retryable: bool = False
    coder_runs: int = 0
    coder_review: dict[str, Any] | None = None
