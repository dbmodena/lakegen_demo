"""What a stage of the pipeline concluded, as a type the engine compares.

The phases still return what they always did (a ``Phase3Result``, the coder
sweep's variant dict, the discovery tuples); ``lakegen.pipeline.adapters``
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
    LLM = "llm"
    JEV = "jev"
    NONE = "none"
    """No selector ran: retrieval returned nothing to choose from."""


@dataclass(frozen=True)
class DiscoveryOutcome:
    status: DiscoveryStatus
    selected: list[str]
    candidates: list[str]
    """This round's retrieval list, for the last attempt's forced selection."""
    keywords: list[str]
    metadata: dict[str, Any]
    reasoning: str
    """The architect's reasoning, verbatim (the prompts ask for ``REJECT_KEYWORDS:``)."""
    trace: str
    tokens: int
    phase_tokens: tuple[int, int] = (0, 0)
    """(keywords, selection) tokens of the divided architecture; the unified agent's all count as keywords."""
    selector: SelectorKind = SelectorKind.AGENT
    keep_tables: list[str] = field(default_factory=list)
    """Tables the architect found useful when rejecting a round; they carry into the next."""
    skip_tables: list[str] = field(default_factory=list)
    """Tables the architect proved unusable; excluded from later rounds."""
    inspection_cache: dict[str, str] = field(default_factory=dict)
    """The unified agent's cached inspections of the kept tables."""
    prepared: Any = None
    """The orchestrated context the tables were chosen from, when one was prepared."""

    @property
    def rejected(self) -> bool:
        return self.status is DiscoveryStatus.REJECT_KEYWORDS


class CodeStatus(StrEnum):
    """How a coding attempt ended."""

    SUCCESS = "succeeded"
    TABLES_REJECTED = "tables_rejected"
    BLOCKED_BY_SELECTION = "blocked_by_selection"
    EXECUTION_ERROR = "failed"


@dataclass(frozen=True)
class CodeOutcome:
    status: CodeStatus
    generated: Any
    """The phase's own result object, for the evaluator, the judge and the telemetry."""
    code: str
    """The code to carry into the next attempt: the clean code, else the raw output."""
    tokens: int
    raw_result: str | None
    error: str
    """The message the next attempt or the final result carries."""
    raw_status: str = ""
    """The sweep's own status word (``completed``, ``rejected``, ...), for the stages it records."""
    rejection_details: dict[str, Any] = field(default_factory=dict)
    keep_tables: list[str] = field(default_factory=list)
    skip_tables: list[str] = field(default_factory=list)
    audit: dict[str, Any] | None = None
    """The coder's context audit (the semantic plan and brief), when it produced one."""
    execution_error: dict[str, Any] | None = None
    retryable: bool = False
    coder_runs: int = 0
    coder_review: dict[str, Any] | None = None
