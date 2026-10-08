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
