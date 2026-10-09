"""What the engine tells its driver, and what it asks.

``Pipeline.steps`` is a generator. It yields signals (a stage started or
finished) that a driver may show, and, in a human-gated run, requests that
the driver must answer by sending one of the answers below back in. An
autonomous run never asks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --- signals -----------------------------------------------------------------

@dataclass(frozen=True)
class RunStarted:
    tables: int
    """How many tables the lake holds."""


@dataclass(frozen=True)
class StageStarted:
    stage: str
    """``discovery`` (a round), ``coding`` (a round), ``code`` (an attempt) or ``answer``."""
    attempt: int = 1


@dataclass(frozen=True)
class StageFinished:
    stage: str
    attempt: int = 1
    detail: str = ""
    """Discovery: ``selected``, ``keywords_rejected`` or ``forced_selection``; code: the attempt's status."""
    message: str = ""
    """Discovery: the architect's reason; code: the result or the error."""
    more: bool = False
    """Another attempt of the same stage follows."""
    details: dict[str, Any] = field(default_factory=dict)
    """What the stage produced, for a driver that shows more than the message:
    discovery: keywords, selected, candidates, reasoning, trace, selection_trace, tokens, hint;
    code: code_raw, clean_code, tokens, error, error_msg, raw_result; answer: answer, tokens."""


# --- requests and their answers ----------------------------------------------

@dataclass(frozen=True)
class ReviewKeywords:
    """The divided architecture wrote keywords; approve them or ask for new ones."""

    keywords: list[str]
    tokens: int
    hint: str = ""
    """What the keyword writer was told for this round."""
    raw: str = ""
    """The writer's raw output."""
    reasoning: str = ""


@dataclass(frozen=True)
class ReviewTables:
    """Discovery selected tables; approve them, ask for another round, or give up."""

    keywords: list[str]
    selected: list[str]
    reasoning: str
    tokens: int
    after_rejection: bool = False
    """The round followed a coder rejection: the driver may only approve or abort."""
    architect_feedback: str = ""
    """The architect's own feedback on this round, when it rejected it; a hint the person adds is merged with it."""


@dataclass(frozen=True)
class CoderRejected:
    """The coder rejected the tables; force the code anyway, or discover again."""

    reason: str


@dataclass(frozen=True)
class Approve:
    pass


@dataclass(frozen=True)
class Regenerate:
    """New keywords, with a hint for the keyword writer."""

    hint: str = ""


@dataclass(frozen=True)
class Reject:
    """Another discovery round, with a hint for the architect."""

    hint: str = ""
    regenerate_keywords: bool = True
    """Divided architecture: write new keywords (the CLI's way) or re-select with the same ones (the chat's)."""


@dataclass(frozen=True)
class Force:
    pass


@dataclass(frozen=True)
class Rediscover:
    pass


@dataclass(frozen=True)
class Abort:
    reason: str = field(default="Workflow cancelled by user")


Signal = RunStarted | StageStarted | StageFinished
Request = ReviewKeywords | ReviewTables | CoderRejected
Answer = Approve | Regenerate | Reject | Force | Rediscover | Abort
