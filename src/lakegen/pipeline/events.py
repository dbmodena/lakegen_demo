"""What the engine tells its driver, and what it asks.

``Pipeline.steps`` is a generator. It yields signals (a stage started or
finished) that a driver may show, and, in a human-gated run, requests that
the driver must answer by sending one of the answers below back in. An
autonomous run never asks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


# --- requests and their answers ----------------------------------------------

@dataclass(frozen=True)
class ReviewKeywords:
    """The divided architecture wrote keywords; approve them or ask for new ones."""

    keywords: list[str]
    tokens: int


@dataclass(frozen=True)
class ReviewTables:
    """Discovery selected tables; approve them, ask for another round, or give up."""

    keywords: list[str]
    selected: list[str]
    reasoning: str
    tokens: int
    after_rejection: bool = False
    """The round followed a coder rejection: the driver may only approve or abort."""


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
