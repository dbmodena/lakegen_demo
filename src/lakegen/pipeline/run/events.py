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
    """The run started: the lake's tables are listed.

    Attributes:
        tables: How many tables the lake holds.
    """

    tables: int


@dataclass(frozen=True)
class StageStarted:
    """A stage started.

    Attributes:
        stage: ``discovery`` (a round), ``coding`` (a round), ``code`` (an attempt) or ``answer``.
        attempt: The round's or the attempt's number, from 1.
    """

    stage: str
    attempt: int = 1


@dataclass(frozen=True)
class StageFinished:
    """A stage finished, with what it produced.

    Attributes:
        stage: The stage, as in ``StageStarted``.
        attempt: The round's or the attempt's number, from 1.
        detail: Discovery: ``selected``, ``keywords_rejected`` or ``forced_selection``;
            code: the attempt's status.
        message: Discovery: the architect's reason; code: the result or the error.
        more: Another attempt of the same stage follows.
        details: What the stage produced, for a driver that shows more than the message:
            discovery: keywords, selected, candidates, reasoning, trace, selection_trace,
            tokens, hint; code: code_raw, clean_code, tokens, error, error_msg, raw_result;
            answer: answer, tokens.
    """

    stage: str
    attempt: int = 1
    detail: str = ""
    message: str = ""
    more: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# --- requests and their answers ----------------------------------------------


@dataclass(frozen=True)
class ReviewKeywords:
    """The divided architecture wrote keywords; approve them or ask for new ones.

    Attributes:
        keywords: The keywords written.
        tokens: The tokens the keyword writer spent.
        hint: What the keyword writer was told for this round.
        raw: The writer's raw output.
        reasoning: The writer's reasoning.
    """

    keywords: list[str]
    tokens: int
    hint: str = ""
    raw: str = ""
    reasoning: str = ""


@dataclass(frozen=True)
class ReviewTables:
    """Discovery selected tables; approve them, ask for another round, or give up.

    Attributes:
        keywords: The round's keywords.
        selected: The selected tables.
        reasoning: Discovery's reasoning.
        tokens: The tokens of the selection (of the whole round for the unified agent).
        after_rejection: The round followed a coder rejection: the driver may only
            approve or abort.
        architect_feedback: The architect's own feedback on this round, when it rejected
            it; a hint the person adds is merged with it.
    """

    keywords: list[str]
    selected: list[str]
    reasoning: str
    tokens: int
    after_rejection: bool = False
    architect_feedback: str = ""


@dataclass(frozen=True)
class CoderRejected:
    """The coder rejected the tables; force the code anyway, or discover again.

    Attributes:
        reason: Why the coder rejected them.
    """

    reason: str


@dataclass(frozen=True)
class Approve:
    """Approve what the request shows."""


@dataclass(frozen=True)
class Regenerate:
    """New keywords, with a hint for the keyword writer.

    Attributes:
        hint: The hint for the keyword writer.
    """

    hint: str = ""


@dataclass(frozen=True)
class Reject:
    """Another discovery round, with a hint for the architect.

    Attributes:
        hint: The hint for the next round.
        regenerate_keywords: Divided architecture: write new keywords (the CLI's way) or
            re-select with the same ones (the chat's).
    """

    hint: str = ""
    regenerate_keywords: bool = True


@dataclass(frozen=True)
class Force:
    """Run the code on the rejected tables anyway."""


@dataclass(frozen=True)
class Rediscover:
    """Accept the coder's rejection: discovery runs again."""


@dataclass(frozen=True)
class Abort:
    """Give up the run.

    Attributes:
        reason: Why; the result's error.
    """

    reason: str = field(default="Workflow cancelled by user")


Signal = RunStarted | StageStarted | StageFinished
Request = ReviewKeywords | ReviewTables | CoderRejected
Answer = Approve | Regenerate | Reject | Force | Rediscover | Abort
