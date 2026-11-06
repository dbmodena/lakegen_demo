"""The terminal's words: colours, headers, and the questions a person answers.

These are the CLI's prompts as they have always been; the transcript
goldens pin them. Every answer is recorded with the gate it answered.
"""

from __future__ import annotations

import time

from lakegen.pipeline.observability.tracing import (
    HumanGate,
    HumanInterventionRecorder,
    PhaseName,
    normalize_hint,
)

COLORS = {
    "bold": "\033[1m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "red": "\033[91m",
    "reset": "\033[0m",
    "dim": "\033[2m",
}


def colour(text: str, color: str) -> str:
    """The text in one of ``COLORS``; an unknown colour leaves it plain."""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def header(title: str) -> None:
    """Print the title in bold between two rules."""
    print(f"\n{'─' * 60}")
    print(colour(f"  {title}", "bold"))
    print(f"{'─' * 60}")


def ask_yes_no(
    prompt: str,
    default: bool = True,
    *,
    recorder: HumanInterventionRecorder,
    phase: PhaseName,
    gate: HumanGate,
) -> bool:
    """Ask a yes/no question and record the answer as an approval at ``gate``.

    An empty answer takes the default; "y", "yes", "si" and "sì" are yes,
    anything else is no.

    Args:
        prompt: The question.
        default: The answer an empty reply gives.
        recorder: Where the answer is recorded.
        phase: The phase that asks.
        gate: The gate the answer approves or rejects.

    Returns:
        Whether the person said yes.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    started = time.monotonic()
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        approved = default
    else:
        approved = answer in ("y", "yes", "si", "sì")
    recorder.record_approval(
        phase=phase,
        gate=gate,
        approved=approved,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return approved


def ask_input(
    prompt: str,
    allow_empty: bool = True,
    *,
    recorder: HumanInterventionRecorder,
    phase: PhaseName,
    gate: HumanGate,
) -> str:
    """Ask for a hint and record whether one was given at ``gate``.

    Args:
        prompt: The question.
        allow_empty: Whether an empty reply is accepted; if not, the question
            is asked again (and each reply recorded).
        recorder: Where the reply is recorded.
        phase: The phase that asks.
        gate: The gate the reply answers.

    Returns:
        The normalized reply.
    """
    started = time.monotonic()
    value = normalize_hint(input(f"{prompt}: "))
    recorder.record_hint(
        phase=phase,
        gate=gate,
        provided=bool(value),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    if not allow_empty and not value:
        return ask_input(
            prompt,
            allow_empty,
            recorder=recorder,
            phase=phase,
            gate=gate,
        )
    return value


def keyword_list(keywords: list[str]) -> str:
    """The keywords comma-separated, or "(none)"."""
    return ", ".join(keywords) if keywords else "(none)"
