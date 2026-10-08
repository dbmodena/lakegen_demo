"""The terminal's words: colours, headers, and the questions a person answers.

These are the CLI's prompts as they have always been; the transcript
goldens pin them. Every answer is recorded with the gate it answered.
"""

from __future__ import annotations

import time

from lakegen.tracing import HumanGate, HumanInterventionRecorder, PhaseName, normalize_hint

COLORS = {
    "bold":    "\033[1m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "cyan":    "\033[96m",
    "red":     "\033[91m",
    "reset":   "\033[0m",
    "dim":     "\033[2m",
}


def colour(text: str, color: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def header(title: str) -> None:
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
    return ", ".join(keywords) if keywords else "(none)"
