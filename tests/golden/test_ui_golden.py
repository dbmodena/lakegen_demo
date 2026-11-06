"""Goldens of the Chainlit workflow, ``lakegen.entrypoints.webapp.workflow.run_lakegen_workflow``.

The interaction golden is the sequence of steps, streamed text, gate prompts
with their buttons and the messages a person sees; it must never change. The
pipeline golden is the call log and the log rows; it changes only in listed
commits. Phases run on a real worker thread, as ``cl.make_async`` runs them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from golden.harness import (
    COMMON_NAMES,
    Case,
    Script,
    experiment,
    make_runtime,
    normalise,
    phase1_result,
    phase3_result,
    prepare_run,
)
from lakegen.entrypoints.webapp.state import LakeGenSession

SCENARIOS: dict[str, Callable[[], tuple[Case, list[str]]]] = {}


def scenario(name: str):
    def register(build):
        SCENARIOS[name] = build
        return build

    return register


SELECTED = (
    ["gold.csv"],
    ["parks"],
    {"gold.csv": {"title": "gold"}},
    "The table covers the question.",
    "trace",
    5,
)
RESELECTED = (
    ["other.csv"],
    ["parks", "2024"],
    {"other.csv": {"title": "other"}},
    "other.csv has the 2024 rows.",
    "trace",
    5,
)
JUDGED = (
    ["gold.csv"],
    ["gold.csv", "other.csv"],
    {"gold.csv": {"title": "gold"}},
    "The selection covers the question.",
    "trace",
    4,
)
ANSWER = ("There are 42 parks.", 3)
REJECTION = phase3_result(
    rejected_reason="gold.csv has no 2024 rows.",
    raw_result=None,
    structured_result=None,
    rejection_keep_tables=[],
    rejection_skip_tables=["gold.csv"],
)
ERROR = phase3_result(
    error="KeyError: 'district'",
    raw_result=None,
    structured_result=None,
    execution_error={"stage": "execution", "category": "runtime_error", "retryable": True},
)
FILES = ("gold.csv", "other.csv")


def unified(**overrides):
    return experiment(interaction_mode="human_gated", **overrides)


def divided(**overrides):
    return experiment(interaction_mode="human_gated", discovery_architecture="divided", **overrides)


def rejecting_phase12(*, state, **_kwargs):
    """The architect rejects and keeps one table."""
    state.rejection_keep_tables = ["kept.csv"]
    return (
        [],
        ["parks", "district"],
        {},
        "REJECT_KEYWORDS: the candidates lack the district column",
        "",
        5,
    )


@scenario("unified_approve")
def _():
    return Case(
        unified(),
        Script(phase12=[SELECTED], phase3=[phase3_result()], phase4=[ANSWER]),
        files=FILES,
    ), ["approve"]


@scenario("unified_architect_rejection_retries_with_its_feedback")
def _():
    # The architect rejects its first round and keeps one table: the next
    # round starts from its feedback and the kept table without a review (the
    # engine's policy since the chat adopted it; the chat used to show the
    # empty selection for approval and merge the person's hint).
    script = Script(
        phase12=[rejecting_phase12, (["kept.csv"], ["parks"], {}, "Selected.", "", 5)],
        phase3=[phase3_result()],
        phase4=[ANSWER],
    )
    return Case(unified(), script, files=FILES + ("kept.csv",)), ["approve"]


@scenario("divided_approve")
def _():
    script = Script(
        phase1=[phase1_result()], phase2=[JUDGED], phase3=[phase3_result()], phase4=[ANSWER]
    )
    return Case(divided(), script, files=FILES), ["approve", "approve"]


@scenario("divided_recalculate_keywords")
def _():
    script = Script(
        phase1=[phase1_result(), phase1_result(("parks", "district", "2024"))],
        phase2=[JUDGED],
        phase3=[phase3_result()],
        phase4=[ANSWER],
    )
    return Case(divided(), script, files=FILES), [
        "recalculate",
        "add the year",
        "approve",
        "approve",
    ]


@scenario("divided_recalculate_tables")
def _():
    script = Script(
        phase1=[phase1_result()],
        phase2=[
            JUDGED,
            (["other.csv"], ["other.csv"], {}, "other.csv matches the hint.", "trace", 4),
        ],
        phase3=[phase3_result()],
        phase4=[ANSWER],
    )
    return Case(divided(), script, files=FILES), [
        "approve",
        "recalculate",
        "use the 2024 table",
        "approve",
    ]


@scenario("divided_judge_rejects_keywords")
def _():
    script = Script(
        phase1=[phase1_result(), phase1_result(("parks", "district", "count"))],
        phase2=[
            (
                [],
                ["gold.csv"],
                {},
                "REJECT_KEYWORDS: the keywords only find fishing licences",
                "trace",
                4,
            ),
            JUDGED,
        ],
        phase3=[phase3_result()],
        phase4=[ANSWER],
    )
    return Case(divided(), script, files=FILES), ["approve", "approve", "approve"]


@scenario("coder_rejection_force")
def _():
    script = Script(phase12=[SELECTED], phase3=[REJECTION, phase3_result()], phase4=[ANSWER])
    return Case(unified(), script, files=FILES), ["approve", "force"]


@scenario("coder_rejection_reevaluate")
def _():
    script = Script(
        phase12=[SELECTED, RESELECTED], phase3=[REJECTION, phase3_result()], phase4=[ANSWER]
    )
    return Case(unified(), script, files=FILES), ["approve", "reevaluate", "approve"]


@scenario("execution_error_retry")
def _():
    script = Script(phase12=[SELECTED], phase3=[ERROR, phase3_result()], phase4=[ANSWER])
    return Case(unified(), script, files=FILES), ["approve"]


@scenario("execution_error_exhausted")
def _():
    script = Script(
        phase12=[SELECTED], phase3=[ERROR], phase4=[("No answer could be computed.", 2)]
    )
    return Case(unified(), script, files=FILES), ["approve"]


class FakeStep:
    """Records what ``cl.Step`` would show."""

    events: list[dict[str, Any]] = []

    def __init__(
        self,
        *,
        name: str,
        type: str = "",
        default_open: bool = True,
        auto_collapse: bool = False,
        **_kwargs,
    ):
        self.name, self.type, self.default_open = name, type, default_open
        self.output = ""

    async def __aenter__(self):
        FakeStep.events.append({"step": self.name, "type": self.type})
        return self

    async def __aexit__(self, *_exc):
        FakeStep.events.append({"step_done": self.name, "output": self.output})
        return False

    async def update(self):
        FakeStep.events.append(
            {"step_update": self.name, "output": self.output, "open": self.default_open}
        )


class FakeBridge:
    def __init__(self, _step):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def emit(self, delta):
        if delta:
            FakeStep.events.append({"stream": delta})


class FakeMessage:
    def __init__(self, *, content: str, elements=None, **_kwargs):
        self.content, self.elements = content, elements or []

    async def send(self):
        FakeStep.events.append(
            {"message": self.content, "elements": [getattr(e, "name", "?") for e in self.elements]}
        )


class FakeText:
    def __init__(self, *, name: str, content: str = "", **_kwargs):
        self.name, self.content = name, content


def make_async(function):
    async def run(*args, **kwargs):
        return await asyncio.to_thread(function, *args, **kwargs)

    return run


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.asyncio
async def test_ui_golden(name, golden, monkeypatch, tmp_path):
    from lakegen.entrypoints.webapp import workflow

    case, answers = SCENARIOS[name]()
    replacements, log, capture = prepare_run(monkeypatch, tmp_path, workflow, case, COMMON_NAMES)
    runtime = make_runtime(case.experiment, tmp_path)
    session = LakeGenSession(runtime=runtime, query=case.question)
    FakeStep.events = []
    remaining = iter(answers)

    def next_answer(prompt: str) -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise AssertionError(
                f"the scenario ran out of answers at the prompt {prompt!r}"
            ) from None

    async def ask_choice(prompt, options, *, phase, gate, approved_value, **_kwargs):
        answer = next_answer(prompt)
        FakeStep.events.append(
            {"ask_choice": prompt, "buttons": [label for _, _, label in options], "answer": answer}
        )
        session.intervention_recorder.record_approval(
            phase=phase, gate=gate, approved=answer == approved_value, elapsed_seconds=0.0
        )
        return answer

    async def ask_hint(prompt, *, phase, gate, **_kwargs):
        answer = next_answer(prompt)
        FakeStep.events.append({"ask_hint": prompt, "answer": answer})
        session.intervention_recorder.record_hint(
            phase=phase, gate=gate, provided=bool(answer), elapsed_seconds=0.0
        )
        return answer

    monkeypatch.setattr(workflow, "get_session", lambda: session)
    monkeypatch.setattr(workflow, "get_runtime_settings", lambda: runtime)
    monkeypatch.setattr(workflow, "_ask_choice", ask_choice)
    monkeypatch.setattr(workflow, "_ask_hint", ask_hint)
    monkeypatch.setattr(workflow, "StepStreamBridge", FakeBridge)
    monkeypatch.setattr(workflow.cl, "Step", FakeStep)
    monkeypatch.setattr(workflow.cl, "Message", FakeMessage)
    monkeypatch.setattr(workflow.cl, "Text", FakeText)
    monkeypatch.setattr(workflow.cl, "make_async", make_async)

    await workflow.run_lakegen_workflow(case.question)

    assert next(remaining, None) is None, "the scenario scripted more answers than the UI asked for"
    golden(f"ui/{name}.interaction", {"events": normalise(FakeStep.events, replacements)})
    golden(f"ui/{name}", {"calls": log.calls, **capture.payload()})
