"""A human-gated run asks its driver at the gates; an autonomous run never does.

The phases are the golden harness's scripted fakes, installed on the service
module and read through ``PipelineDeps.from_module``; a scripted driver
answers each request and the test checks what was asked, in which order,
and how the run ended.
"""

from __future__ import annotations

import pytest

import lakegen.service as service
from golden.harness import (
    SERVICE_NAMES,
    Case,
    Script,
    experiment,
    make_runtime,
    phase1_result,
    phase3_result,
    prepare_run,
)
from lakegen.pipeline.context import RunContext, SurfaceOptions
from lakegen.pipeline.deps import PipelineDeps
from lakegen.pipeline.engine import Pipeline
from lakegen.pipeline.events import (
    Abort,
    Approve,
    CoderRejected,
    Force,
    Rediscover,
    Regenerate,
    Reject,
    ReviewKeywords,
    ReviewTables,
    RunStarted,
    StageFinished,
    StageStarted,
)

SELECTED = (["gold.csv"], ["parks"], {"gold.csv": {"title": "gold"}}, "The table covers the question.", "trace", 5)
RESELECTED = (["other.csv"], ["parks", "2024"], {"other.csv": {"title": "other"}}, "other.csv has the 2024 rows.", "trace", 5)
JUDGED = (["gold.csv"], ["gold.csv", "other.csv"], {"gold.csv": {"title": "gold"}}, "The selection covers the question.", "trace", 4)
ANSWER = ("There are 42 parks.", 3)
REJECTION = phase3_result(rejected_reason="gold.csv has no 2024 rows.", raw_result=None, structured_result=None,
                          rejection_skip_tables=["gold.csv"])
FILES = ("gold.csv", "other.csv")


def _drive(monkeypatch, tmp_path, case: Case, answers: list, *, gated: bool = True):
    """Run the case with the scripted answers; the requests asked and the result."""
    _replacements, log, _capture = prepare_run(monkeypatch, tmp_path, service, case, SERVICE_NAMES)
    ctx = RunContext.from_runtime(make_runtime(case.experiment, tmp_path), SurfaceOptions(human_gated=gated))
    steps = Pipeline(ctx, PipelineDeps.from_module(service)).steps(case.question)
    asked, signals, remaining = [], [], iter(answers)
    event = next(steps)
    try:
        while True:
            if isinstance(event, RunStarted | StageStarted | StageFinished):
                signals.append(event)
                event = steps.send(None)
                continue
            asked.append(event)
            event = steps.send(next(remaining))
    except StopIteration as stop:
        result = stop.value
    assert next(remaining, None) is None, "more answers than requests"
    return asked, signals, result, log


def test_an_autonomous_run_asks_nothing_and_a_gated_one_reviews_the_tables(monkeypatch, tmp_path):
    case = Case(experiment(), Script(phase12=[SELECTED], phase3=[phase3_result()], phase4=[ANSWER]), files=FILES)
    asked, signals, result, _ = _drive(monkeypatch, tmp_path, case, [], gated=False)
    assert asked == [] and result.status == "completed"
    assert signals[0] == RunStarted(tables=2)
    assert [(s.stage, type(s).__name__) for s in signals[1:]] == [
        ("discovery", "StageStarted"), ("discovery", "StageFinished"),
        ("coding", "StageStarted"), ("code", "StageStarted"), ("code", "StageFinished"),
        ("answer", "StageStarted"), ("coding", "StageFinished"),
    ]
    assert signals[2].detail == "selected" and signals[5].detail == "succeeded" and signals[5].message == "42"
    asked, _, result, _ = _drive(monkeypatch, tmp_path, case, [Approve()])
    assert asked == [ReviewTables(keywords=["parks"], selected=["gold.csv"], reasoning="The table covers the question.", tokens=5)]
    assert result.status == "completed" and result.answer == "There are 42 parks."


def test_a_rejected_review_runs_discovery_again_with_the_hint_without_spending_the_budget(monkeypatch, tmp_path):
    case = Case(experiment(), Script(phase12=[SELECTED, RESELECTED], phase3=[phase3_result()], phase4=[ANSWER]), files=FILES)
    asked, _, result, log = _drive(monkeypatch, tmp_path, case, [Reject("look for the 2024 edition"), Approve()])
    assert [a.selected for a in asked] == [["gold.csv"], ["other.csv"]]
    hints = [call["kwargs"]["hint"] for call in log.calls if call["call"] == "phase12_agent"]
    assert hints == ["", "look for the 2024 edition"]
    assert [r["attempt"] for r in result.discovery["selection_attempts"]] == [1, 1]
    assert result.status == "completed" and result.tables == ["other.csv"]


def test_divided_keywords_are_reviewed_before_the_selection(monkeypatch, tmp_path):
    case = Case(
        experiment(discovery_architecture="divided"),
        Script(phase1=[phase1_result(), phase1_result(("parks", "district", "2024"))], phase2=[JUDGED],
               phase3=[phase3_result()], phase4=[ANSWER]),
        files=FILES,
    )
    asked, _, result, log = _drive(monkeypatch, tmp_path, case, [Regenerate("add the year"), Approve(), Approve()])
    assert asked[0] == ReviewKeywords(keywords=["parks", "district"], tokens=2)
    assert asked[1] == ReviewKeywords(keywords=["parks", "district", "2024"], tokens=2)
    assert isinstance(asked[2], ReviewTables) and asked[2].tokens == 4
    keyword_calls = [call["kwargs"] for call in log.calls if call["call"] == "phase1_generate_keywords"]
    assert keyword_calls[1]["hint"] == "add the year" and keyword_calls[1]["avoid_keywords"] == ["parks", "district"]
    assert result.status == "completed"


def test_a_coder_rejection_can_be_forced_or_sent_back_to_discovery(monkeypatch, tmp_path):
    forced = Case(experiment(), Script(phase12=[SELECTED], phase3=[REJECTION, phase3_result()], phase4=[ANSWER]), files=FILES)
    asked, _, result, log = _drive(monkeypatch, tmp_path, forced, [Approve(), Force()])
    assert asked[1] == CoderRejected(reason="gold.csv has no 2024 rows.")
    phase3_calls = [call for call in log.calls if call["call"] == "phase3_generate_and_execute"]
    assert "force_execution" not in phase3_calls[0]["kwargs"] and phase3_calls[1]["kwargs"]["force_execution"] is True
    assert phase3_calls[1]["kwargs"]["retries"] == 0 and result.status == "completed"

    again = Case(experiment(), Script(phase12=[SELECTED, RESELECTED], phase3=[REJECTION, phase3_result()], phase4=[ANSWER]), files=FILES)
    asked, _, result, log = _drive(monkeypatch, tmp_path, again, [Approve(), Rediscover(), Approve()])
    assert isinstance(asked[2], ReviewTables) and asked[2].after_rejection and asked[2].selected == ["other.csv"]
    second_hint = [call["kwargs"]["hint"] for call in log.calls if call["call"] == "phase12_agent"][1]
    assert second_hint.startswith("The code generator rejected the previous table combination.")
    assert result.status == "completed" and result.tables == ["other.csv"]

    aborted = Case(experiment(), Script(phase12=[SELECTED, RESELECTED], phase3=[REJECTION]), files=FILES)
    asked, _, result, _ = _drive(monkeypatch, tmp_path, aborted, [Approve(), Rediscover(), Abort()])
    assert result.status == "cancelled" and result.error == "Workflow cancelled by user"


def test_a_wrong_answer_type_is_refused(monkeypatch, tmp_path):
    case = Case(experiment(), Script(phase12=[SELECTED], phase3=[phase3_result()], phase4=[ANSWER]), files=FILES)
    asked, _, result, _ = _drive(monkeypatch, tmp_path, case, [Force()])
    assert result.status == "failed" and "TypeError" in result.error
    with pytest.raises(RuntimeError, match="cannot answer"):
        ctx = RunContext.from_runtime(make_runtime(case.experiment, tmp_path), SurfaceOptions(human_gated=True))
        prepare_run(monkeypatch, tmp_path, service, case, SERVICE_NAMES)
        Pipeline(ctx, PipelineDeps.from_module(service)).run(case.question)
