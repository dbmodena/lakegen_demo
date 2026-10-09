"""format_* are pure string builders (no Chainlit import needed to test
them) -- payloads mirror the exact scratchpad-validated bug shapes from
Phase 1/2's own test fixtures (test_value_grounding.py's FY2016/17 span
bug, test_code_judge.py's grounding-violation rejection). The one Chainlit-
wiring integration test at the bottom reuses test_live_steps.py's FakeStep
pattern."""
import asyncio

import pytest

from lakegen.column_lineage import Violation
from lakegen.interfaces.chainlit import live_steps
from lakegen.interfaces.chainlit.reviewed_coder_steps import (
    emit_review_step, format_code_judge_step, format_pipeline_summary,
    format_plan_step, format_validator_step, step_name_for_event,
)


# ---------------------------------------------------------------------------
# step_name_for_event
# ---------------------------------------------------------------------------

def test_step_name_includes_attempt_tag_for_each_stage():
    assert step_name_for_event({"stage": "plan", "attempt": 2, "max_attempts": 3}) == "🧭 Plan Review · attempt 2/3"
    assert step_name_for_event({"stage": "validator", "attempt": 1, "max_attempts": 3}) == "🔍 Query Validator · attempt 1/3"
    assert step_name_for_event({"stage": "code_judge", "attempt": 3, "max_attempts": 3}) == "⚖️ Code Judge · attempt 3/3"


def test_step_name_for_summary_has_no_attempt_tag():
    assert step_name_for_event({"stage": "summary", "outcome": "validated"}) == "🏁 Pipeline Summary"


# ---------------------------------------------------------------------------
# format_plan_step
# ---------------------------------------------------------------------------

def test_plan_step_approved_shows_check_and_rationale():
    event = {
        "stage": "plan", "attempt": 1, "approved": True, "lineage_violations": [],
        "plan_judgment": {
            "approves_question_fidelity": True, "confidence": 0.9,
            "rationale": "matches the question", "missing_requirements": [], "unjustified_steps": [],
        },
        "max_attempts": 3,
    }
    text = format_plan_step(event)
    assert "✔" in text
    assert "Approved" in text
    assert "matches the question" in text


def test_plan_step_rejected_shows_lineage_violation_and_judge_feedback():
    event = {
        "stage": "plan", "attempt": 1, "approved": False,
        "lineage_violations": [Violation(category="unknown_column", step_order=1, column="Bogus", suggestion="did you mean 'Real'?")],
        "plan_judgment": {
            "approves_question_fidelity": False, "confidence": 0.8,
            "rationale": "amount not read", "missing_requirements": ["Amount written-off"],
            "unjustified_steps": [],
        },
        "max_attempts": 3,
    }
    text = format_plan_step(event)
    assert "✖" in text
    assert "Rejected" in text
    assert "Bogus" in text
    assert "did you mean 'Real'?" in text
    assert "amount not read" in text
    assert "Amount written-off" in text


def test_plan_step_generation_error_is_flagged_distinctly():
    text = format_plan_step({"stage": "plan", "attempt": 2, "event": "generation_error: timeout"})
    assert "⚡" in text
    assert "timeout" in text


# ---------------------------------------------------------------------------
# format_validator_step
# ---------------------------------------------------------------------------

def test_validator_step_cleared_shows_no_issues():
    event = {
        "stage": "validator", "attempt": 2, "approved": True,
        "grounding_violations": [], "validator_flagged": False, "validator_reason": "",
        "max_attempts": 3,
    }
    text = format_validator_step(event)
    assert "✔" in text
    assert "Cleared" in text
    assert "No value-grounding" in text


def test_validator_step_flags_the_fy_span_bug():
    """The exact motivating bug shape: a bare-year filter against a fiscal-
    year-span-coded column."""
    event = {
        "stage": "validator", "attempt": 1, "approved": False,
        "grounding_violations": [{
            "column": "Financial Year", "literal": 2016, "confidence": "HIGH",
            "message": "'Financial Year' holds fiscal-year-span codes like '201617', but the "
                       "code compares it to bare year 2016 which never appears.",
        }],
        "validator_flagged": True,
        "validator_reason": "Result is empty/zero AND a HIGH-confidence value-grounding "
                             "violation was found -- this result is likely wrong, not genuinely zero.",
        "max_attempts": 3,
    }
    text = format_validator_step(event)
    assert "✖" in text
    assert "Flagged" in text
    assert "Financial Year" in text
    assert "fiscal-year-span" in text
    assert "likely wrong, not genuinely zero" in text


def test_validator_step_generation_error_is_flagged_distinctly():
    text = format_validator_step({"stage": "validator", "event": "generation_error: connection reset"})
    assert "⚡" in text
    assert "connection reset" in text


# ---------------------------------------------------------------------------
# format_code_judge_step
# ---------------------------------------------------------------------------

def test_code_judge_step_approved_shows_both_votes():
    event = {
        "stage": "code_judge", "attempt": 1, "approved": True,
        "code_judgment": {
            "plan_compliance_approval": True, "present_result_approval": True,
            "violated_criteria": [], "feedback": "correct", "suggestions": [],
        },
        "max_attempts": 3,
    }
    text = format_code_judge_step(event)
    assert "✔" in text
    assert "Approved" in text
    assert "plan compliance" in text
    assert "present result" in text


def test_code_judge_step_rejected_shows_violated_criteria_and_suggestions():
    event = {
        "stage": "code_judge", "attempt": 1, "approved": False,
        "code_judgment": {
            "plan_compliance_approval": False, "present_result_approval": False,
            "violated_criteria": ["silent_filter_bias"],
            "feedback": "wrong fiscal year code",
            "suggestions": ["[FIX QUERY] use 201617 not 2016"],
        },
        "max_attempts": 3,
    }
    text = format_code_judge_step(event)
    assert "✖" in text
    assert "silent_filter_bias" in text
    assert "wrong fiscal year code" in text
    assert "[FIX QUERY] use 201617 not 2016" in text


def test_code_judge_step_generation_error_is_flagged_distinctly():
    text = format_code_judge_step({"stage": "code_judge", "event": "generation_error: rate limited"})
    assert "⚡" in text
    assert "rate limited" in text


# ---------------------------------------------------------------------------
# format_pipeline_summary
# ---------------------------------------------------------------------------

def test_summary_validated():
    text = format_pipeline_summary({
        "stage": "summary", "outcome": "validated", "reason": "",
        "generation_calls": 3, "wall_time": 12.4,
    })
    assert "✔" in text
    assert "Validated" in text
    assert "3" in text
    assert "12.4" in text


def test_summary_declined_shows_reason():
    text = format_pipeline_summary({
        "stage": "summary", "outcome": "declined",
        "reason": "Plan stage never approved after 3 tries: amount not read.",
        "generation_calls": 3, "wall_time": 9.1,
    })
    assert "✖" in text
    assert "Declined" in text
    assert "amount not read" in text


# ---------------------------------------------------------------------------
# emit_review_step -- Chainlit wiring integration (reuses test_live_steps.py's FakeStep)
# ---------------------------------------------------------------------------

class FakeStep:
    events: list[dict] = []

    def __init__(self, name, type="tool"):
        self.name = name
        self.type = type
        self.parent_id = None
        self.output = None
        self.is_error = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        FakeStep.events.append({"name": self.name, "output": self.output})
        return False


@pytest.fixture(autouse=True)
def _fake_cl_step(monkeypatch):
    FakeStep.events = []
    monkeypatch.setattr(live_steps.cl, "Step", FakeStep)
    yield


@pytest.mark.asyncio
async def test_emit_review_step_renders_each_trace_event_as_its_own_step():
    emitter = live_steps.ThreadSafeStepEmitter()
    counter = {"n": 0}

    trace = [
        {"stage": "plan", "attempt": 1, "approved": True, "lineage_violations": [],
         "plan_judgment": {"approves_question_fidelity": True, "rationale": "ok",
                            "missing_requirements": [], "unjustified_steps": []},
         "max_attempts": 3},
        {"stage": "validator", "attempt": 1, "approved": True, "grounding_violations": [],
         "validator_flagged": False, "validator_reason": "", "max_attempts": 3},
        {"stage": "summary", "outcome": "validated", "reason": "", "generation_calls": 1, "wall_time": 1.0},
    ]
    for event in trace:
        emit_review_step(emitter, event, counter)
    await emitter.aclose()

    assert counter["n"] == 3
    assert len(FakeStep.events) == 3
    assert FakeStep.events[0]["name"] == "🧭 Plan Review · attempt 1/3"
    assert FakeStep.events[1]["name"] == "🔍 Query Validator · attempt 1/3"
    assert FakeStep.events[2]["name"] == "🏁 Pipeline Summary"
    assert "Approved" in FakeStep.events[0]["output"]
