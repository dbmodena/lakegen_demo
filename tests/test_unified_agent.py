"""The unified agent's run: what its answer, its stalls and its errors become."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lakegen.discovery.tools import P12State
from lakegen.discovery.unified import phase12_agent
from lakegen.domain.errors import WorkflowCancelled
from lakegen.integrations.llm.agents.activity import Phase2AgentStall

FILES = ("a.csv", "b.csv")


def _payload(**fields) -> str:
    return "FINAL_PAYLOAD: " + json.dumps(fields)


def _inspected(*tables: str) -> P12State:
    state = P12State()
    state.all_candidates = list(FILES)
    state.visible_candidate_count = len(FILES)
    state.inspection_cache = {table: f"schema of {table}" for table in tables}
    return state


def _run(monkeypatch, tmp_path, responses, *, state=None, require_semantic_plan=False):
    """Run the agent with scripted answers of the agent runner; the result, the agents run, the stream."""
    calls: list[str] = []
    streamed: list[str] = []

    def fake_run(**kwargs):
        calls.append(kwargs["agent_name"])
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("lakegen.integrations.llm.agents.runner.run_agent_workflow", fake_run)
    for name in FILES:
        (tmp_path / name).write_text("value\n1\n", encoding="utf-8")
    result = phase12_agent(
        "How many?", SimpleNamespace(token_usage_total=7), SimpleNamespace(render=lambda *a, **k: "prompt"),
        list(FILES), object(), tmp_path, state=state or P12State(),
        require_semantic_plan=require_semantic_plan, stream_callback=streamed.append,
    )
    return result, calls, "".join(streamed)


def test_a_final_payload_selects_its_known_tables_and_keeps_its_plan(monkeypatch, tmp_path):
    state = P12State()
    plan = {"table_roles": {"a.csv": "primary"}, "combination_strategy": "single_table"}
    answer = _payload(tables="a.csv, zzz.csv", reasoning="a fits", selection_plan=plan, advisories=["check the year"])
    (selected, keywords, metadata, reasoning, trace, tokens), calls, stream = _run(monkeypatch, tmp_path, [answer], state=state)
    assert selected == ["a.csv"] and keywords == [] and metadata == {}
    assert reasoning.startswith("a fits") and "AGENTIC SELECTION PLAN" in reasoning and "check the year" in reasoning
    assert trace.startswith("--- Unified Phase Activity Log ---") and "Unified Architect & Search agent started" in trace
    assert tokens == 7 and calls == ["unified_explorer"]
    assert state.selection_plan == plan and state.selection_plan_source == "final_payload"
    assert state.selection_advisories == ["check the year"]


def test_prose_without_a_payload_falls_back_to_the_inspected_candidates(monkeypatch, tmp_path):
    state = _inspected("a.csv")
    (selected, _, _, reasoning, _, _), _, _ = _run(monkeypatch, tmp_path, ["I think a.csv answers it"], state=state)
    assert selected == ["a.csv"] and reasoning.startswith("I think a.csv answers it")
    assert state.selection_plan_source == "minimal_fallback" and state.selection_plan


def test_a_rejection_payload_stays_a_rejection(monkeypatch, tmp_path):
    state = P12State()
    (selected, _, _, reasoning, _, _), _, _ = _run(
        monkeypatch, tmp_path, [_payload(tables="", reasoning="REJECT_KEYWORDS: nothing fits")], state=state,
    )
    assert selected == [] and reasoning == "REJECT_KEYWORDS: nothing fits"
    assert state.selection_plan == {} and state.selection_plan_source == "minimal_fallback"


def test_a_stall_without_inspected_tables_selects_nothing_and_says_why(monkeypatch, tmp_path):
    state = P12State()
    (selected, _, _, reasoning, _, _), calls, stream = _run(monkeypatch, tmp_path, [Phase2AgentStall("loop")], state=state)
    assert selected == [] and reasoning.startswith("Phase loop guard triggered: loop")
    assert state.initial_stall_reason == "loop" and not state.recovery_started
    assert "Loop guard triggered" in stream and calls == ["unified_explorer"]


def test_a_stall_with_inspected_tables_and_a_required_plan_runs_the_planner_recovery(monkeypatch, tmp_path):
    state = _inspected("a.csv")
    (selected, _, _, reasoning, _, _), calls, stream = _run(
        monkeypatch, tmp_path, [Phase2AgentStall("loop"), _payload(tables="a.csv", reasoning="recovered")],
        state=state, require_semantic_plan=True,
    )
    assert calls == ["unified_explorer", "semantic_planner_recovery"]
    assert selected == ["a.csv"] and reasoning.startswith("recovered")
    assert state.recovery_started and state.initial_stall_reason == "loop" and state.recovery_stop_reason is None
    assert "Semantic planner recovery started" in stream


def test_a_planner_that_stalls_too_rejects_the_keywords(monkeypatch, tmp_path):
    state = _inspected("a.csv")
    (selected, _, _, reasoning, _, _), calls, _ = _run(
        monkeypatch, tmp_path, [Phase2AgentStall("loop"), Phase2AgentStall("again")], state=state, require_semantic_plan=True,
    )
    assert calls == ["unified_explorer", "semantic_planner_recovery"]
    # The rejection names no table, so the inspected ones are the fallback selection.
    assert selected == ["a.csv"] and reasoning.startswith("REJECT_KEYWORDS: semantic planner could not establish")
    assert state.recovery_stop_reason == "again"


def test_an_agent_error_falls_back_to_the_top_inspected_tables(monkeypatch, tmp_path):
    state = _inspected("a.csv", "b.csv")
    (selected, _, _, reasoning, _, _), _, stream = _run(
        monkeypatch, tmp_path, [RuntimeError("Max iterations reached")], state=state,
    )
    assert selected == ["a.csv", "b.csv"]
    assert reasoning.startswith("Agent exceeded maximum iterations without a final decision.")
    assert state.initial_stall_reason == "RuntimeError: Max iterations reached"
    assert "[agent error] Max iterations reached" in stream


def test_an_agent_error_with_a_required_plan_runs_the_recovery_first(monkeypatch, tmp_path):
    state = _inspected("a.csv")
    (selected, _, _, reasoning, _, _), calls, _ = _run(
        monkeypatch, tmp_path, [RuntimeError("boom"), _payload(tables="a.csv", reasoning="recovered after error")],
        state=state, require_semantic_plan=True,
    )
    assert calls == ["unified_explorer", "semantic_planner_recovery"]
    assert selected == ["a.csv"] and reasoning.startswith("recovered after error")


def test_a_cancellation_propagates(monkeypatch, tmp_path):
    with pytest.raises(WorkflowCancelled):
        _run(monkeypatch, tmp_path, [WorkflowCancelled("stop")])
