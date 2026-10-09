"""The divided architecture's judge: what its answer, its stalls and its errors become."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lakegen.domain.errors import WorkflowCancelled
from lakegen.integrations.llm.agents.activity import Phase2AgentStall
from lakegen.retrieval import RetrievalConfig
from lakegen.tools.unified import P12State
from lakegen.workflow.discovery import divided

FILES = ["a.parquet", "b.parquet", "c.parquet"]
META = {name: {"title": name, "columns": []} for name in FILES}


def _payload(**fields) -> str:
    return "FINAL_PAYLOAD: " + json.dumps(fields)


def _run(monkeypatch, tmp_path, responses, *, candidates=FILES, state=None):
    """Run the judge on scripted retrieval and agent answers; the result, the agents run, the stream."""
    calls: list[str] = []
    streamed: list[str] = []
    monkeypatch.setattr(divided, "_solr_and_search", lambda *a, **k: (list(candidates), {c: META[c] for c in candidates}))

    def fake_run(**kwargs):
        calls.append(kwargs["agent_name"])
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr("lakegen.integrations.llm.agents.runner.run_agent_workflow", fake_run)
    result = divided.phase2_select_tables(
        "How many?", SimpleNamespace(token_usage_total=5), SimpleNamespace(render=lambda *a, **k: "prompt"),
        FILES, ["parks"], object(), tmp_path, stream_callback=streamed.append,
        retrieval_config=RetrievalConfig(), selection_state=state,
    )
    return result, calls, "".join(streamed)


def test_a_final_payload_selects_its_known_tables(monkeypatch, tmp_path):
    state = P12State()
    (selected, candidates, metadata, reasoning, trace, tokens), calls, stream = _run(
        monkeypatch, tmp_path, [_payload(tables="b.parquet, zzz.parquet", reasoning="b fits")], state=state,
    )
    assert selected == ["b.parquet"] and candidates == FILES and metadata == META
    assert reasoning == "b fits" and tokens == 5 and calls == ["table_judge"]
    assert trace.startswith("--- Phase 2 Activity Log ---") and "Table Judge agent started" in trace
    assert "showing 3 initially" in stream
    assert state.rejection_keep_tables == [] and state.rejection_skip_tables == [] and state.selection_plan == {}


def test_no_candidates_reject_the_keywords_before_any_agent_runs(monkeypatch, tmp_path):
    (selected, candidates, metadata, reasoning, trace, tokens), calls, stream = _run(monkeypatch, tmp_path, [], candidates=[])
    assert (selected, candidates, metadata, tokens) == ([], [], {}, 0)
    assert reasoning == "REJECT_KEYWORDS: No tables found with keyword retrieval for [parks]. Try broader or different terms."
    assert trace.startswith("[phase2] keyword retrieval with keywords=['parks'] returned 0 results.")
    assert calls == [] and "No tables found" in stream


def test_a_rejection_answer_rejects_the_keywords(monkeypatch, tmp_path):
    (selected, _, _, reasoning, _, _), _, _ = _run(monkeypatch, tmp_path, ["REJECT_KEYWORDS: nothing covers 2024"])
    assert selected == [] and reasoning == "REJECT_KEYWORDS: nothing covers 2024"


def test_a_stall_falls_back_to_the_top_candidates(monkeypatch, tmp_path):
    state = P12State()
    (selected, _, _, reasoning, _, _), _, stream = _run(monkeypatch, tmp_path, [Phase2AgentStall("loop")], state=state)
    # Nothing was inspected, so the fallback names no table and the parser takes the top two.
    assert selected == ["a.parquet", "b.parquet"]
    assert reasoning == "Phase 2 loop guard triggered: loop. Fallback restricted to inspected candidates."
    assert "Phase 2 loop guard triggered" in stream and state.selection_plan == {}


def test_an_agent_error_falls_back_and_says_so(monkeypatch, tmp_path):
    (selected, _, _, reasoning, _, _), _, stream = _run(monkeypatch, tmp_path, [RuntimeError("Max iterations reached")])
    assert selected == ["a.parquet", "b.parquet"]
    assert reasoning == "Agent exceeded maximum iterations. Fallback to top candidates."
    assert "[phase2 agent error] Max iterations reached" in stream
    (_, _, _, reasoning, _, _), _, _ = _run(monkeypatch, tmp_path, [RuntimeError("boom")])
    assert reasoning == "Agent error: boom. Fallback to top 2."


def test_a_cancellation_propagates(monkeypatch, tmp_path):
    with pytest.raises(WorkflowCancelled):
        _run(monkeypatch, tmp_path, [WorkflowCancelled("stop")])
