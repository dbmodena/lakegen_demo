"""Functional checks for the delegated_final_selection_enabled override at
the top of confirm_unified_selection."""

import json
import re

import pandas as pd
import pytest

from lakegen.agent_tools.tools_p12 import P12State, Phase12ToolsManager
from lakegen.config.experiment import DiscoveryConfig


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeResponse:
    def __init__(self, content):
        self.message = FakeMessage(content)


class ScriptedLLM:
    """Dispatches on which of candidate_inspection's/delegated_selection's
    two distinct prompt shapes it's being asked, so one fake covers both
    the verdict-compression path and the final selection call."""

    def __init__(self, delegated_selected, delegated_reasoning="a.parquet is the only real match."):
        self.delegated_selected = delegated_selected
        self.delegated_reasoning = delegated_reasoning
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        prompt = messages[0].content
        if "Respond with ONLY a JSON list" in prompt:
            return FakeResponse('["value"]')
        if "table-selection agent" in prompt:
            payload = {"selected": self.delegated_selected, "reasoning": self.delegated_reasoning}
            return FakeResponse(json.dumps(payload))
        # candidate_inspection.miniagent_inspect's judging prompt
        return FakeResponse("VERDICT: RELEVANT\nCOLUMNS: value\nREASON: plausible match.")


def _manager(tmp_path, *, discovery, llm=None):
    for name in ("a", "b"):
        pd.DataFrame({"value": [1, 2, 3]}).to_parquet(tmp_path / f"{name}.parquet")
    state = P12State()
    state.all_candidates = ["a.parquet", "b.parquet"]
    state.visible_candidate_count = 2
    for name in state.all_candidates:
        state.inspection_cache[name.casefold()] = f"Schema for {name}"
        state.solr_meta[name] = {"title": name, "description": "", "tags": [], "columns.name": ["value"]}
        state.inspection_counts[name.casefold()] = 1
    return state, Phase12ToolsManager(
        state, object(), state.all_candidates, tmp_path,
        question="what is the value?",
        discovery_config=discovery,
        llm=llm,
    )


def test_delegated_pick_overrides_the_agents_own_proposal(tmp_path):
    llm = ScriptedLLM(delegated_selected=["a.parquet"])
    state, manager = _manager(
        tmp_path, discovery=DiscoveryConfig(delegated_final_selection_enabled=True), llm=llm,
    )

    result = manager.confirm_unified_selection(
        "Both tables look relevant.", ["a.parquet", "b.parquet"],
    )

    assert "FINAL_PAYLOAD" in result
    assert state.confirmed_tables == ["a.parquet"]  # narrowed, not the agent's ["a", "b"]
    assert len(state.delegated_selection_log) == 1
    logged = state.delegated_selection_log[0]
    assert logged["agent_proposed"] == ["a.parquet", "b.parquet"]
    assert logged["delegated_selected"] == ["a.parquet"]
    assert "a.parquet is the only real match" in state.selection_reasoning


def test_empty_delegated_pick_falls_through_to_the_normal_empty_selection_error(tmp_path):
    llm = ScriptedLLM(delegated_selected=[])
    _state, manager = _manager(
        tmp_path, discovery=DiscoveryConfig(delegated_final_selection_enabled=True), llm=llm,
    )
    with pytest.raises(ValueError, match="at least one inspected table is required"):
        manager.confirm_unified_selection(
            "I think a is right.", ["a.parquet"],
        )


def test_reuses_cached_verdicts_from_parallel_inspection_without_rejudging(tmp_path):
    llm = ScriptedLLM(delegated_selected=["a.parquet"])
    state, manager = _manager(
        tmp_path, discovery=DiscoveryConfig(delegated_final_selection_enabled=True), llm=llm,
    )
    # Pre-seed verdict_cache as automatic inspection would -- confirm
    # the override does NOT re-run miniagent_inspect for these.
    for name in state.all_candidates:
        state.verdict_cache[name.casefold()] = {
            "rank": 1, "file": name, "title": name, "publisher": "",
            "verdict": "RELEVANT", "raw": "VERDICT: RELEVANT\nCOLUMNS: value\nREASON: cached.",
        }
    calls_before = llm.calls
    manager.confirm_unified_selection("Both look relevant.", ["a.parquet", "b.parquet"])
    # Only the final_selection call itself should have hit the LLM -- no
    # verdict-compression calls, since both candidates were already cached.
    assert llm.calls == calls_before + 1


def test_flag_off_leaves_the_agents_own_selection_untouched(tmp_path):
    llm = ScriptedLLM(delegated_selected=["a.parquet"])
    state, manager = _manager(
        tmp_path, discovery=DiscoveryConfig(delegated_final_selection_enabled=False), llm=llm,
    )
    manager.confirm_unified_selection("Both tables are needed.", ["a.parquet", "b.parquet"])
    assert state.confirmed_tables == ["a.parquet", "b.parquet"]
    assert llm.calls == 0
    assert state.delegated_selection_log == []
