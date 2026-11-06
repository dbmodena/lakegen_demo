"""The unified agent's search rule, offline: a fake catalogue answers the searches, parquet files the inspections."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.tools import ToolSelection

from lakegen.llm.prompts import get_prompt_manager
from lakegen.pipeline.discovery import unified
from lakegen.pipeline.discovery.selectors.memgpt.agent import MAX_EXPANSIONS, MemGPTAgent
from lakegen.pipeline.discovery.state import DiscoveryState
from lakegen.pipeline.discovery.unified import (
    MAX_SEARCHES,
    REACTIVATIONS,
    UnifiedAgent,
    discover_tables,
    search_words,
)

QUESTION = "How many tickets were sold at San Siro in June 2026?"
JUNE = "san_siro_tickets_june_2026.parquet"
REQS = ["measure: tickets sold at San Siro", "period: June 2026"]


@pytest.fixture()
def lake(tmp_path):
    """The San Siro table at #1 among 14 unrelated ones."""
    frames = {
        JUNE: pd.DataFrame(
            {"match_date": [f"2026-06-{d:02d}" for d in range(1, 21)], "tickets_sold": range(20)}
        )
    }
    for i in range(14):
        frames[f"other_{i:02d}.parquet"] = pd.DataFrame({"a": range(5), "b": [f"x{i}"] * 5})
    for name, frame in frames.items():
        frame.to_parquet(tmp_path / name, index=False)
    order = list(frames)
    metadata = {
        name: {
            "title": "San Siro tickets sold June 2026"
            if name == JUNE
            else f"Unrelated dataset {name[6:8]}",
            "description": "",
            "columns.name": list(frames[name].columns),
        }
        for name in order
    }
    return SimpleNamespace(dir=tmp_path, found=[(name, metadata[name]) for name in order])


class ScriptedLLM:
    """Answers each turn with the next scripted tool call."""

    def __init__(self, script):
        self.script = list(script)
        self.offered: list[list[str]] = []
        self.token_usage_total = 0

    def reset_token_usage(self):
        self.token_usage_total = 0

    def chat_with_tools(self, tools, chat_history=None, **_kwargs):
        self.offered.append([t.metadata.name for t in tools])
        name, args = (
            self.script.pop(0) if self.script else ("confirm", {"reasoning": "out of script"})
        )
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT, content="", additional_kwargs={"call": (name, args)}
            )
        )

    def get_tool_calls_from_response(self, response, error_on_no_tool_call=True):
        name, args = response.message.additional_kwargs["call"]
        return [ToolSelection(tool_id="x", tool_name=name, tool_kwargs=args)]


def _agent(lake, found=None, llm=None) -> UnifiedAgent:
    agent = UnifiedAgent(
        question=QUESTION,
        table_dir=lake.dir,
        llm=llm,
        prompts=get_prompt_manager(),
        portal="Test portal",
    )
    agent.searcher = lambda query, first: lake.found if found is None else found
    return agent


def test_requirements_come_first_then_search_alone(lake):
    agent = _agent(lake)
    assert agent.available() == ["define_requirements"]
    agent.define_requirements(REQS)
    assert agent.available() == ["search", "reject"]
    assert agent.next_step().startswith("search(query)")
    assert "search(query, requirement)" in agent.system_message()


def test_a_search_that_finds_candidates_withdraws_search(lake):
    agent = _agent(lake)
    agent.define_requirements(REQS)
    text = agent.search("San Siro tickets 2026")
    assert text.startswith("Found 15 candidates") and "5 hidden" in text
    assert agent.closed and agent.searches == 1 and agent.reactivations == 0
    assert len(agent.candidates) == 15 and agent.memory.visible == 10
    assert "search" not in agent.available() and "inspect" in agent.available()
    assert agent.queries[0]["first"] and len(agent.queries[0]["new"]) == 15
    assert "Search 1: 'San Siro tickets 2026' -> 15 new candidates" in agent.system_message()
    assert "searches 1 of" in agent.status()[0]


def test_a_search_that_finds_nothing_keeps_search_offered(lake):
    agent = _agent(lake, found=[])
    agent.define_requirements(REQS)
    assert agent.search("nothing").startswith("No new candidate")
    assert not agent.closed and "search" in agent.available() and "other words" in agent.next_step()


def test_search_comes_back_for_an_open_requirement_once_nothing_is_left_to_inspect(lake):
    agent = _agent(lake)
    agent.define_requirements([*REQS, "filter: something no table mentions zzqx"])
    agent.search("first")
    agent.expansions = MAX_EXPANSIONS  # nothing hidden can be expanded for it
    assert "search" not in agent.available()  # nothing inspected since the search
    agent.inspect([1, 2])
    while agent.memory.queue() and agent.actions_left > 0:
        agent.inspect([n for n, _ in agent.memory.queue()[:4]])
    assert agent.memory.open_requirements() and not agent.memory.queue()
    assert "search" in agent.available() and agent.next_step().startswith("search for R")
    assert agent.search("again").startswith("No new candidate") and agent.reactivations == 1
    assert "search" in agent.available()  # nothing found: still offered
    agent.searcher = lambda query, first: [("brand_new.parquet", {"title": "Brand new table"})]
    assert agent.search("other words").startswith("Found 1 candidates")
    assert agent.reactivations == REACTIVATIONS and agent.memory.visible == len(agent.candidates)
    assert "search" not in agent.available()  # the cap on reactivations
    assert MAX_SEARCHES >= REACTIVATIONS + 1


def test_the_tools_keep_production_descriptions_and_the_search_call_is_recovered_from_text(lake):
    agent = _agent(lake)
    assert agent.functions()["search"].__doc__.startswith("Search the catalogue")
    assert agent.functions()["confirm"].__doc__ == MemGPTAgent.confirm.__doc__
    assert search_words("Tickets sold at San Siro in 2026") == [
        "tickets",
        "sold",
        "san",
        "siro",
        "2026",
    ]


def _offline_round(monkeypatch, lake, script, found=None):
    """``discover_tables`` with the fake catalogue and no contrast, drop or alternatives check."""
    state = DiscoveryState()

    class Searcher:
        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, query, first):
            state.search_attempts.append({"query": query, "first": first})
            state.used_keywords = search_words(query)
            return lake.found if found is None else found

    monkeypatch.setattr(unified, "CatalogueSearcher", Searcher)
    monkeypatch.setattr(unified.contrast, "check", lambda *_a, **_k: None)
    monkeypatch.setattr(unified.drop, "check", lambda *_a, **_k: None)
    monkeypatch.setattr(unified.alternatives, "check", lambda *_a, **_k: None)
    llm = ScriptedLLM(script)
    result = discover_tables(
        QUESTION,
        llm,
        get_prompt_manager(),
        [n for n, _ in lake.found],
        None,
        lake.dir,
        portal_name="Test portal",
        state=state,
        lake="test",
    )
    return result, state, llm


def test_a_confirmed_round_records_the_selection_and_the_searches_on_the_state(monkeypatch, lake):
    script = [
        ("define_requirements", {"requirements": REQS}),
        ("search", {"query": "San Siro tickets June 2026"}),
        ("inspect", {"candidates": [1]}),
        ("select", {"table": 1, "covers": ["R1", "R2"]}),
        ("confirm", {"reasoning": "#1 holds the June 2026 tickets."}),
    ]
    (selected, keywords, meta, reasoning, trace, _tokens), state, llm = _offline_round(
        monkeypatch, lake, script
    )
    assert selected == [JUNE] and keywords == search_words("San Siro tickets June 2026")
    assert (
        set(meta) == {n for n, _ in lake.found} and reasoning == "#1 holds the June 2026 tickets."
    )
    assert trace.startswith("--- MemGPT Table Selector ---")
    assert state.confirmed_tables == [JUNE] and state.selection_plan_source == "memgpt_selector"
    assert state.selection_plan["table_roles"] and state.all_candidates == [
        n for n, _ in lake.found
    ]
    assert state.visible_candidate_count == 10 and state.used_keywords == keywords
    assert [a["first"] for a in state.search_attempts] == [True]
    assert llm.offered[1] == ["search", "reject"] and "search" not in llm.offered[2]


def test_a_rejected_round_asks_for_another_with_nothing_selected(monkeypatch, lake):
    script = [
        ("define_requirements", {"requirements": REQS}),
        ("search", {"query": "nothing matches"}),
        ("inspect", {"candidates": [1, 2]}),
        ("reject", {"reasoning": "no candidate holds tickets", "suggestion": "stadium attendance"}),
    ]
    unrelated = [(name, meta) for name, meta in lake.found if name != JUNE]
    (selected, _keywords, _meta, reasoning, _trace, _tokens), state, _llm = _offline_round(
        monkeypatch, lake, script, found=unrelated
    )
    assert selected == [] and reasoning.startswith("REJECT_KEYWORDS: no candidate holds tickets")
    assert state.confirmed_tables == [] and state.rejection_skip_tables == []
