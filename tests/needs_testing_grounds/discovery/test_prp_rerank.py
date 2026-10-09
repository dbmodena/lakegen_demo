"""prp_rerank_candidates (DiscoveryConfig flag): the production port of
experiments/retrieval_lab's "V1" selection arm. When on with an llm, the
accumulated candidate pool is reranked once per search action by a pairwise
LLM judge (PRP quicksort), each reranked candidate's card gains a short
covers/lacks note -- all before the agent's own search/inspect/confirm
loop runs, whose contract is unchanged."""

import json
import re
from types import SimpleNamespace

import pytest

from lakegen.config.experiment import DiscoveryConfig
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode
from lakegen.tools import unified as tools_p12

DOCS = {
    "housing_2021.parquet": {
        "title": "Housing Survey", "publisher": "DLUHC",
        "description": "annual housing survey report", "words": {"annual", "survey"},
    },
    "housing_2021_copy.parquet": {
        "title": "Housing Survey", "publisher": "DLUHC",
        "description": "annual housing survey report", "words": {"annual", "survey"},
    },
    "population.parquet": {
        "title": "Population Survey", "publisher": "ONS",
        "description": "annual population estimates", "words": {"annual", "survey"},
    },
}


def _hit(resource_id, rank):
    doc = DOCS[resource_id]
    return RetrievalHit(
        document={
            "resource_id": resource_id, "title": doc["title"], "publisher": doc["publisher"],
            "description": doc["description"], "columns": [],
        },
        score=1.0 / rank, rank=rank, lexical_rank=rank,
    )


class _FakeRetrievalService:
    def __init__(self, docs: dict):
        self.docs = docs

    def retrieve(self, **kwargs):
        words = {str(w).casefold() for w in kwargs["keywords"]}
        matched = [name for name, doc in self.docs.items() if words <= doc["words"]]
        return [_hit(name, rank) for rank, name in enumerate(matched, 1)]


class _FakeChatResponse:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _FakeJudgeLLM:
    """Only understands the PRP judge system prompt; decides a winner from the
    two `File:` lines in the candidate cards via the supplied rank lookup."""

    def __init__(self, rank: dict[str, int]):
        self.rank = rank
        self.calls: list[str] = []

    def chat(self, messages):
        system, user = messages[0].content, messages[1].content
        self.calls.append(system)
        assert "judge comparing two candidate tables" in system
        a, b = re.findall(r"File: (\S+)", user)
        winner = "A" if self.rank.get(a, 0) >= self.rank.get(b, 0) else "B"
        return _FakeChatResponse(json.dumps({
            "winner": winner,
            "a_covers": f"{a} supplies X", "a_lacks": "nothing",
            "b_covers": f"{b} supplies Y", "b_lacks": "wrong period",
        }))


def _manager(monkeypatch, tmp_path, *, discovery=None, llm=None):
    service = _FakeRetrievalService(DOCS)
    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_a, **_k: service)
    state = tools_p12.P12State()
    manager = tools_p12.Phase12ToolsManager(
        state, solr_client=None, all_files=list(DOCS), csv_dir=tmp_path,
        question="Which housing survey covers the most recent year?",
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
        discovery_config=discovery or DiscoveryConfig(keyword_search="single"),
        llm=llm,
    )
    return state, manager, service


def test_flag_defaults_to_on():
    assert DiscoveryConfig().prp_rerank_candidates is True


def test_llm_none_leaves_retrieval_order_and_makes_no_calls(monkeypatch, tmp_path):
    discovery = DiscoveryConfig(keyword_search="single", prp_rerank_candidates=True)
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=None)

    response = manager.search_keyword_concepts(["annual", "survey"])

    assert "PRP rerank" not in response
    assert "PRP judge notes" not in response
    assert state.all_candidates[0] == "housing_2021.parquet"  # unchanged retrieval order


def test_flag_off_leaves_retrieval_order_untouched(monkeypatch, tmp_path):
    llm = _FakeJudgeLLM(rank={"population.parquet": 99})  # would win every comparison if used
    discovery = DiscoveryConfig(keyword_search="single", prp_rerank_candidates=False)
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=llm)

    response = manager.search_keyword_concepts(["annual", "survey"])

    assert llm.calls == []
    assert "PRP rerank" not in response
    assert state.all_candidates[0] == "housing_2021.parquet"


def test_reranks_candidates_by_judge_verdict(monkeypatch, tmp_path):
    # Retrieval order is housing_2021, housing_2021_copy, population (see DOCS
    # insertion order); the judge is scripted to always prefer population.
    llm = _FakeJudgeLLM(rank={"population.parquet": 1})
    discovery = DiscoveryConfig(keyword_search="single", prp_rerank_candidates=True)
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=llm)

    response = manager.search_keyword_concepts(["annual", "survey"])

    assert llm.calls  # at least one real pairwise judge call happened
    assert state.all_candidates[0] == "population.parquet"
    assert "PRP judge notes: covers: population.parquet supplies X" in response


def test_confirm_unified_selection_still_requires_the_agents_own_inspect_call(monkeypatch, tmp_path):
    """PRP's internal schema profiling must not silently satisfy
    confirm_unified_selection's mandatory inspect_columns requirement -- the
    agent still has to inspect whatever it selects itself."""
    llm = _FakeJudgeLLM(rank={"population.parquet": 1})
    discovery = DiscoveryConfig(keyword_search="single", prp_rerank_candidates=True)
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=llm)

    manager.search_keyword_concepts(["annual", "survey"])
    assert "population.parquet" in state.all_candidates[: state.visible_candidate_count]

    with pytest.raises(ValueError, match="inspect_columns is mandatory"):
        manager.confirm_unified_selection(
            reasoning="test", tables=["population.parquet"],
            requirement_coverage={"year": {"table": "population.parquet", "columns": ["year"]}},
            table_roles={"population.parquet": "fact"},
        )


def test_quicksort_rerank_orders_top_n_and_keeps_rest_in_retrieval_order(monkeypatch, tmp_path):
    scores = {"a": 1, "b": 6, "c": 3, "d": 5, "e": 2, "f": 4}
    state, manager, _ = _manager(monkeypatch, tmp_path, llm=object())
    monkeypatch.setattr(manager, "_prp_confront", lambda a, b: {
        "winner": "A" if scores[a] > scores[b] else "B", "a_covers": a, "b_covers": b,
    })

    ranked, notes, stats = manager._prp_quicksort_rerank(list("abcdef"), 3)

    assert ranked == ["b", "d", "f", "a", "c", "e"]
    assert stats["calls"] > 0 and set(notes) <= set("abcdef")
