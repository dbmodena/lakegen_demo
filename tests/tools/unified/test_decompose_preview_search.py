"""keyword_search="decompose_preview" (DiscoveryConfig): the production port of
experiments/retrieval_lab's decompose_tuned_preview_noacronym. With an llm,
search_keyword_concepts splits the ORIGINAL question into per-table search
phrases, previews real AND-word candidates for each, and lets the model pick
before anything actually runs."""

import json
from types import SimpleNamespace

from lakegen.config.experiment import DiscoveryConfig
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode
from lakegen.tools import unified as tools_p12
from lakegen.tools.unified import search as search_tools


def _hit(resource_id, rank=1):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id, "columns": []},
        score=1.0 / rank, rank=rank, lexical_rank=rank,
    )


class _FakeRetrievalService:
    """keywords -> hits, by exact-word-subset containment, like the real AND search."""

    def __init__(self, tables: dict[str, set[str]]):
        self.tables = tables
        self.retrieves: list[list[str]] = []

    def retrieve(self, **kwargs):
        words = {str(w).casefold() for w in kwargs["keywords"]}
        self.retrieves.append(sorted(words))
        matched = [name for name, table_words in self.tables.items() if words <= table_words]
        return [_hit(name, rank) for rank, name in enumerate(matched, 1)]


class _FakeSolr:
    """Same little universe of documents answers word-frequency counts, AND-pair
    counts, and the top-3 preview query, all through one .select()."""

    def __init__(self, docs: list[dict]):
        self.docs = docs

    def select(self, tokens, *, q_op="AND", rows=10, fl=None, **_kwargs):
        words = {str(t).casefold() for t in tokens}
        matched = [d for d in self.docs if words <= d["words"]]
        if rows == 0:
            return {"response": {"numFound": len(matched)}}
        return {"response": {"docs": [
            {"title": d.get("title", ""), "publisher": d.get("publisher", ""),
             "description": d.get("description", "")}
            for d in matched[:rows]
        ]}}


class _FakeChatResponse:
    def __init__(self, content: str):
        self.message = SimpleNamespace(content=content)


class _FakeLLM:
    """Routes on a marker unique to each system prompt; each pick call consumes
    the next scripted response in order."""

    def __init__(self, decompose_json: str, pick_jsons: list[str]):
        self.decompose_json = decompose_json
        self.pick_jsons = list(pick_jsons)
        self.calls: list[str] = []

    def chat(self, messages):
        system = messages[0].content
        self.calls.append(system)
        if "distinct tables the question needs" in system:
            return _FakeChatResponse(self.decompose_json)
        return _FakeChatResponse(self.pick_jsons.pop(0))


DOCS = [
    {"words": {"organogram", "office", "rail", "road", "november", "2021"},
     "title": "2021-11 Organogram", "publisher": "Office of Rail and Road", "description": "staff roles"},
    {"words": {"organogram", "transport", "focus", "january", "2022"},
     "title": "2022-01 Organogram", "publisher": "Transport Focus", "description": "staff roles"},
]
TABLE_WORDS = {f"{d['title']}.parquet": d["words"] for d in DOCS}


def _manager(monkeypatch, tmp_path, *, discovery=None, llm=None, files=None):
    solr = _FakeSolr(DOCS)
    service = _FakeRetrievalService(TABLE_WORDS)
    monkeypatch.setattr(search_tools, "get_table_retrieval_service", lambda *_a, **_k: service)
    state = tools_p12.P12State()
    manager = tools_p12.Phase12ToolsManager(
        state, solr, files if files is not None else list(TABLE_WORDS), tmp_path,
        question="What is the total junior FTE across the Office of Rail and Road "
                  "November 2021 and Transport Focus January 2022 organograms?",
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
        discovery_config=discovery or DiscoveryConfig(),
        llm=llm,
    )
    return state, manager, service


def test_flag_off_uses_the_plain_single_shot_search(monkeypatch, tmp_path):
    llm = _FakeLLM(decompose_json="{}", pick_jsons=[])
    discovery = DiscoveryConfig(keyword_search="single")
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=llm)

    manager.search_keyword_concepts(["organogram", "office", "rail", "road"])

    assert llm.calls == []  # never touched
    assert len(service.retrieves) == 1


def test_llm_none_falls_back_to_plain_search_even_with_flag_on(monkeypatch, tmp_path):
    discovery = DiscoveryConfig(keyword_search="decompose_preview")
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=None)

    response = manager.search_keyword_concepts(["organogram", "office", "rail", "road"])

    assert len(service.retrieves) == 1
    assert "Candidates in retrieval order" in response


def test_decompose_splits_and_merges_pools_from_two_sub_questions(monkeypatch, tmp_path):
    decompose_json = json.dumps({"tables": [
        {"detail": "ORR Nov 2021", "request": "organogram Office of Rail and Road November 2021"},
        {"detail": "Transport Focus Jan 2022", "request": "organogram Transport Focus January 2022"},
    ]})
    pick_jsons = [json.dumps({"picks": [1]}), json.dumps({"picks": [1]})]
    llm = _FakeLLM(decompose_json, pick_jsons)
    discovery = DiscoveryConfig(keyword_search="decompose_preview")
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=llm)

    response = manager.search_keyword_concepts(["organogram"])

    assert "2021-11 Organogram.parquet" in response
    assert "2022-01 Organogram.parquet" in response
    assert "Searched (split by table): table 1:" in response
    assert len(service.retrieves) == 2  # one real AND search per sub-question's pick
    # exactly one decompose call + one pick call per sub-question
    assert sum("distinct tables the question needs" in c for c in llm.calls) == 1
    assert len(llm.calls) == 3


def test_decompose_preview_search_still_rejects_a_known_banned_subset(monkeypatch, tmp_path):
    llm = _FakeLLM(decompose_json="{}", pick_jsons=[])
    discovery = DiscoveryConfig(keyword_search="decompose_preview")
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=llm)
    state.failed_keyword_combinations = [frozenset({"organogram", "office"})]

    response = manager.search_keyword_concepts(["organogram", "office", "extra"])

    assert response.startswith("Search rejected before retrieval")
    assert llm.calls == []
    assert service.retrieves == []


def test_decompose_falls_back_to_plain_search_when_nothing_is_pickable(monkeypatch, tmp_path):
    # Every sub-question word is unknown to the index, so no candidate subset
    # is ever found and the pipeline must fall back to a plain search.
    decompose_json = json.dumps({"tables": [{"detail": "x", "request": "xyzzy plugh unindexed nonsense"}]})
    llm = _FakeLLM(decompose_json, pick_jsons=[])
    discovery = DiscoveryConfig(keyword_search="decompose_preview")
    state, manager, service = _manager(monkeypatch, tmp_path, discovery=discovery, llm=llm)

    response = manager.search_keyword_concepts(["organogram", "office", "rail", "road"])

    assert "Candidates in retrieval order" in response
    assert len(service.retrieves) == 1
    assert service.retrieves[0] == sorted({"organogram", "office", "rail", "road"})
