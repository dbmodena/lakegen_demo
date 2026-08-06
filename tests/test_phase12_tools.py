from lakegen.agent_tools import tools_p12
from lakegen.agent_tools.tools_p12 import P12State, Phase12ToolsManager
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode


def _hit(resource_id, rank, columns):
    return RetrievalHit(
        document={
            "resource_id": resource_id,
            "title": resource_id,
            "columns": [{"name": name} for name in columns],
        },
        score=1.0 / rank,
        rank=rank,
        lexical_rank=rank,
    )


def test_unified_search_accumulates_attempts_retains_best_rank_and_skips_duplicates(
    monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, *, keywords, **_kwargs):
            calls.append(list(keywords))
            if keywords == ["school"]:
                return [_hit("generic", 1, ["School Name"])]
            return [_hit("gold", 3, ["School Name", "Mbps Bandwidth"])]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    state = P12State()
    manager = Phase12ToolsManager(
        state,
        object(),
        ["generic.parquet", "gold.parquet"],
        tmp_path,
        question="Which school has Mbps bandwidth?",
        retrieval_config=RetrievalConfig(top_k=10),
    )

    first = manager.search_solr("school")
    second = manager.search_solr("bandwidth")
    repeated = manager.search_solr("bandwidth")

    assert "generic.parquet" in first
    assert "gold.parquet" in second and "generic.parquet" in second
    assert second.index("gold.parquet") < second.index("generic.parquet")
    assert state.best_ranks == {"generic.parquet": 1, "gold.parquet": 3}
    assert len(state.search_attempts) == 2
    assert repeated.startswith("Search skipped")
    assert calls == [["school"], ["bandwidth"]]


def test_semantic_search_caches_by_original_question_and_does_not_request_keywords(
    monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, *, question, keywords, **kwargs):
            calls.append((question, list(keywords), kwargs["top_k"]))
            return [_hit("gold", 1, ["Bandwidth"])]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    state = P12State()
    manager = Phase12ToolsManager(
        state,
        object(),
        ["gold.parquet"],
        tmp_path,
        question="Which school has the highest bandwidth?",
        retrieval_config=RetrievalConfig(
            mode=RetrievalMode.SEMANTIC, top_k=3
        ),
    )

    first = manager.search_solr("invented keyword")
    repeated = manager.search_solr("different invented keyword")
    description = manager.get_tools()[0].metadata.description

    assert "gold.parquet" in first
    assert repeated.startswith("Search skipped: the original question")
    assert calls == [("Which school has the highest bandwidth?", [], 15)]
    assert state.used_keywords == []
    assert "Do not invent or vary keywords" in description


def test_solr_candidates_are_mapped_and_deduplicated_before_final_top_k(
    monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(kwargs)
            return [
                _hit("missing-1", 1, ["noise"]),
                _hit("missing-2", 2, ["noise"]),
                _hit("local-a", 3, ["useful"]),
                _hit("local-b", 4, ["useful"]),
                _hit("local-c", 5, ["useful"]),
            ]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    state = P12State()
    manager = Phase12ToolsManager(
        state,
        object(),
        ["local-a.parquet", "local-b.parquet", "local-c.parquet"],
        tmp_path,
        question="useful tables",
        retrieval_config=RetrievalConfig(top_k=2),
    )

    result = manager.search_solr("useful")

    assert calls[0]["top_k"] == calls[0]["lexical_fetch_k"] == 15
    assert state.all_candidates == ["local-a.parquet", "local-b.parquet"]
    assert "local-a.parquet" in result and "local-b.parquet" in result
    assert "local-c.parquet" not in result
