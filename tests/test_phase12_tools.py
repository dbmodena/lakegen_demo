from lakegen.agent_tools import tools_p12
from lakegen.agent_tools.tools_p12 import P12State, Phase12ToolsManager
from lakegen.retrieval import RetrievalConfig, RetrievalHit


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
