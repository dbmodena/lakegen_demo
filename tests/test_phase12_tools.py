import pytest

from lakegen.agent_tools import tools_p12
from lakegen.agent_tools.tools_p12 import P12State, Phase12ToolsManager
from lakegen.agent_tools.tools_p2 import Phase2JudgeToolsManager
from lakegen.retrieval import (
    EmbeddingGenerationError,
    RetrievalConfig,
    RetrievalHit,
    RetrievalMode,
)


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


def test_search_returns_bounded_schema_preview_in_solr_order(
    monkeypatch, tmp_path
):
    columns = [
        {
            "name": f"column_{index}",
            "type": "string",
            "description": f"Description for column {index}",
        }
        for index in range(15)
    ]

    class FakeService:
        def retrieve(self, **_kwargs):
            return [
                RetrievalHit(
                    document={
                        "resource_id": "wide-table",
                        "title": "Wide table",
                        "description": "A useful dataset description.",
                        "tags": ["schools", "connectivity"],
                        "columns": columns,
                    },
                    score=1.0,
                    rank=1,
                    semantic_rank=1,
                )
            ]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    manager = Phase12ToolsManager(
        P12State(), object(), ["wide-table.parquet"], tmp_path,
        question="School connectivity",
        retrieval_config=RetrievalConfig(mode=RetrievalMode.SEMANTIC, top_k=10),
    )

    result = manager.search_solr("schools")

    assert "Candidates in Solr order" in result
    assert "Candidate 1 (Solr rank 1)" in result
    assert "Description: A useful dataset description." in result
    assert "Indexed schema preview: 12 of 15 columns" in result
    assert "column_0 [string]" in result
    assert "column_11 [string]" in result
    assert "column_12" not in result
    assert "3 additional columns omitted" in result


def test_unified_search_runs_once_and_reuses_the_same_candidates(
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
    assert second.startswith("Search skipped: the configured retrieval")
    assert "generic.parquet" in second
    assert state.best_ranks == {"generic.parquet": 1}
    assert len(state.search_attempts) == 1
    assert repeated.startswith("Search skipped")
    assert calls == [["school"]]


def test_configured_search_contract_is_mode_neutral_while_semantic_uses_question(
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
    assert repeated.startswith("Search skipped: the configured retrieval")
    assert calls == [("Which school has the highest bandwidth?", [], 15)]
    assert state.used_keywords == ["invented", "keyword"]
    assert "Provide 1-2 concise dataset concepts" in description


def test_semantic_embedding_failure_is_labeled_and_not_retried_by_agent(
    monkeypatch, tmp_path
):
    calls = []

    class FailingService:
        def retrieve(self, **_kwargs):
            calls.append(1)
            error = EmbeddingGenerationError("failed after 3 attempts")
            error.__cause__ = RuntimeError("unsupported value: NaN")
            raise error

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FailingService()
    )
    manager = Phase12ToolsManager(
        P12State(),
        object(),
        [],
        tmp_path,
        question="Which schools have 10.0 mbps?",
        retrieval_config=RetrievalConfig(mode=RetrievalMode.SEMANTIC),
    )

    first = manager.search_solr("school")
    repeated = manager.search_solr("different keywords")

    assert first.startswith("Error generating the configured retrieval representation")
    assert "must not be repeated" in first
    assert repeated.startswith("Configured retrieval skipped")
    assert calls == [1]


def test_search_tool_description_is_identical_for_all_retrieval_modes(tmp_path):
    descriptions = []
    for mode in RetrievalMode:
        manager = Phase12ToolsManager(
            P12State(), object(), [], tmp_path,
            question="Which tables are relevant?",
            retrieval_config=RetrievalConfig(mode=mode),
        )
        descriptions.append(manager.get_tools()[0].metadata.description)

    assert len(set(descriptions)) == 1
    assert "Provide 1-2 concise dataset concepts" in descriptions[0]


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_search_tool_allows_one_call_in_every_retrieval_mode(
    mode, monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(kwargs)
            return [_hit("table", 1, ["value"])]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    manager = Phase12ToolsManager(
        P12State(), object(), ["table.parquet"], tmp_path,
        question="Count road incidents",
        retrieval_config=RetrievalConfig(mode=mode),
    )

    first = manager.search_solr("road incidents")
    second = manager.search_solr("traffic crashes")

    assert "table.parquet" in first
    assert second.startswith("Search skipped: the configured retrieval")
    assert len(calls) == 1
    assert calls[0]["question"] == "Count road incidents"
    expected_concepts = (
        []
        if mode in (RetrievalMode.SEMANTIC, RetrievalMode.PNEUMA)
        else ["road", "incidents"]
    )
    assert calls[0]["keywords"] == expected_concepts


def test_inspect_columns_allows_two_attempts_but_reads_file_once(
    monkeypatch, tmp_path
):
    reads = []

    def fake_inspect(_directory, name):
        reads.append(name)
        return f"Schema for {name}"

    monkeypatch.setattr(tools_p12, "_inspect_columns", fake_inspect)
    state = P12State()
    state.all_candidates = ["table.parquet"]
    state.visible_candidate_count = 1
    manager = Phase12ToolsManager(state, object(), [], tmp_path)

    first = manager.inspect_columns(filename="table.parquet")
    second = manager.inspect_columns(
        filename="table.parquet", file_name="table.parquet"
    )
    third = manager.inspect_columns(file_name="table.parquet")

    assert first == "Schema for table.parquet"
    assert second.startswith("Cached inspection (attempt 2/2)")
    assert third.startswith("Inspection skipped")
    assert reads == ["table.parquet"]


def test_unified_selection_requires_every_selected_table_to_be_inspected(tmp_path):
    state = P12State()
    state.all_candidates = ["a.parquet", "b.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache["a.parquet"] = "Schema for a.parquet"
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    with pytest.raises(ValueError, match="inspect_columns is mandatory"):
        manager.confirm_unified_selection("both are needed", ["a.parquet", "b.parquet"])

    state.inspection_cache["b.parquet"] = "Schema for b.parquet"
    result = manager.confirm_unified_selection(
        "both are needed", ["a.parquet", "b.parquet"]
    )
    assert '"tables": "a.parquet, b.parquet"' in result


def test_unified_selection_blocks_proven_temporal_mismatch(tmp_path):
    state = P12State()
    state.all_candidates = ["history.parquet"]
    state.visible_candidate_count = 1
    state.inspection_cache["history.parquet"] = (
        "Schema for history.parquet:\nTemporal coverage:\n"
        "- Year: 2010 to 2018 (missing/unparseable 0.0%)"
    )
    manager = Phase12ToolsManager(
        state,
        object(),
        state.all_candidates,
        tmp_path,
        question="How many permits were filed in 2020?",
    )

    with pytest.raises(ValueError, match="outside the inspected temporal coverage"):
        manager.confirm_unified_selection("year is covered", ["history.parquet"])


def test_phase2_selection_requires_inspection(tmp_path):
    manager = Phase2JudgeToolsManager(["table.parquet"], tmp_path)

    with pytest.raises(ValueError, match="inspect_columns is mandatory"):
        manager.confirm_table_selection("relevant", ["table.parquet"])

    manager._inspection_cache["table.parquet"] = "Schema for table.parquet"
    assert "FINAL_PAYLOAD" in manager.confirm_table_selection(
        "relevant", ["table.parquet"]
    )


def test_phase2_selection_blocks_proven_temporal_mismatch(tmp_path):
    manager = Phase2JudgeToolsManager(
        ["history.parquet"],
        tmp_path,
        question="How many permits were filed in 2020?",
    )
    manager._inspection_cache["history.parquet"] = (
        "Schema for history.parquet:\nTemporal coverage:\n"
        "- Year: 2010 to 2018 (missing/unparseable 0.0%)"
    )

    with pytest.raises(ValueError, match="outside the inspected temporal coverage"):
        manager.confirm_table_selection("year is covered", ["history.parquet"])


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


def test_solr_candidate_order_is_preserved_without_schema_reranking(
    monkeypatch, tmp_path
):
    class FakeService:
        def retrieve(self, **_kwargs):
            return [
                _hit("solr-first", 1, ["unrelated"]),
                _hit("schema-match", 2, ["requested", "measure"]),
                _hit("solr-third", 3, ["other"]),
            ]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    state = P12State()
    manager = Phase12ToolsManager(
        state,
        object(),
        [
            "solr-first.parquet",
            "schema-match.parquet",
            "solr-third.parquet",
        ],
        tmp_path,
        question="requested measure",
        retrieval_config=RetrievalConfig(top_k=2),
    )

    result = manager.search_solr("requested")

    assert state.all_candidates == [
        "solr-first.parquet",
        "schema-match.parquet",
    ]
    assert result.index("solr-first.parquet") < result.index("schema-match.parquet")
    assert "solr-third.parquet" not in result


def test_unified_adaptive_candidates_reveal_ten_then_five(
    monkeypatch, tmp_path
):
    hits = [_hit(f"table-{index}", index, ["value"]) for index in range(1, 21)]

    class FakeService:
        def retrieve(self, **_kwargs):
            return hits

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    monkeypatch.setattr(
        tools_p12, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    files = [f"table-{index}.parquet" for index in range(1, 21)]
    state = P12State()
    manager = Phase12ToolsManager(
        state,
        object(),
        files,
        tmp_path,
        retrieval_config=RetrievalConfig(top_k=20),
    )

    initial = manager.search_solr("tables")
    assert "table-10.parquet" in initial
    assert "table-11.parquet" not in initial
    assert "10 additional ranked candidates" in initial
    assert manager.expand_candidates().startswith("Expansion blocked")

    assert manager.inspect_columns("table-1.parquet").startswith("Schema")
    expanded = manager.expand_candidates()
    assert "candidates 11-15" in expanded
    assert "table-15.parquet" in expanded
    assert "table-16.parquet" not in expanded
    assert "5 ranked candidates remain hidden" in expanded

    final_expansion = manager.expand_candidates()
    assert "Expansion limit reached" in final_expansion
    assert "Do not call expand_candidates again" in final_expansion


def test_unified_adaptive_inspection_limits_are_enforced(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tools_p12, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    state = P12State()
    state.all_candidates = [f"table-{index}.parquet" for index in range(1, 21)]
    state.visible_candidate_count = 10
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    for index in range(1, 4):
        assert manager.inspect_columns(f"table-{index}.parquet").startswith("Schema")
    assert manager.inspect_columns("table-4.parquet").startswith("Inspection blocked")

    assert "candidates 11-15" in manager.expand_candidates()
    assert manager.inspect_columns("table-4.parquet").startswith("Schema")
    assert manager.inspect_columns("table-5.parquet").startswith("Schema")
    assert manager.inspect_columns("table-6.parquet").startswith("Inspection blocked")


def test_phase2_adaptive_candidates_use_the_same_thresholds(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lakegen.agent_tools.tools_p2._inspect_columns",
        lambda _directory, name: f"Schema for {name}",
    )
    candidates = [f"table-{index}.parquet" for index in range(1, 21)]
    manager = Phase2JudgeToolsManager(candidates, tmp_path)

    assert len(manager.visible_candidates()) == 10
    assert manager.inspect_columns("table-11.parquet").startswith("Error:")
    for index in range(1, 4):
        assert manager.inspect_columns(f"table-{index}.parquet").startswith("Schema")
    assert manager.inspect_columns("table-4.parquet").startswith("Inspection blocked")
    first_expansion = manager.expand_candidates()
    assert "candidates 11-15" in first_expansion
    assert "5 ranked candidates remain hidden" in first_expansion
    assert len(manager.visible_candidates()) == 15

    final_expansion = manager.expand_candidates()
    assert "Expansion limit reached" in final_expansion
    assert "Do not call expand_candidates again" in final_expansion
