import json

import pytest
import pandas as pd

from lakegen.agent_tools import tools_p12
from lakegen.agent_tools.tools_p12 import (
    P12State, Phase12ToolsManager, _normalize_semantic_plan,
    compile_semantic_plan_draft,
)


def test_inspected_candidates_accepts_structured_recovery_candidates():
    state = P12State()
    state.all_candidates = [
        {"file": "events.parquet", "score": 0.9},
        {"filename": "boroughs.parquet"},
        {"unexpected": "ignored"},
    ]
    state.inspection_cache = {
        "events.parquet": "columns: id, borough",
        "boroughs.parquet": "Error: unreadable",
    }

    assert state.inspected_candidates() == ["events.parquet"]
from lakegen.agent_tools import tools_p2
from lakegen.agent_tools.tools_p2 import Phase2JudgeToolsManager
from lakegen.keyword_memory import format_question_retrieval_memory
from lakegen.agent_tools.requirement_ledger import (
    _period_years, build_minimal_selection_fallback, build_requirement_ledger,
    requirement_ledger_blockers,
)
from lakegen.phases.phase12 import (
    _conservative_draft_from_requirements,
    _extract_plausible_json,
    _inspected_runtime_evidence,
    _reasoning_with_selection_plan,
    _recover_blocked_selection,
    _recover_minimal_selection_plan,
    _semantic_planner_prompt,
)
from lakegen.retrieval import (
    EmbeddingGenerationError,
    RetrievalConfig,
    RetrievalHit,
    RetrievalMode,
)


def _semantic_plan(table="a.parquet", measure_column="value"):
    return {
        "filters": [{
            "requirement": "year 2020", "table": table, "column": "year",
            "operator": "equals", "value": "2020", "evidence": "observed schema",
        }],
        "dimensions": [{
            "output": "district", "table": table, "column": "district",
            "evidence": "observed schema",
        }],
        "measures": [{
            "output": "average_value", "operation": "mean", "table": table,
            "columns": [measure_column], "evidence": "numeric inspected column",
        }],
        "joins": [], "ordering": [], "limit": None,
        "output_columns": ["district", "average_value"],
    }


def test_contract_first_selection_validates_and_records_semantic_bindings(tmp_path):
    pd.DataFrame({
        "year": [2020], "district": [1], "value": [2.0]
    }).to_parquet(tmp_path / "a.parquet")
    state = P12State()
    state.all_candidates = ["a.parquet"]
    state.visible_candidate_count = 1
    state.inspection_cache["a.parquet"] = "inspected schema"
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)
    kwargs = {
        "requirement_coverage": {
            "2020 district average": {
                "table": "a.parquet", "columns": ["year", "district", "value"],
            }
        },
        "table_roles": {"a.parquet": "fact records"},
        "semantic_plan": _semantic_plan(),
    }

    result = manager.confirm_unified_selection(
        "The table contains all required facts.", ["a.parquet"], **kwargs
    )

    assert "FINAL_PAYLOAD" in result
    assert state.selection_plan["semantic_plan"]["measures"][0]["operation"] == "mean"

    kwargs["semantic_plan"] = _semantic_plan(measure_column="invented")
    with pytest.raises(ValueError, match="column not found"):
        manager.confirm_unified_selection(
            "The table contains all required facts.", ["a.parquet"], **kwargs
        )


def test_selection_builds_shared_requirement_ledger_without_blocking_computation(tmp_path):
    pd.DataFrame({
        "year": [2020], "district": [1], "value": [2.0]
    }).to_parquet(tmp_path / "a.parquet")
    state = P12State()
    state.all_candidates = ["a.parquet"]
    state.visible_candidate_count = 1
    state.inspection_cache["a.parquet"] = "inspected schema"
    manager = Phase12ToolsManager(
        state, object(), state.all_candidates, tmp_path,
        question="What is the correlation by district in 2020?",
    )

    manager.confirm_unified_selection(
        "The table supplies the requested facts.", ["a.parquet"],
        requirement_coverage={
            "district and year 2020": {
                "table": "a.parquet", "columns": ["district", "year"],
            }
        },
        table_roles={"a.parquet": "fact records"},
        requirements={
            "grouping": ["district"], "measures": ["correlation"],
            "result_type": "number",
        },
        semantic_plan=_semantic_plan(),
    )

    ledger = state.selection_plan["requirement_ledger"]
    assert "requirement_ledger" not in state.selection_plan["coder_brief"]
    assert any(
        item["status"] == "bound" and item["request"] == "district and year 2020"
        for item in ledger
    )
    assert any(
        item["status"] == "computational"
        and item["kind"] == "derived_operation"
        and item["request"] == "correlation"
        and item["role"] == "final"
        for item in ledger
    )
    assert any(
        item["status"] == "bound"
        and item["kind"] == "temporal_scope"
        and item["request"] == "2020"
        for item in ledger
    )
    assert any(
        item["kind"] == "dimension"
        and item["request"] == "district"
        and item["role"] == "intermediate"
        for item in ledger
    )
    assert any(
        item["kind"] == "output"
        and item.get("shape") == "scalar"
        and item.get("role") == "final"
        for item in ledger
    )


def test_requirement_ledger_merges_bound_computation_and_avoids_inferred_gaps(tmp_path):
    manager = Phase12ToolsManager(
        P12State(), object(), [], tmp_path,
        question="Count partnered plazas in Brooklyn in 2023.",
    )
    ledger = manager._build_requirement_ledger(
        {
            "measure count partnered plazas": {
                "table": "plazas.parquet", "columns": ["PlazaName"],
            },
            "Brooklyn and year 2023": {
                "table": "plazas.parquet", "columns": ["Borough", "Year"],
            },
        },
        {"measures": [{
            "type": "count", "column": "PlazaName",
            "alias": "partnered_plaza_count",
        }]},
        [], None,
    )

    assert len(ledger) <= 10
    assert not any(item["status"] == "unresolved" for item in ledger)
    assert not any("{" in item["request"] for item in ledger)
    measure_items = [item for item in ledger if item["kind"] == "measure"]
    assert len(measure_items) == 1
    assert measure_items[0]["status"] == "bound"
    assert measure_items[0]["computation"] == "partnered_plaza_count count PlazaName"


def test_requirement_ledger_marks_per_group_as_intermediate_for_scalar_average(tmp_path):
    manager = Phase12ToolsManager(
        P12State(), object(), [], tmp_path,
        question="What was the average number of transaction records per block?",
    )
    ledger = manager._build_requirement_ledger(
        {}, {"grouping": ["Block"], "measures": ["transaction records"]},
        [], None,
    )

    assert next(item for item in ledger if item["kind"] == "dimension")["role"] == "intermediate"
    output = next(item for item in ledger if item["kind"] == "output")
    assert output["role"] == "final"
    assert output["shape"] == "scalar"


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


def test_default_search_retains_hidden_candidates_for_expansion(monkeypatch, tmp_path):
    class FakeService:
        def retrieve(self, **kwargs):
            assert kwargs["top_k"] == 20
            return [_hit(f"table-{i}", i, ["value"]) for i in range(1, 21)]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_: FakeService())
    monkeypatch.setattr(tools_p12, "_inspect_columns", lambda *_: "Schema: value")
    files = [f"table-{i}.parquet" for i in range(1, 21)]
    state = P12State()
    manager = Phase12ToolsManager(state, object(), files, tmp_path)
    manager.search_tables("values")
    assert len(state.all_candidates) == 20
    assert state.visible_candidate_count == 10
    manager.inspect_columns(files[0])
    assert "table-15.parquet" in manager.expand_candidates("value")
    assert state.visible_candidate_count == 15


def test_search_excludes_previously_rejected_tables(monkeypatch, tmp_path):
    class FakeService:
        def retrieve(self, **kwargs):
            return [_hit("bad", 1, ["value"]), _hit("good", 2, ["value"])]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_: FakeService())
    files = ["bad.parquet", "good.parquet"]
    state = P12State()
    state.excluded_tables = {"bad.parquet"}
    manager = Phase12ToolsManager(state, object(), files, tmp_path)

    result = manager.search_tables("value")

    assert "good.parquet" in result
    assert "bad.parquet" not in result
    assert state.all_candidates == ["good.parquet"]


def test_search_finding_only_banned_tables_is_empty_and_bans_its_words(
    monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(kwargs["keywords"])
            if "fresh" in kwargs["keywords"]:
                return [_hit("good", 1, ["value"])]
            return [_hit("bad", 1, ["value"]), _hit("worse", 2, ["value"])]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args, **_kwargs: FakeService()
    )
    state = P12State()
    state.excluded_tables = {"bad.parquet", "worse.parquet"}
    manager = Phase12ToolsManager(
        state, object(), ["bad.parquet", "worse.parquet", "good.parquet"], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    first = manager.search_tables("water licences")
    refused = manager.search_tables("water licences ni")
    recovered = manager.search_tables("fresh words")

    assert "No usable tables" in first
    assert "bad.parquet, worse.parquet" in first
    assert "{licences, water} is now banned for this question" in first
    assert "Candidate" not in first
    assert refused.startswith("Search rejected before retrieval")
    assert "found only tables already banned" in refused
    assert "good.parquet" in recovered
    assert calls == [["water", "licences"], ["fresh", "words"]]
    # Question-scoped: never joins the persisted cross-question banlist.
    assert state.banned_table_keyword_combinations == [frozenset({"water", "licences"})]
    assert state.failed_keyword_combinations == []
    # The banned-only search spends a zero-result retry, not a found attempt.
    assert state.search_call_outcomes() == [False, True]
    assert state.retrieval_memory_events[0] == {
        "outcome": "banned_tables_only",
        "terms": ["water", "licences"],
        "tables": ["bad.parquet", "worse.parquet"],
    }
    assert "found only tables already banned: bad.parquet, worse.parquet" in (
        format_question_retrieval_memory(state.retrieval_memory_events)
    )


def test_search_widens_fetch_by_excluded_count(monkeypatch, tmp_path):
    # Excluded candidates are filtered out of the same fixed-size fetch, so
    # without compensation every accumulated ban silently shrinks the visible
    # pool below top_k. The raw Solr fetch must widen by the ban count
    # (mirrors DIVIDED's fetch_k = config.top_k + len(excluded) in
    # phase2.py's _solr_and_search).
    captured_kwargs = []

    class FakeService:
        def retrieve(self, **kwargs):
            captured_kwargs.append(kwargs)
            return [_hit(f"table-{i}", i, ["value"]) for i in range(1, 21)]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_: FakeService())
    files = [f"table-{i}.parquet" for i in range(1, 21)]

    baseline_state = P12State()
    Phase12ToolsManager(baseline_state, object(), files, tmp_path).search_tables("values")
    baseline_fetch_k = captured_kwargs[-1]["top_k"]

    excluded_state = P12State()
    excluded_state.excluded_tables = {"bad-1", "bad-2", "bad-3"}
    Phase12ToolsManager(excluded_state, object(), files, tmp_path).search_tables("values")
    widened_fetch_k = captured_kwargs[-1]["top_k"]

    assert widened_fetch_k == baseline_fetch_k + 3
    assert captured_kwargs[-1]["lexical_fetch_k"] == widened_fetch_k


def test_search_carries_forward_previously_kept_tables(monkeypatch, tmp_path):
    class FakeService:
        def retrieve(self, **kwargs):
            return [_hit("fresh", 1, ["value"])]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_: FakeService())
    files = ["fresh.parquet", "kept.parquet"]
    state = P12State()
    state.carried_tables = ["kept.parquet"]
    state.carried_metadata = {"kept.parquet": {"title": "Kept Table"}}
    manager = Phase12ToolsManager(state, object(), files, tmp_path)

    result = manager.search_tables("value")

    # Carried tables re-surface even though this round's own retrieval
    # never returned them, and lead the ranking the same way DIVIDED's
    # carried_tables do.
    assert "kept.parquet" in result
    assert "Kept Table" in result


def test_carried_table_does_not_need_re_inspection(monkeypatch, tmp_path):
    # A table carried forward with its cached inspect_columns text pre-seeded
    # onto inspection_cache must count as already inspected -- otherwise
    # every round re-burns part of its bounded inspection budget re-proving
    # something the previous round already established.
    class FakeService:
        def retrieve(self, **kwargs):
            return [_hit("fresh", 1, ["value"])]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_: FakeService())
    inspect_calls = []
    monkeypatch.setattr(
        tools_p12, "_inspect_columns",
        lambda _directory, name: inspect_calls.append(name) or f"Schema for {name}",
    )
    files = ["fresh.parquet", "kept.parquet"]
    state = P12State()
    state.carried_tables = ["kept.parquet"]
    state.carried_metadata = {"kept.parquet": {"title": "Kept Table"}}
    state.inspection_cache = {"kept.parquet": "Schema for kept.parquet (from a prior round)"}
    manager = Phase12ToolsManager(state, object(), files, tmp_path)

    manager.search_tables("value")

    assert "kept.parquet" in state.inspected_candidates()
    assert inspect_calls == []
    assert state.all_candidates[0] == "kept.parquet"


def test_reject_unified_selection_splits_inspected_candidates_into_keep_and_skip(tmp_path):
    state = P12State()
    state.all_candidates = ["a.parquet", "b.parquet", "c.parquet"]
    state.inspection_cache = {
        "a.parquet": "Schema for a.parquet",
        "b.parquet": "Schema for b.parquet",
        "c.parquet": "Schema for c.parquet",
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    manager.reject_unified_selection(
        "b covers the count but not the district breakdown",
        "look for a district-level table",
        ban_tables={"c.parquet": "has no district or count column at all"},
    )

    # Only the explicitly-banned candidate is excluded; everything else
    # inspected is kept by default, even without an explicit endorsement.
    assert state.rejection_keep_tables == ["a.parquet", "b.parquet"]
    assert state.rejection_skip_tables == ["c.parquet"]


def test_reject_unified_selection_rejects_unjustified_ban_tables(tmp_path):
    state = P12State()
    state.all_candidates = ["a.parquet"]
    state.inspection_cache = {"a.parquet": "Schema for a.parquet"}
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    with pytest.raises(ValueError, match="concrete evidence"):
        manager.reject_unified_selection(
            "a.parquet does not contain the required organisation",
            "search for the correct organisation",
            ban_tables={"a.parquet": "bad"},
        )


def test_partition_filters_preserve_unique_exact_table_bindings(tmp_path):
    pd.DataFrame({"Trip_distance": [40]}).to_parquet(tmp_path / "early.parquet")
    pd.DataFrame({"trip_distance": [50]}).to_parquet(tmp_path / "late.parquet")
    tables = ["early.parquet", "late.parquet"]
    manager = Phase12ToolsManager(P12State(), object(), tables, tmp_path)
    brief = manager._build_coder_brief(
        tables, {}, {"filters": [
            {"column": "Trip_distance", "operator": ">", "value": 30},
            {"column": "trip_distance", "operator": ">", "value": 30},
        ]}, "aggregate_separately", {}, None,
    )
    assert [item["table"] for item in brief["filters"]] == tables
    assert brief["selected_columns"] == {
        "early.parquet": ["Trip_distance"], "late.parquet": ["trip_distance"],
    }
    assert not brief["normalization_errors"]


def test_same_column_in_multiple_tables_requires_explicit_binding(tmp_path):
    for name in ("a.parquet", "b.parquet"):
        pd.DataFrame({"value": [40]}).to_parquet(tmp_path / name)
    tables = ["a.parquet", "b.parquet"]
    manager = Phase12ToolsManager(P12State(), object(), tables, tmp_path)
    brief = manager._build_coder_brief(tables, {}, {"filters": [
        {"column": "value", "operator": ">", "value": 30},
        {"table": "b.parquet", "column": "value", "operator": ">", "value": 30},
    ]}, "aggregate_separately", {}, None)
    assert "table" not in brief["filters"][0]
    assert brief["filters"][1]["table"] == "b.parquet"


def test_requirement_ledger_keeps_distinct_periods_and_sources():
    coverage = {
        "year 2014 trips": {"table": "a.parquet", "columns": ["pickup"]},
        "year 2022 trips": {"table": "b.parquet", "columns": ["pickup"]},
        "completed count north": {"table": "north.parquet", "columns": ["status"]},
        "completed count south": {"table": "south.parquet", "columns": ["status"]},
    }
    ledger = build_requirement_ledger("Compare annual trips", coverage, {}, [])
    for request, evidence in coverage.items():
        assert next(item for item in ledger if item["request"] == request)["evidence"] == evidence


def test_fallback_does_not_turn_narrative_into_a_join():
    plan, _ = build_minimal_selection_fallback(
        ["a.parquet", "b.parquet"], "Could merge on a shared key if one exists."
    )
    assert plan["combination_strategy"] == "unspecified"


def test_search_returns_bounded_schema_preview_in_retrieval_order(
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

    result = manager.search_tables("schools")

    assert "Candidates in retrieval order" in result
    assert "Candidate 1 (retrieval rank 1)" in result
    assert "Description: A useful dataset description." in result
    assert "Indexed schema preview: 12 of 15 columns" in result
    assert "column_0 [string]" in result
    assert "column_11 [string]" in result
    assert "column_12" not in result
    assert "3 additional columns omitted" in result


def test_unified_search_allows_one_initial_search_then_guided_expansion(
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

    first = manager.search_tables("school")
    second = manager.search_tables("bandwidth")
    repeated = manager.search_tables("bandwidth")

    assert "generic.parquet" in first
    assert second.startswith("Search limit reached")
    assert repeated.startswith("Search limit reached")
    assert state.best_ranks == {"generic.parquet": 1}
    assert len(state.search_attempts) == 1
    assert calls == [["school"]]

    limited = manager.search_tables("third distinct concept")
    assert limited.startswith("Search limit reached")


def test_search_refinement_is_blocked_after_schema_inspection(monkeypatch, tmp_path):
    class FakeService:
        def retrieve(self, **_kwargs):
            return [_hit("table", 1, ["value"])]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args: FakeService()
    )
    monkeypatch.setattr(
        tools_p12, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    manager = Phase12ToolsManager(
        P12State(), object(), ["table.parquet"], tmp_path,
        question="value", retrieval_config=RetrievalConfig(top_k=10),
    )
    manager.search_tables("first concept")
    manager.inspect_columns("table.parquet")

    assert manager.search_tables("new concept").startswith(
        "Search refinement blocked"
    )


@pytest.mark.parametrize("mode", [RetrievalMode.SEMANTIC, RetrievalMode.PNEUMA])
def test_configured_search_contract_is_mode_neutral_while_question_only_modes_use_question(
    monkeypatch, tmp_path, mode
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
        retrieval_config=RetrievalConfig(mode=mode, top_k=3),
    )

    first = manager.search_tables("invented keyword")
    repeated = manager.search_tables("invented keyword")
    different = manager.search_tables("different invented keyword")
    description = manager.get_tools()[0].metadata.description

    assert "gold.parquet" in first
    assert calls == [("Which school has the highest bandwidth?", [], 15)]
    assert state.used_keywords == ["invented", "keyword"]
    assert "Pass `concepts` as a list of 2-4 single-word dataset concepts" in description
    # Deduplicated by the agent's own concepts, as in keyword modes: a different
    # search is not "identical", it is over the attempt limit.
    assert repeated.startswith("Search skipped: identical concepts")
    assert different.startswith("Search limit reached (1 attempt(s))")


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

    first = manager.search_tables("school")
    repeated = manager.search_tables("different keywords")

    assert first.startswith("Error generating the configured retrieval representation")
    assert "must not be repeated" in first
    assert repeated.startswith("Configured retrieval skipped")
    assert calls == [1]


def _search_description(tmp_path, mode):
    manager = Phase12ToolsManager(
        P12State(), object(), [], tmp_path,
        question="Which tables are relevant?",
        retrieval_config=RetrievalConfig(mode=mode),
    )
    return manager.get_tools()[0].metadata.description


def test_search_tool_description_is_identical_for_all_topic_based_modes(tmp_path):
    """The agent is not told which retriever runs, so it cannot adapt to it.

    Pneuma-Seeker is the exception because it scans for named entities, so
    asking that arm for dataset topics would handicap it by construction.
    """
    descriptions = {
        _search_description(tmp_path, mode)
        for mode in RetrievalMode
        if not (mode.value_keywords or mode.verbatim_entities)
    }

    assert len(descriptions) == 1
    assert "Pass `concepts` as a list of 2-4 single-word dataset concepts" in descriptions.pop()


def test_pneuma_seeker_asks_the_agent_for_verbatim_entities(monkeypatch, tmp_path):
    description = _search_description(tmp_path, RetrievalMode.PNEUMA_SEEKER)
    assert "exactly as it appears in the question" in description
    assert "dataset concepts" not in description
    assert "pneuma" not in description.lower()  # still never names the retriever

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
        question="Ferry trips on the East River",
        retrieval_config=RetrievalConfig(mode=RetrievalMode.PNEUMA_SEEKER),
    )
    assert manager.get_tools()[0].metadata.name == "search_tables"

    result = manager.search_table_entities(["East  River"])

    assert "table.parquet" in result
    assert calls[0]["entities"] == ["East River"]
    assert calls[0]["keywords"] == []  # Pneuma ranks the question itself


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_search_tool_allows_one_initial_call_for_every_backend(
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

    first = manager.search_tables("road incidents")
    second = manager.search_tables("traffic crashes")

    assert "table.parquet" in first
    assert len(calls) == 1
    assert calls[0]["question"] == "Count road incidents"
    expected_concepts = (
        []
        if mode
        in (RetrievalMode.SEMANTIC, RetrievalMode.PNEUMA, RetrievalMode.PNEUMA_SEEKER)
        else ["road", "incidents"]
    )
    assert calls[0]["keywords"] == expected_concepts


def test_keyword_zero_results_are_banned_and_force_a_new_and_query(
    monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(kwargs)
            if kwargs["keywords"] in (["alpha", "beta"], ["alpha"]):
                return []
            return [_hit("table", 1, ["gamma"])]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args, **_kwargs: FakeService()
    )
    state = P12State()
    manager = Phase12ToolsManager(
        state, object(), ["table.parquet"], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    first = manager.search_tables("alpha beta")
    narrower = manager.search_tables("alpha")
    blocked = manager.search_tables("alpha gamma")
    recovered = manager.search_tables("gamma")

    assert "added to the zero-result banlist" in first
    assert "added to the zero-result banlist" in narrower
    assert "known zero-result keyword subset {alpha}" in blocked
    assert "table.parquet" in recovered
    assert [call["q_op"] for call in calls] == ["AND", "AND", "AND"]
    assert [sorted(item) for item in state.failed_keyword_combinations] == [["alpha"]]
    assert state.keyword_history == [["alpha", "beta"], ["alpha"], ["gamma"]]


def test_keyword_failed_subset_rejection_does_not_spend_retry_budget(
    monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(kwargs["keywords"])
            return []

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args, **_kwargs: FakeService()
    )
    manager = Phase12ToolsManager(
        P12State(), object(), [], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    manager.search_tables("alpha")
    for suffix in ("beta", "gamma", "delta"):
        assert manager.search_tables(f"alpha {suffix}").startswith(
            "Search rejected before retrieval"
        )
    manager.search_tables("epsilon")
    manager.search_tables("zeta")
    limited = manager.search_tables("eta")

    assert calls == [["alpha"], ["epsilon"], ["zeta"]]
    assert limited.startswith("Search limit reached")


def test_hybrid_zero_results_use_keyword_banlist_and_retry_budget(
    monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(kwargs["keywords"])
            return []

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_args, **_kwargs: FakeService()
    )
    state = P12State()
    manager = Phase12ToolsManager(
        state, object(), [], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.HYBRID),
    )

    first = manager.search_tables("alpha beta")
    narrower = manager.search_tables("alpha")
    blocked = manager.search_tables("alpha gamma")
    manager.search_tables("delta")
    limited = manager.search_tables("epsilon")

    assert "added to the zero-result banlist" in first
    assert "added to the zero-result banlist" in narrower
    assert "known zero-result keyword subset {alpha}" in blocked
    assert [sorted(item) for item in state.failed_keyword_combinations] == [
        ["alpha"], ["delta"]
    ]
    assert calls == [["alpha", "beta"], ["alpha"], ["delta"]]
    assert limited.startswith("Search limit reached")


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


def test_unified_selection_records_agentic_plan_without_blocking_advisories(tmp_path):
    state = P12State()
    state.all_candidates = ["a.parquet", "b.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache = {
        "a.parquet": "Schema for a.parquet",
        "b.parquet": "Schema for b.parquet",
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    result = manager.confirm_unified_selection(
        "both yearly partitions are required",
        ["a.parquet", "b.parquet"],
        requirement_coverage={
            "2019 records": {"table": "a.parquet", "columns": ["year", "value"]},
            "2020 records": {"table": "b.parquet", "columns": ["year", "value"]},
        },
        table_roles={
            "a.parquet": "2019 partition",
            "b.parquet": "2020 partition",
        },
        combination_strategy="concat_partitions",
    )

    payload = json.loads(result.split("FINAL_PAYLOAD: ", 1)[1])
    assert payload["advisories"] == []
    assert payload["selection_plan"]["combination_strategy"] == "concat_partitions"
    assert state.selection_plan == payload["selection_plan"]


def test_unified_selection_advisories_are_non_blocking(tmp_path):
    state = P12State()
    state.all_candidates = ["a.parquet", "b.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache = {
        "a.parquet": "Schema for a.parquet",
        "b.parquet": "Schema for b.parquet",
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    result = manager.confirm_unified_selection(
        "use both", ["a.parquet", "b.parquet"]
    )

    payload = json.loads(result.split("FINAL_PAYLOAD: ", 1)[1])
    assert payload["tables"] == "a.parquet, b.parquet"
    assert any("without an explicit role" in item for item in payload["advisories"])
    assert any("strategy is single_table" in item for item in payload["advisories"])


def test_missing_agentic_plan_is_recovered_as_flexible_coder_context():
    plan, advisories = _recover_minimal_selection_plan(
        ["events.parquet", "boroughs.parquet"],
        "Join the event records to the borough lookup using the shared key.",
    )
    context = _reasoning_with_selection_plan("selected", plan, advisories)

    assert plan["combination_strategy"] == "unspecified"
    assert plan["recovered_from_existing_discovery_context"] is True
    assert set(plan["table_roles"]) == {"events.parquet", "boroughs.parquet"}
    assert "coder may complete computational details" in context


def test_divided_and_unified_recovery_use_the_same_minimal_fallback():
    selected = ["events.parquet", "boroughs.parquet"]
    reasoning = "Join the event records to the borough lookup using the shared key."

    divided_plan, divided_advisories = build_minimal_selection_fallback(
        selected, reasoning
    )
    unified_plan, unified_advisories = _recover_minimal_selection_plan(
        selected, reasoning
    )

    assert divided_plan == unified_plan
    assert divided_advisories == unified_advisories
    assert divided_plan["recovered_from_existing_discovery_context"] is True


def test_recovery_evidence_serializes_cached_inspections_without_external_context():
    state = P12State()
    state.all_candidates = ["events.parquet", "broken.parquet"]
    state.inspection_cache = {
        "events.parquet": "Schema for events.parquet: borough, category",
        "broken.parquet": "Error: unreadable",
    }

    evidence = json.loads(_inspected_runtime_evidence(state))

    assert evidence == {
        "events.parquet": "Schema for events.parquet: borough, category"
    }


def test_semantic_plan_normalizes_reasonable_aliases_and_join_shape():
    normalized = _normalize_semantic_plan({
        "filters": [{"column": "year", "operator": "year_eq", "value": 2020}],
        "measures": [{"column": "value", "operation": "average"}],
        "dimensions": [], "temporal_filters": [],
        "joins": [{
            "left_table": "facts.parquet", "right_table": "lookup.parquet",
            "left_key": "district_id", "right_key": "id", "how": "left",
        }],
    }, ["facts.parquet", "lookup.parquet"])
    assert normalized["filters"][0]["operator"] == "equals"
    assert normalized["filters"][0]["value"] == "2020"
    assert normalized["measures"][0]["operation"] == "mean"
    assert normalized["measures"][0]["columns"] == ["value"]
    assert normalized["joins"][0]["tables"] == ["facts.parquet", "lookup.parquet"]
    assert normalized["joins"][0]["keys"] == {
        "facts.parquet": "district_id", "lookup.parquet": "id"
    }


def test_draft_compiler_adds_only_runtime_verifiable_fields():
    compiled = compile_semantic_plan_draft(
        {
            "filters": [["status", "equals", "completed"]],
            "dimensions": [["borough_name", "borough"]],
            "measures": [["event_count", "count", ["id"]]],
            "ordering": [["event_count", "descending"]], "limit": 3,
        },
        ["events.parquet"], {"events.parquet": "fact records"},
        {"events.parquet": {"status", "borough", "id"}},
    )
    assert compiled["measures"][0]["operation"] == "count_rows"
    assert compiled["measures"][0]["table"] == "events.parquet"
    assert compiled["measures"][0]["output"] == "event_count"
    assert compiled["dimensions"][0]["evidence"].startswith(
        "Inspected runtime schema"
    )
    assert compiled["output_columns"] == ["borough_name", "event_count"]


def test_draft_compiler_refuses_to_infer_table_for_multi_table_binding():
    with pytest.raises(ValueError, match="must name a table"):
        compile_semantic_plan_draft(
            {"measures": [["row_count", "count_rows", []]]},
            ["a.parquet", "b.parquet"],
            {"a.parquet": "facts", "b.parquet": "lookup"},
            {"a.parquet": {"id"}, "b.parquet": {"id"}},
        )


def test_draft_compiler_normalizes_and_evidences_join():
    compiled = compile_semantic_plan_draft(
        {
            "measures": [{
                "output": "row_count", "operation": "count_rows", "columns": [],
                "table": "facts.parquet",
            }],
            "joins": [{
                "left_table": "facts.parquet", "right_table": "lookup.parquet",
                "left_key": "district_id", "right_key": "id", "how": "left",
            }],
        },
        ["facts.parquet", "lookup.parquet"],
        {"facts.parquet": "facts", "lookup.parquet": "lookup"},
        {"facts.parquet": {"district_id"}, "lookup.parquet": {"id"}},
    )
    assert compiled["joins"][0]["keys"] == {
        "facts.parquet": "district_id", "lookup.parquet": "id"
    }
    assert "facts.parquet.district_id" in compiled["joins"][0]["evidence"]


def test_selection_only_is_followed_by_compiled_draft(tmp_path):
    pd.DataFrame({"borough": ["Queens"], "id": [1]}).to_parquet(
        tmp_path / "events.parquet"
    )
    state = P12State()
    state.all_candidates = ["events.parquet"]
    state.visible_candidate_count = 1
    state.inspection_cache["events.parquet"] = "Schema: borough, id"
    manager = Phase12ToolsManager(
        state, object(), state.all_candidates, tmp_path, question="Count by borough"
    )
    selection = manager.confirm_unified_selection(
        "Events contain borough records.", ["events.parquet"],
        requirement_coverage={
            "group by borough": {"table": "events.parquet", "columns": ["borough"]}
        },
        table_roles={"events.parquet": "fact records"},
        requirements={
            "grouping": ["borough"], "measures": ["count rows"],
            "filters": [], "ordering": "row_count descending", "limit": 3,
        },
    )
    selection_plan = json.loads(
        selection.split("FINAL_PAYLOAD: ", 1)[1]
    )["selection_plan"]
    assert "semantic_plan" not in selection_plan
    brief = selection_plan["coder_brief"]
    assert brief == {
        "tables": ["events.parquet"],
        "selected_columns": {"events.parquet": ["borough"]},
        "task": {
            "grouping": ["borough"], "measures": ["count rows"],
            "filters": [], "ordering": "row_count descending", "limit": 3,
        },
        "filters": [], "operations": ["count rows"],
        "result_type": "auto", "ordering": "row_count descending",
        "limit": 3, "joins": [], "normalization_errors": [],
        "temporal_filters": [], "dimensions": [{
            "table": "events.parquet", "column": "borough", "output": "borough",
        }],
        "measures": ["count rows"], "output_columns": [],
        "null_policy": "", "table_roles": {"events.parquet": "fact records"},
    }
    planned = manager.submit_semantic_plan_draft({
        "filters": [], "dimensions": [["borough", "borough"]],
        "measures": [["row_count", "count_rows", []]],
        "ordering": [["row_count", "descending"]], "limit": 3,
    })
    payload = json.loads(planned.split("FINAL_PAYLOAD: ", 1)[1])
    assert payload["selection_plan"]["semantic_plan"]["measures"][0]["operation"] == "count_rows"
    assert state.semantic_planner_attempts == 1


def test_coder_brief_normalizes_annotated_join_columns_without_fuzzy_matching(tmp_path):
    pd.DataFrame({"Partner": ["Community A"], "PlazaName": ["One"]}).to_parquet(
        tmp_path / "plazas.parquet"
    )
    pd.DataFrame({"Organization name": ["Community A"]}).to_parquet(
        tmp_path / "organizations.parquet"
    )
    state = P12State()
    state.all_candidates = ["plazas.parquet", "organizations.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache = {
        "plazas.parquet": "Schema: Partner, PlazaName",
        "organizations.parquet": "Schema: Organization name",
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)
    result = manager.confirm_unified_selection(
        "Join explicitly chosen organization names to plaza partners.",
        state.all_candidates,
        requirement_coverage={
            "join organization": {
                "table": "join",
                "columns": [
                    "partner (plazas.parquet)",
                    "Organization name (organizations.parquet)",
                ],
            },
        },
        table_roles={"plazas.parquet": "facts", "organizations.parquet": "lookup"},
        combination_strategy="join",
    )
    brief = json.loads(result.split("FINAL_PAYLOAD: ", 1)[1])["selection_plan"]["coder_brief"]
    assert brief["selected_columns"] == {
        "plazas.parquet": ["Partner"],
        "organizations.parquet": ["Organization name"],
    }
    assert brief["joins"] == [{
        "tables": ["plazas.parquet", "organizations.parquet"],
        "keys": {
            "plazas.parquet": "Partner",
            "organizations.parquet": "Organization name",
        },
        "how": "inner",
    }]
    assert brief["normalization_errors"] == []


def test_coder_brief_does_not_semantically_expand_short_column_names(tmp_path):
    pd.DataFrame({
        "Home Broadband Adoption (Percentage of Households)": [70.0]
    }).to_parquet(tmp_path / "connectivity.parquet")
    state = P12State()
    state.all_candidates = ["connectivity.parquet"]
    state.visible_candidate_count = 1
    state.inspection_cache["connectivity.parquet"] = "Schema inspected"
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)
    result = manager.confirm_unified_selection(
        "Use broadband adoption.", ["connectivity.parquet"],
        requirement_coverage={
            "measure": {
                "table": "connectivity.parquet",
                "columns": ["Home Broadband Adoption"],
            },
        },
        table_roles={"connectivity.parquet": "facts"},
    )
    brief = json.loads(result.split("FINAL_PAYLOAD: ", 1)[1])["selection_plan"]["coder_brief"]
    assert brief["selected_columns"] == {"connectivity.parquet": []}
    assert "not one unambiguous column" in brief["normalization_errors"][0]


def test_textual_json_is_recovered_but_still_requires_validation():
    recovered = _extract_plausible_json(
        'I corrected it: {"draft":{"measures":[["count","count_rows",[]]]}}'
    )
    assert recovered == {"draft": {"measures": [["count", "count_rows", []]]}}


def test_conservative_fallback_requires_exact_runtime_columns():
    draft = _conservative_draft_from_requirements({
        "grouping": ["borough"], "measures": ["count rows"],
        "filters": ["status = completed"],
        "ordering": "row_count descending", "limit": 3,
    }, ["events.parquet"], {"borough", "status"})
    assert draft == {
        "filters": [["status", "equals", "completed"]],
        "temporal_filters": [], "dimensions": [["borough", "borough"]],
        "measures": [["row_count", "count_rows", []]], "joins": [],
        "ordering": [["row_count", "descending"]], "limit": 3,
    }
    assert _conservative_draft_from_requirements(
        {"grouping": ["neighborhood"], "measures": ["count rows"]},
        ["events.parquet"], {"borough"},
    ) is None


def test_semantic_planner_prompt_contains_runtime_only_context():
    state = P12State()
    state.all_candidates = ["events.parquet"]
    state.confirmed_tables = ["events.parquet"]
    state.selection_requirements = {"measures": ["count rows"]}
    state.inspection_cache["events.parquet"] = "Schema: id"
    prompt = _semantic_planner_prompt("Count events", state)
    assert "Schema: id" in prompt
    assert "SOURCE_REFERENCE_RESULT" not in prompt
    assert "BENCHMARK_SECRET_DO_NOT_LEAK" not in prompt
    assert "expected_result" not in prompt


def test_selection_blocks_uncovered_data_requirement(tmp_path):
    state = P12State()
    state.all_candidates = ["selected.parquet", "alternative.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache = {
        "selected.parquet": "Schema for selected.parquet",
        "alternative.parquet": "Schema for alternative.parquet",
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    with pytest.raises(ValueError, match="requested historical year"):
        manager.confirm_unified_selection(
            "selected is the strongest available source",
            ["selected.parquet"],
            requirement_coverage={
                "measure": {"table": "selected.parquet", "columns": ["value"]},
            },
            table_roles={"selected.parquet": "fact records"},
            uncovered_requirements=["requested historical year"],
            alternatives_rejected={
                "alternative.parquet": "requested historical year",
            },
        )


def test_selection_allows_uncovered_computational_requirement(tmp_path):
    state = P12State()
    state.all_candidates = ["selected.parquet"]
    state.visible_candidate_count = 1
    state.inspection_cache = {"selected.parquet": "Schema: rotation"}
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    result = manager.confirm_unified_selection(
        "rotation values are available", ["selected.parquet"],
        requirement_coverage={
            "rotation values": {
                "table": "selected.parquet", "columns": ["rotation"],
            },
        },
        table_roles={"selected.parquet": "fact records"},
        uncovered_requirements=["derive rotation range categories"],
    )

    ledger = json.loads(result.split("FINAL_PAYLOAD: ", 1)[1])[
        "selection_plan"
    ]["requirement_ledger"]
    assert any(
        item["request"] == "derive rotation range categories"
        and item["status"] == "computational"
        for item in ledger
    )


def test_divided_selection_blocks_uncovered_data_requirement(tmp_path):
    manager = Phase2JudgeToolsManager(
        ["selected.parquet"], tmp_path, question="Count records in 2020"
    )
    manager._inspection_cache["selected.parquet"] = "Schema: value"

    with pytest.raises(ValueError, match="historical year 2020"):
        manager.confirm_table_selection(
            "selected is relevant", ["selected.parquet"],
            requirement_coverage={
                "measure": {
                    "table": "selected.parquet", "columns": ["value"],
                },
            },
            uncovered_requirements=["historical year 2020"],
        )


def test_strong_alternative_without_concrete_missing_requirement_is_advisory(tmp_path):
    state = P12State()
    state.all_candidates = ["selected.parquet", "strong-alternative.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache = {
        "selected.parquet": "Schema for selected.parquet",
        "strong-alternative.parquet": "Schema for strong-alternative.parquet",
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    result = manager.confirm_unified_selection(
        "selected is preferred",
        ["selected.parquet"],
        alternatives_rejected={
            "strong-alternative.parquet": {
                "matched_requirements": ["hydrography subject", "2024 edition"],
                "missing_requirement": "less relevant",
            },
        },
    )

    payload = json.loads(result.split("FINAL_PAYLOAD: ", 1)[1])
    assert payload["tables"] == "selected.parquet"
    assert any("Reconsider including or preferring" in item for item in payload["advisories"])


def test_unified_selection_blocks_exact_coder_rejected_combination(tmp_path):
    state = P12State()
    state.all_candidates = ["a.parquet", "b.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache = {
        "a.parquet": "Schema for a.parquet",
        "b.parquet": "Schema for b.parquet",
    }
    state.rejected_selections = {("a.parquet", "b.parquet")}
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    with pytest.raises(ValueError, match="exact table combination"):
        manager.confirm_unified_selection(
            "retry the same tables", ["b.parquet", "a.parquet"]
        )

    result = manager.confirm_unified_selection("use a different set", ["b.parquet"])
    assert '"tables": "b.parquet"' in result


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


def test_divided_inspect_columns_resolves_candidate_number(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tools_p2, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    manager = Phase2JudgeToolsManager(
        ["a.parquet", "b.parquet", "c.parquet"], tmp_path
    )

    # candidate_number is 1-indexed and resolves without the filename ever
    # being typed out, so there is nothing for the model to transcribe wrong.
    assert manager.inspect_columns(candidate_number=2) == "Schema for b.parquet"
    assert manager.inspect_columns(candidate_number=0).startswith("Error:")
    assert manager.inspect_columns(candidate_number=4).startswith("Error:")


def test_divided_expand_candidates_numbering_continues_from_visible_count(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        tools_p2, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    candidates = [f"table-{index}.parquet" for index in range(1, 21)]
    manager = Phase2JudgeToolsManager(candidates, tmp_path)
    manager.metadata = {
        name: {"title": name, "description": "value"} for name in candidates
    }
    manager.inspect_columns(candidate_number=1)

    expanded = manager.expand_candidates("value")
    # The first newly-revealed candidate continues the same numbering the
    # agent already saw (11), instead of resetting to "Candidate 1" and
    # colliding with the first candidate from the initial listing.
    assert "Candidate 11 " in expanded
    assert "Candidate 1 " not in expanded


def test_reject_selection_splits_inspected_candidates_into_keep_and_skip(tmp_path):
    manager = Phase2JudgeToolsManager(
        ["a.parquet", "b.parquet", "c.parquet"], tmp_path
    )
    manager._inspection_cache["a.parquet"] = "Schema for a.parquet"
    manager._inspection_cache["b.parquet"] = "Schema for b.parquet"
    manager._inspection_cache["c.parquet"] = "Schema for c.parquet"

    manager.reject_selection(
        "b covers the count but not the district breakdown",
        "look for a district-level table",
        ban_tables={"c.parquet": "has no count or district column at all"},
    )

    # Only the explicitly-banned candidate is excluded; everything else
    # inspected is kept by default, even without an explicit endorsement.
    assert manager.rejection_keep_tables == ["a.parquet", "b.parquet"]
    assert manager.rejection_skip_tables == ["c.parquet"]


def test_reject_selection_ignores_uninspected_or_unknown_ban_tables(tmp_path):
    manager = Phase2JudgeToolsManager(["a.parquet", "b.parquet"], tmp_path)
    manager._inspection_cache["a.parquet"] = "Schema for a.parquet"
    # b.parquet was never inspected; c.parquet is not even a candidate.

    manager.reject_selection(
        "neither table was actually judged", "try something else",
        ban_tables={
            "b.parquet": "has the missing district column",
            "c.parquet": "has the missing district column",
        },
    )

    # A ban naming an uninspected or unknown table bans nothing; the one
    # genuinely-inspected candidate is kept by default.
    assert manager.rejection_keep_tables == ["a.parquet"]
    assert manager.rejection_skip_tables == []


def test_reject_selection_rejects_unjustified_ban_tables(tmp_path):
    manager = Phase2JudgeToolsManager(["a.parquet", "b.parquet"], tmp_path)
    manager._inspection_cache["a.parquet"] = "Schema for a.parquet"
    manager._inspection_cache["b.parquet"] = "Schema for b.parquet"

    with pytest.raises(ValueError, match="concrete evidence"):
        manager.reject_selection(
            "a.parquet does not contain the required organisation",
            "search for the correct organisation",
            ban_tables={"a.parquet": "bad"},
        )

    # A bare empty justification is rejected the same way.
    with pytest.raises(ValueError, match="concrete evidence"):
        manager.reject_selection(
            "a.parquet does not contain the required organisation",
            "search for the correct organisation",
            ban_tables={"a.parquet": ""},
        )

    # State from the rejected calls above must not leak into a later, valid call.
    manager.reject_selection(
        "a covers the count but b lacks the district breakdown",
        "look for a district-level table",
        ban_tables={"b.parquet": "has no district or count column at all"},
    )
    assert manager.rejection_keep_tables == ["a.parquet"]
    assert manager.rejection_skip_tables == ["b.parquet"]


def test_phase2_selection_requires_inspection(tmp_path):
    manager = Phase2JudgeToolsManager(["table.parquet"], tmp_path)

    with pytest.raises(ValueError, match="inspect_columns is mandatory"):
        manager.confirm_table_selection("relevant", ["table.parquet"])

    manager._inspection_cache["table.parquet"] = "Schema for table.parquet"
    assert "FINAL_PAYLOAD" in manager.confirm_table_selection(
        "relevant", ["table.parquet"]
    )


def test_divided_and_unified_selection_build_identical_requirement_ledgers(tmp_path):
    question = "What was the average number of records per district?"
    coverage = {
        "district dimension": {
            "table": "table.parquet", "columns": ["district"],
        }
    }
    requirements = {
        "grouping": ["district"], "measures": ["record count"],
        "result_type": "number",
    }
    divided = Phase2JudgeToolsManager(
        ["table.parquet"], tmp_path, question=question
    )
    divided._inspection_cache["table.parquet"] = "Schema for table.parquet"
    divided_payload = json.loads(divided.confirm_table_selection(
        "relevant", ["table.parquet"], requirement_coverage=coverage,
        table_roles={"table.parquet": "fact records"},
        requirements=requirements,
    ).split("FINAL_PAYLOAD: ", 1)[1])

    unified = Phase12ToolsManager(
        P12State(), object(), [], tmp_path, question=question
    )
    unified_ledger = unified._build_requirement_ledger(
        coverage, requirements, [], None
    )

    assert divided_payload["selection_plan"]["requirement_ledger"] == unified_ledger


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

    result = manager.search_tables("useful")

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

    result = manager.search_tables("requested")

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

    initial = manager.search_tables("tables")
    assert "table-10.parquet" in initial
    assert "table-11.parquet" not in initial
    assert "10 additional ranked candidates" in initial
    assert manager.expand_candidates("value").startswith("Expansion blocked")

    assert manager.inspect_columns("table-1.parquet").startswith("Schema")
    expanded = manager.expand_candidates("value")
    assert "Guided expansion" in expanded
    assert "table-15.parquet" in expanded
    assert "table-16.parquet" not in expanded
    assert "5 ranked candidates remain hidden" in expanded

    final_expansion = manager.expand_candidates("value")
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

    assert "Guided expansion" in manager.expand_candidates("table")
    assert manager.inspect_columns("table-4.parquet").startswith("Schema")
    assert manager.inspect_columns("table-5.parquet").startswith("Schema")
    assert manager.inspect_columns("table-6.parquet").startswith("Schema")
    assert manager.inspect_columns("table-7.parquet").startswith("Inspection blocked")


def test_unified_inspect_columns_resolves_candidate_number(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tools_p12, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    state = P12State()
    state.all_candidates = [f"table-{index}.parquet" for index in range(1, 4)]
    state.visible_candidate_count = 3
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    # candidate_number is 1-indexed and resolves without the filename ever
    # being typed out, so there is nothing for the model to transcribe wrong.
    assert manager.inspect_columns(candidate_number=2) == "Schema for table-2.parquet"
    assert manager.inspect_columns(candidate_number=0).startswith("Error:")
    assert manager.inspect_columns(candidate_number=4).startswith("Error:")


def test_unified_expand_candidates_numbering_continues_from_visible_count(
    monkeypatch, tmp_path
):
    class FakeService:
        def retrieve(self, **kwargs):
            return [_hit(f"table-{i}", i, ["value"]) for i in range(1, 21)]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_: FakeService())
    monkeypatch.setattr(tools_p12, "_inspect_columns", lambda *_: "Schema: value")
    files = [f"table-{i}.parquet" for i in range(1, 21)]
    state = P12State()
    manager = Phase12ToolsManager(state, object(), files, tmp_path)
    manager.search_tables("values")
    manager.inspect_columns(files[0])

    expanded = manager.expand_candidates("value")
    # The first newly-revealed candidate continues the same numbering the
    # agent already saw (11), instead of resetting to "Candidate 1" and
    # colliding with the first candidate from the initial listing.
    assert "Candidate 11 " in expanded
    assert "Candidate 1 " not in expanded


def test_guided_expansion_prefers_hidden_metadata_covering_missing_requirement(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        tools_p12, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    state = P12State()
    state.all_candidates = [f"table-{index}.parquet" for index in range(1, 14)]
    state.visible_candidate_count = 10
    state.solr_meta = {
        "table-11.parquet": {"columns": [{"name": "unrelated"}]},
        "table-12.parquet": {"columns": [{"name": "Borough"}, {"name": "Year"}]},
        "table-13.parquet": {"columns": [{"name": "other"}]},
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)
    manager.inspect_columns("table-1.parquet")

    expanded = manager.expand_candidates("borough year")

    assert "table-12.parquet" in expanded
    assert "table-11.parquet" not in expanded
    assert state.expansion_requirements == ["borough", "year"]
    assert state.all_candidates[10] == "table-12.parquet"


def test_guided_expansion_stops_when_hidden_metadata_has_no_coverage(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        tools_p12, "_inspect_columns", lambda _directory, name: f"Schema for {name}"
    )
    state = P12State()
    state.all_candidates = [f"table-{index}.parquet" for index in range(1, 12)]
    state.visible_candidate_count = 10
    state.solr_meta = {
        "table-11.parquet": {"columns": [{"name": "unrelated"}]},
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)
    manager.inspect_columns("table-1.parquet")

    result = manager.expand_candidates("borough")

    assert result.startswith("No hidden candidate")
    assert state.expansion_count == 1


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
    first_expansion = manager.expand_candidates("table")
    assert "Guided expansion" in first_expansion
    assert "5 ranked candidates remain hidden" in first_expansion
    assert len(manager.visible_candidates()) == 15

    final_expansion = manager.expand_candidates("table")
    assert "Expansion limit reached" in final_expansion
    assert "Do not call expand_candidates again" in final_expansion


def test_keyword_tool_splits_multiword_concepts_into_solr_words(monkeypatch, tmp_path):
    """search_keyword_concepts must split each concept into the individual
    words Solr ANDs together (WordDelimiterGraphFilter), not send it as one
    joined phrase -- otherwise the banlist can't key on what Solr matched."""
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(kwargs["keywords"])
            return []

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    manager = Phase12ToolsManager(
        P12State(), object(), [], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )
    tool = manager.get_tools()[0]

    parameters = tool.metadata.get_parameters_dict()
    assert parameters["properties"]["concepts"]["type"] == "array"
    tool.call(
        concepts=[" Transport  for Greater Manchester ", "invoices"]
    )

    assert calls == [["transport", "for", "greater", "manchester", "invoices"]]


def test_keyword_banlist_uses_actual_and_terms_inside_concepts(tmp_path):
    state = P12State()
    state.failed_keyword_combinations = [frozenset({"belfast", "lough"})]
    manager = Phase12ToolsManager(
        state, object(), [], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    result = manager.search_keyword_concepts(["Belfast Lough", "cells"])

    assert "known zero-result keyword subset {belfast, lough}" in result


# A live UK run: "How many water abstraction licences were in place in
# Northern Ireland as of 16 May 2023?". The architect bound the snapshot column
# Dataset_Da (sampled '2023/05/16 ...') under keys that never repeated "2023",
# was blocked three times, then wrote its own primary source into its bans.
_NI_QUESTION = (
    "How many water abstraction licences were in place in Northern Ireland "
    "as of 16 May 2023?"
)
_NI_INSPECTION = """Schema for ni.parquet:
Rows: 853
Temporal coverage:
- date_licen: 2007-07-10 to 2023-02-24 (missing/unparseable 38.5%)
- Date_Appli: 2007-01-17 to 2023-03-29 (missing/unparseable 5.7%)
Columns (types and categories sampled from first 500 rows):
- apprefno (str)
- Source (Category sample): ['Groundwater', 'Surface water']
- date_licen (str)
- Dataset_Da (Category sample): ['2023/05/16 00:00:00+00']"""


def _ni_manager(tmp_path):
    pd.DataFrame({
        "apprefno": ["A1"], "date_licen": ["2020/01/01"],
        "Dataset_Da": ["2023/05/16 00:00:00+00"],
    }).to_parquet(tmp_path / "ni.parquet")
    pd.DataFrame({"value": [1]}).to_parquet(tmp_path / "other.parquet")
    state = P12State()
    state.all_candidates = ["ni.parquet", "other.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache["ni.parquet"] = _NI_INSPECTION
    state.inspection_cache["other.parquet"] = "Schema for other.parquet:\n- value (int64)"
    manager = Phase12ToolsManager(
        state, object(), state.all_candidates, tmp_path, question=_NI_QUESTION,
    )
    return state, manager


def _temporal_scope(ledger, period="2023"):
    return next(
        item for item in ledger
        if item["kind"] == "temporal_scope" and item["request"] == period
    )


def test_period_binds_to_bound_column_whose_inspected_values_show_it(tmp_path):
    state, manager = _ni_manager(tmp_path)

    result = manager.confirm_unified_selection(
        "Dataset_Da is the 16 May 2023 snapshot; count licences.", ["ni.parquet"],
        requirement_coverage={
            "licence count": {"table": "ni.parquet", "columns": ["apprefno"]},
            "snapshot date": {"table": "ni.parquet", "columns": ["Dataset_Da"]},
        },
        table_roles={"ni.parquet": "licence snapshot"},
        requirements={"measures": ["count distinct licences"], "result_type": "number"},
    )

    assert "FINAL_PAYLOAD" in result
    item = _temporal_scope(state.selection_plan["requirement_ledger"])
    assert item["status"] == "bound"
    assert item["evidence"] == {"table": "ni.parquet", "columns": ["Dataset_Da"]}


def test_sampled_values_beat_a_date_range_that_merely_spans_the_period():
    ledger = build_requirement_ledger(
        _NI_QUESTION,
        {
            "licence date": {"table": "ni.parquet", "columns": ["date_licen"]},
            "snapshot": {"table": "ni.parquet", "columns": ["Dataset_Da"]},
        },
        {}, [], inspections={"ni.parquet": _NI_INSPECTION},
    )

    assert _temporal_scope(ledger)["evidence"]["columns"] == ["Dataset_Da"]


def test_period_binds_through_measured_range_only_when_inside_it():
    coverage = {"issue date": {"table": "ni.parquet", "columns": ["date_licen"]}}
    inspections = {"ni.parquet": _NI_INSPECTION}

    inside = build_requirement_ledger(
        "How many licences were issued in 2015?", coverage, {}, [],
        inspections=inspections,
    )
    outside = build_requirement_ledger(
        "How many licences were issued in 2024?", coverage, {}, [],
        inspections=inspections,
    )

    assert _temporal_scope(inside, "2015")["evidence"]["columns"] == ["date_licen"]
    assert _temporal_scope(outside, "2024")["status"] == "unresolved"


def test_period_years_expand_fiscal_and_slash_periods():
    assert _period_years("2023") == [2023]
    assert _period_years("2019-20") == [2019, 2020]
    assert _period_years("1999-00") == [1999, 2000]
    assert _period_years("2019/2020") == [2019, 2020]


def test_unproven_period_blocks_with_an_actionable_message(tmp_path):
    state, manager = _ni_manager(tmp_path)

    with pytest.raises(ValueError) as blocked:
        manager.confirm_unified_selection(
            "Count licences in the NI snapshot.", ["ni.parquet"],
            requirement_coverage={
                "licence count": {"table": "ni.parquet", "columns": ["apprefno"]},
            },
            table_roles={"ni.parquet": "licence snapshot"},
        )

    message = str(blocked.value)
    assert 'no requirement_coverage key names "2023"' in message
    assert '"2023": {"table": "ni.parquet", "columns": ["<date or snapshot column>"]}' in message
    assert "reject it instead of confirming again" in message
    assert state.blocked_selection["tables"] == ["ni.parquet"]
    assert state.selection_plan == {}


def test_rejection_cannot_ban_a_table_proposed_in_the_same_attempt(tmp_path):
    state, manager = _ni_manager(tmp_path)
    with pytest.raises(ValueError):
        manager.confirm_unified_selection(
            "Count licences in the NI snapshot.", ["ni.parquet"],
            requirement_coverage={
                "licence count": {"table": "ni.parquet", "columns": ["apprefno"]},
            },
        )

    response = manager.reject_unified_selection(
        "No licence status column.", "licence status",
        ban_tables={
            "ni.parquet": "Only a dataset-level snapshot column.",
            "other.parquet": "Holds a single unrelated value column.",
        },
    )

    assert state.rejection_skip_tables == ["other.parquet"]
    assert state.rejection_keep_tables == ["ni.parquet"]
    assert "Not banned, kept for the next attempt" in response
    assert "ni.parquet" in response.splitlines()[-1]
    assert state.retrieval_memory_events[-1]["evidence"] == {
        "other.parquet": "Holds a single unrelated value column.",
    }


def test_blocked_confirmation_is_recovered_instead_of_top_candidates(tmp_path):
    state, manager = _ni_manager(tmp_path)
    with pytest.raises(ValueError):
        manager.confirm_unified_selection(
            "Count licences in the NI snapshot.", ["ni.parquet"],
            requirement_coverage={
                "licence count": {"table": "ni.parquet", "columns": ["apprefno"]},
            },
            table_roles={"ni.parquet": "licence snapshot"},
        )

    tables, reasoning, plan, advisory = _recover_blocked_selection(
        state, ["ni.parquet", "other.parquet"]
    )

    assert tables == ["ni.parquet"]
    assert reasoning == "Count licences in the NI snapshot."
    assert plan["requirement_coverage"]["licence count"]["columns"] == ["apprefno"]
    assert plan["recovered_from_existing_discovery_context"] is True
    assert requirement_ledger_blockers(plan["requirement_ledger"], tables) == ["2023"]
    assert "No selected-table column was bound for: 2023" in advisory
    assert _recover_blocked_selection(state, ["other.parquet"]) is None
    assert _recover_blocked_selection(P12State(), ["ni.parquet"]) is None


def test_catalogue_text_is_plain_words_with_real_less_than_signs_kept():
    from lakegen.core.catalogue import clean_catalogue_text

    styled = (
        "<span style='font-family:Lato, &quot;Avenir Next&quot;; "
        "font-size:18px;'>This file is a best fit lookup</span>"
    )

    assert clean_catalogue_text(styled) == "This file is a best fit lookup"
    assert clean_catalogue_text("centres with <25 eyes") == "centres with <25 eyes"
    assert clean_catalogue_text("<P>accuracy is &lt;10m.</P><P></P>") == "accuracy is <10m."
    assert clean_catalogue_text("Roles &amp; Salaries&nbsp;2021") == "Roles & Salaries 2021"
    assert clean_catalogue_text(None) == ""


def test_candidate_context_serves_the_description_without_portal_html():
    from lakegen.phases.utils import format_candidate_context, solr_metadata_from_doc

    metadata = solr_metadata_from_doc({
        "title": "CSV",
        "description": (
            '<DIV STYLE="text-align:Left;"><DIV><DIV><P><SPAN><SPAN>In Northern '
            "Ireland water abstraction and impoundment is controlled by The "
            "Water Abstraction and Impoundment (Licensing) Regulations"
            "</SPAN></SPAN></P></DIV></DIV></DIV>"
        ),
        "tags": ["<b>Abstraction</b>", "NIEA"],
        "columns": [{"name": "apprefno", "description": "<p>Reference</p>", "type": "str"}],
    })

    context = format_candidate_context(["ni.parquet"], {"ni.parquet": metadata})

    assert "  Description: In Northern Ireland water abstraction and impoundment" in context
    assert "  Topics: Abstraction, NIEA" in context
    assert "apprefno [str] — Reference" in context
    assert "<" not in context


def test_count_rows_measure_needs_no_columns(tmp_path):
    """Live UK run: the architect's licence count, {"operation": "count_rows"}
    with no `columns`, was rejected as "measures.0.columns Field required"."""
    state, manager = _ni_manager(tmp_path)
    count_licences = {
        "output": "licence_count", "operation": "count_rows",
        "table": "ni.parquet", "evidence": "one row per licence in the snapshot",
    }

    result = manager.confirm_unified_selection(
        "Dataset_Da is the 16 May 2023 snapshot; count its licence rows.",
        ["ni.parquet"],
        requirement_coverage={
            "snapshot 16 May 2023": {"table": "ni.parquet", "columns": ["Dataset_Da"]},
        },
        table_roles={"ni.parquet": "licence snapshot"},
        semantic_plan={"measures": [count_licences]},
    )

    assert "FINAL_PAYLOAD" in result
    assert state.selection_plan["semantic_plan"]["measures"][0]["columns"] == []
    compiled = compile_semantic_plan_draft(
        {"measures": [{"output": "licence_count", "operation": "count_rows"}]},
        ["ni.parquet"], {"ni.parquet": "licence snapshot"},
        {"ni.parquet": {"apprefno", "Dataset_Da"}},
    )
    assert compiled["measures"][0]["operation"] == "count_rows"


def test_aggregating_measure_without_columns_is_still_rejected(tmp_path):
    _state, manager = _ni_manager(tmp_path)

    with pytest.raises(ValueError, match="columns is required for operation 'sum'"):
        manager.confirm_unified_selection(
            "Sum the daily volumes.", ["ni.parquet"],
            requirement_coverage={
                "snapshot 16 May 2023": {"table": "ni.parquet", "columns": ["Dataset_Da"]},
            },
            table_roles={"ni.parquet": "licence snapshot"},
            semantic_plan={"measures": [{
                "output": "total_volume", "operation": "sum",
                "table": "ni.parquet", "evidence": "vol_perday is numeric",
            }]},
        )


# --- context management: state board, compaction, loop guards ---------------


def _tool_call(name, call_id, **kwargs):
    from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock

    return ChatMessage(role="assistant", blocks=[
        ToolCallBlock(tool_name=name, tool_kwargs=kwargs, tool_call_id=call_id)
    ])


def _tool_result(text, call_id):
    from llama_index.core.base.llms.types import ChatMessage

    return ChatMessage(role="tool", content=text, additional_kwargs={"tool_call_id": call_id})


def test_state_board_keeps_what_each_inspected_candidate_shows(tmp_path):
    state, manager = _ni_manager(tmp_path)
    state.inspection_counts["ni.parquet"] = 1

    board = manager.context.render_board()

    assert board.startswith("[DISCOVERY STATE")
    assert '- Candidate 1 "Unknown": 853 rows; 2023: in Dataset_Da values' in board
    assert "question words found nowhere: water, abstraction, licences, northern, ireland" in board
    assert "Budgets: inspections 1 of 3 used (6 after expand_candidates)" in board
    assert "Tools still worth calling: search_tables" not in board
    assert "confirm_unified_selection, reject_unified_selection" in board


def test_state_board_reports_the_last_blocked_confirmation(tmp_path):
    state, manager = _ni_manager(tmp_path)
    confirm = manager._tracked(manager.confirm_unified_selection, "confirm_unified_selection")
    arguments = dict(
        reasoning="Count licences.", tables=["ni.parquet"],
        requirement_coverage={"licence count": {"table": "ni.parquet", "columns": ["apprefno"]}},
    )

    with pytest.raises(ValueError):
        confirm(**arguments)
    with pytest.raises(ValueError) as again:
        confirm(**arguments)

    from lakegen.agent_tools.discovery_context import REPEATED_BLOCK

    assert str(again.value).startswith(REPEATED_BLOCK)
    assert state.confirm_blocks == 2
    board = manager.context.render_board()
    assert "Last confirm_unified_selection was blocked (2 blocked so far): Selection blocked: the question asks about 2023" in board


def test_compaction_shortens_superseded_results_and_keeps_one_current_board(tmp_path):
    from lakegen.agent_tools.discovery_context import COMPACTED

    state, manager = _ni_manager(tmp_path)
    del state.inspection_cache["other.parquet"]
    state.inspection_counts["ni.parquet"] = 1
    state.solr_meta["ni.parquet"] = {"title": "NI licences", "description": "Register", "columns.name": ["apprefno"]}
    state.solr_meta["other.parquet"] = {"title": "Other", "description": "Unrelated values", "columns.name": ["value"]}
    listing = "Attempt: 1\nSearched: concepts ['water']\n\nCandidates in retrieval order after local-file mapping:\nCandidate 1 (retrieval rank 1)\n  File: ni.parquet\n  Description: " + "x" * 900 + "\n\nCandidate 2 (retrieval rank 2)\n  File: other.parquet"
    manager.context.record("search_tables", listing)
    manager.context.record("inspect_columns", _NI_INSPECTION)
    first_block = "Selection blocked: first reason. Add an entry."
    second_block = "Selection blocked: second reason. Add an entry."
    manager.context.record("confirm_unified_selection", first_block, error=True)
    manager.context.record("confirm_unified_selection", second_block, error=True)
    history = [
        _tool_call("search_tables", "c0"), _tool_result(listing, "c0"),
        _tool_call("inspect_columns", "c1"), _tool_result(_NI_INSPECTION, "c1"),
        _tool_call("confirm_unified_selection", "c2"), _tool_result(first_block, "c2"),
        _tool_call("confirm_unified_selection", "c3"), _tool_result(second_block, "c3"),
    ]

    edited = manager.context.edit(history)
    texts = [message.content for message in edited if message.role.value == "tool"]

    assert texts[0].startswith(COMPACTED + "Attempt: 1 Searched: concepts ['water']")
    assert "Candidate 1: NI licences -- inspected, profile below" in texts[0]
    assert "Candidate 2: Other -- Unrelated values | columns: value" in texts[0]
    assert "x" * 50 not in texts[0]
    assert texts[1] == _NI_INSPECTION
    assert texts[2] == COMPACTED + "Earlier confirmation, superseded: Selection blocked: first reason."
    assert texts[3].startswith(second_block + "\n\n[DISCOVERY STATE")
    assert sum("[DISCOVERY STATE" in text for text in texts) == 1
    assert [m.additional_kwargs for m in edited] == [m.additional_kwargs for m in history]
    # Editing an already-edited history changes nothing but the refreshed board.
    assert [m.content for m in manager.context.edit(edited)] == [m.content for m in edited]


def test_context_editor_rewrites_what_the_model_sees_on_the_real_agent_loop():
    import itertools

    from llama_index.core.base.llms.types import ChatMessage, ToolCallBlock
    from llama_index.core.llms.mock import MockFunctionCallingLLM
    from llama_index.core.tools import FunctionTool

    from lakegen.agents.agent_runner import run_agent_workflow

    seen = []
    script = iter([("lookup", {}), None])
    ids = itertools.count()

    def respond(messages):
        seen.append([m.content for m in messages if m.role.value == "tool"])
        step = next(script)
        if step is None:
            return ChatMessage(role="assistant", content="done")
        return ChatMessage(role="assistant", blocks=[
            ToolCallBlock(tool_name=step[0], tool_kwargs=step[1], tool_call_id=f"c{next(ids)}")
        ])

    def lookup() -> str:
        """Lookup."""
        return "raw result"

    from llama_index.core.base.llms.types import TextBlock

    def editor(messages):
        return [
            m.model_copy(update={"blocks": [TextBlock(text=m.content + " +edited")]})
            if m.role.value == "tool" else m
            for m in messages
        ]

    run_agent_workflow(
        llm=MockFunctionCallingLLM(response_generator=respond), system_prompt="s",
        user_prompt="u", agent_name="t", emit_stream=lambda _t: None,
        tools=[FunctionTool.from_defaults(fn=lookup)], context_editor=editor,
    )

    assert seen == [[], ["raw result +edited"]]


def test_repeated_search_points_back_and_a_withdrawn_tool_says_so(monkeypatch, tmp_path):
    class Service:
        def retrieve(self, **_kwargs):
            return [RetrievalHit(document={"resource_id": "a", "title": "A"}, score=1.0, rank=1)]

    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_a, **_k: Service())
    pd.DataFrame({"value": [1]}).to_parquet(tmp_path / "a.parquet")
    manager = Phase12ToolsManager(
        P12State(), object(), ["a.parquet"], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )
    search = manager._tracked(manager.search_keyword_concepts, "search_tables")

    first = search(["alpha"])
    repeated = search(["alpha"])
    manager.inspect_columns(candidate_number=1)
    after_inspection = search(["beta"])

    assert "Candidate 1" in first
    assert repeated == (
        "search_tables is no longer available in this attempt; the state block "
        "lists the tools still worth calling. Search skipped: identical concepts "
        "were already used (Attempt: 1, shown above). Do not repeat this search."
    )
    assert after_inspection.startswith("search_tables is no longer available in this attempt")


def test_context_management_off_reproduces_the_previous_behaviour(monkeypatch, tmp_path):
    from lakegen.experiment_config import DiscoveryConfig

    state, manager = _ni_manager(tmp_path)
    manager.discovery = DiscoveryConfig(state_board=False, compact_history=False)

    tools = {tool.metadata.name: tool for tool in manager.get_tools()}
    tools["inspect_columns"].call(candidate_number=1)

    assert not manager.context.enabled
    assert manager.context.kinds == {}
    assert not tools["inspect_columns"].metadata.description.startswith("inspect_columns(")


def test_unified_prompt_mentions_the_state_board_only_when_it_is_on():
    from lakegen.core.resources import get_prompt_manager

    def render(**flags):
        return get_prompt_manager().render(
            "unified_architect", "system_prompt", portal_name="UK", hint="",
            initial_shortlist_size=3, max_inspected_candidates=6, **flags
        )

    assert "DISCOVERY STATE block" in render(state_board=True, compact_history=True)
    assert "shown compacted" in render(state_board=True, compact_history=True)
    assert "shown compacted" not in render(state_board=True, compact_history=False)
    assert "DISCOVERY STATE" not in render(state_board=False, compact_history=False)
