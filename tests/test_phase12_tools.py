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
from lakegen.agent_tools.tools_p2 import Phase2JudgeToolsManager
from lakegen.phases.phase12 import (
    _conservative_draft_from_requirements,
    _extract_plausible_json,
    _inspected_runtime_evidence,
    _reasoning_with_selection_plan,
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


def test_unified_search_allows_one_distinct_refinement_and_merges_candidates(
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
    assert "gold.parquet" in second
    assert state.best_ranks == {"generic.parquet": 1, "gold.parquet": 3}
    assert len(state.search_attempts) == 2
    assert repeated.startswith("Search skipped")
    assert calls == [["school"], ["bandwidth"]]

    limited = manager.search_solr("third distinct concept")
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
    manager.search_solr("first concept")
    manager.inspect_columns("table.parquet")

    assert manager.search_solr("new concept").startswith(
        "Search refinement blocked"
    )


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
    assert repeated.startswith("Search skipped: identical concepts")
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
def test_search_tool_allows_at_most_two_distinct_calls(
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
    expected_calls = (
        1 if mode in (RetrievalMode.SEMANTIC, RetrievalMode.PNEUMA) else 2
    )
    assert len(calls) == expected_calls
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


def test_missing_agentic_plan_is_recovered_as_non_blocking_coder_context():
    plan, advisories = _recover_minimal_selection_plan(
        ["events.parquet", "boroughs.parquet"],
        "Join the event records to the borough lookup using the shared key.",
    )
    context = _reasoning_with_selection_plan("selected", plan, advisories)

    assert plan["combination_strategy"] == "lookup"
    assert plan["recovered_from_existing_discovery_context"] is True
    assert set(plan["table_roles"]) == {"events.parquet", "boroughs.parquet"}
    assert "treat it as guidance, not as a blocking constraint" in context


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
    assert selection_plan["coder_brief"] == {
        "tables": ["events.parquet"],
        "selected_columns": {"events.parquet": ["borough"]},
        "task": {
            "grouping": ["borough"], "measures": ["count rows"],
            "filters": [], "ordering": "row_count descending", "limit": 3,
        },
        "filters": [], "operations": ["count rows"],
        "result_type": "auto", "ordering": "row_count descending",
        "limit": 3, "joins": [], "normalization_errors": [],
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
        "left_table": "plazas.parquet", "left_columns": ["Partner"],
        "right_table": "organizations.parquet", "right_columns": ["Organization name"],
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


def test_selection_records_uncovered_requirements_and_rejected_alternatives(tmp_path):
    state = P12State()
    state.all_candidates = ["selected.parquet", "alternative.parquet"]
    state.visible_candidate_count = 2
    state.inspection_cache = {
        "selected.parquet": "Schema for selected.parquet",
        "alternative.parquet": "Schema for alternative.parquet",
    }
    manager = Phase12ToolsManager(state, object(), state.all_candidates, tmp_path)

    result = manager.confirm_unified_selection(
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

    payload = json.loads(result.split("FINAL_PAYLOAD: ", 1)[1])
    plan = payload["selection_plan"]
    assert plan["uncovered_requirements"] == ["requested historical year"]
    assert plan["alternatives_rejected"] == {
        "alternative.parquet": {
            "matched_requirements": [],
            "missing_requirement": "requested historical year",
        }
    }
    assert any("still marked uncovered" in item for item in payload["advisories"])
    context = _reasoning_with_selection_plan("selected", plan, payload["advisories"])
    assert "alternative.parquet: matches []; lacks requested historical year" in context


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
    assert manager.inspect_columns("table-6.parquet").startswith("Inspection blocked")


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
