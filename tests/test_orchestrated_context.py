import json
import re
from types import SimpleNamespace

import pytest

from lakegen.experiment_config import DiscoveryArchitecture
from lakegen.orchestrated_context import (
    PreparedCandidate,
    PreparedDiscoveryContext,
    prepare_discovery_context,
)
from lakegen.phases.orchestrated_discovery import (
    DiscoveryResult,
    OrchestratedContextPreparationError,
    OrchestratedSelectorError,
    RetrievalRequestProtocolError,
    parse_retrieval_request,
    run_unified_orchestrated_discovery,
    select_from_prepared_context,
)
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.models import RetrievalHit
from lakegen.retrieval.intent import parse_retrieval_intent
from lakegen.experiment_config import ExperimentConfig
from lakegen.service import run_question
from prompts.prompt_manager import PromptManager


def _intent(concepts, **overrides):
    payload = {
        "status": "resolved", "concepts": concepts, "entities": [],
        "measures": [], "filters": [], "time_constraints": [],
        "group_by": [], "order_by": [], "limit": None,
        "join_requirements": [], "missing_evidence": [],
    }
    payload.update(overrides)
    return "RETRIEVAL_INTENT: " + json.dumps(payload)


@pytest.mark.parametrize(
    "mode", [RetrievalMode.KEYWORD, RetrievalMode.SEMANTIC, RetrievalMode.HYBRID]
)
def test_preparer_forwards_existing_retrieval_config_and_preserves_order(
    mode, monkeypatch
):
    calls = []
    hits = [
        RetrievalHit(
            document={
                "resource_id": "b",
                "title": "Second",
                "tags": ["two"],
                "columns": [{"name": "value", "type": "number"}],
            },
            score=0.9,
            rank=1,
            lexical_score=2.0 if mode != RetrievalMode.SEMANTIC else None,
            semantic_score=0.8 if mode != RetrievalMode.KEYWORD else None,
        ),
        RetrievalHit(
            document={"resource_id": "a", "title": "First", "columns": []},
            score=0.7,
            rank=2,
        ),
    ]

    class FakeRetriever:
        def retrieve(self, **kwargs):
            calls.append(kwargs)
            return hits

    config = RetrievalConfig(mode=mode, top_k=2, alpha=0.37)
    monkeypatch.setattr(
        "lakegen.orchestrated_context.get_table_retrieval_service",
        lambda client, actual_config: (
            calls.append({"client": client, "config": actual_config})
            or FakeRetriever()
        ),
    )
    context, metadata = prepare_discovery_context(
        query="question",
        keywords=["alpha"],
        solr_client="fake-solr",
        all_files=["a.csv", "b.csv"],
        retrieval_config=config,
    )

    assert calls[0] == {"client": "fake-solr", "config": config}
    assert calls[1]["question"] == "question"
    assert calls[1]["keywords"] == ["alpha"]
    assert calls[1]["top_k"] == 2
    assert [item.dataset for item in context.candidates] == ["b.csv", "a.csv"]
    assert context.retrieved_hit_count == 2
    assert context.prepared_candidate_count == 2
    assert [item.retrieval_rank for item in context.candidates] == [1, 2]
    assert metadata["b.csv"]["title"] == "Second"
    assert context.stable_json() == context.stable_json()


def test_agent_facing_context_has_one_mode_neutral_schema_and_telemetry_keeps_signals():
    contexts = []
    for mode in RetrievalMode:
        context = PreparedDiscoveryContext(
            query="question",
            retrieval_mode=mode.value,
            candidates=[PreparedCandidate(
                retrieval_rank=1,
                prepared_position=1,
                dataset="table.csv",
                scores={
                    "score": 0.9,
                    "lexical_score": 2.0,
                    "semantic_score": 0.8,
                },
                missing_signals=["example_signal"],
                metadata={
                    "title": "Table",
                    "description": "Description",
                    "tags": ["tag"],
                    "columns.name": ["value"],
                    "columns.description": ["Measured value"],
                    "columns.type": ["number"],
                    "retrieval": {"semantic_score": 0.8},
                },
            )],
            retrieved_hit_count=1,
            prepared_candidate_count=1,
        )
        agent_payload = json.loads(context.agent_json())
        technical_payload = json.loads(context.stable_json())
        contexts.append(agent_payload)

        assert set(agent_payload) == {"query", "candidates"}
        assert set(agent_payload["candidates"][0]) == {
            "position", "dataset", "metadata"
        }
        assert set(agent_payload["candidates"][0]["metadata"]) == {
            "title", "description", "tags", "columns"
        }
        serialized_agent = context.agent_json()
        for hidden in (
            "retrieval_mode", "lexical_score", "semantic_score",
            "missing_signals", "retrieval_rank", "scores", "retrieval",
        ):
            assert hidden not in serialized_agent

        assert technical_payload["retrieval_mode"] == mode.value
        candidate = technical_payload["candidates"][0]
        assert candidate["scores"]["lexical_score"] == 2.0
        assert candidate["scores"]["semantic_score"] == 0.8
        assert candidate["missing_signals"] == ["example_signal"]

    assert contexts[0] == contexts[1] == contexts[2]


def test_rendered_discovery_prompts_are_mode_neutral():
    prompt_manager = PromptManager()
    rendered = [
        prompt_manager.render(
            "unified_architect", "system_prompt", portal_name="NYC", hint=""
        ),
        prompt_manager.render(
            "unified_architect", "user_prompt", question="Count road incidents"
        ),
        prompt_manager.render("keyword_generator", "system_prompt"),
        prompt_manager.render(
            "keyword_generator", "user_prompt",
            portal_name="NYC", question="Count road incidents",
            raw_keywords_str="road incidents", avoid_keywords_str="",
            keyword_hint="",
        ),
        prompt_manager.render(
            "data_architect", "system_prompt", portal_name="NYC", hint=""
        ),
        prompt_manager.render(
            "data_architect", "user_prompt",
            question="Count road incidents", keywords_str="road incidents",
            enriched_candidates_info="table.csv", table_hint="",
        ),
    ]
    forbidden = re.compile(r"\b(keyword|semantic|hybrid|bm25|knn)\b", re.IGNORECASE)
    assert all(forbidden.search(prompt) is None for prompt in rendered)


def test_tool_free_selector_receives_context_and_no_callable_tools(monkeypatch):
    hit = RetrievalHit(
        document={"resource_id": "table", "title": "Table", "columns": []},
        score=1.0,
        rank=1,
    )
    monkeypatch.setattr(
        "lakegen.orchestrated_context.get_table_retrieval_service",
        lambda *_args: SimpleNamespace(retrieve=lambda **_kwargs: [hit]),
    )
    context, _ = prepare_discovery_context(
        query="question",
        keywords=["table"],
        solr_client=object(),
        all_files=["table.csv"],
        retrieval_config=RetrievalConfig(top_k=1),
    )
    observed = {}

    def fake_agent(**kwargs):
        observed.update(kwargs)
        return 'FINAL_PAYLOAD: {"tables":"table.csv","reasoning":"best"}'

    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery.run_agent_workflow", fake_agent
    )
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery.get_llm_token_usage", lambda _llm: 0
    )
    selected, reasoning, _trace, _tokens = select_from_prepared_context(
        query="question",
        llm=object(),
        context=context,
        all_files=["table.csv"],
        architecture=DiscoveryArchitecture.UNIFIED,
    )

    assert observed["tools"] == []
    assert "table.csv" in observed["user_prompt"]
    forbidden = re.compile(r"\b(keyword|semantic|hybrid|bm25|knn)\b", re.IGNORECASE)
    assert forbidden.search(observed["system_prompt"]) is None
    assert forbidden.search(observed["user_prompt"]) is None
    assert selected == ["table.csv"]
    assert reasoning == "best"


def test_preparation_error_is_explicit_without_agentic_fallback(monkeypatch):
    monkeypatch.setattr(
        "lakegen.orchestrated_context.get_table_retrieval_service",
        lambda *_args: SimpleNamespace(
            retrieve=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("offline boom"))
        ),
    )
    with pytest.raises(RuntimeError, match="offline boom"):
        prepare_discovery_context(
            query="question",
            keywords=["table"],
            solr_client=object(),
            all_files=["table.csv"],
            retrieval_config=RetrievalConfig(),
        )


def test_retrieval_request_is_strict_and_normalizes_concepts():
    request = parse_retrieval_request(_intent(
        [" road   safety ", "", "ROAD SAFETY", "crashes"]
    ))
    assert request.concepts == ["road safety", "crashes"]
    assert request.keywords == ["road safety", "crashes"]
    for malformed in ("hello", "RETRIEVAL_INTENT: {}", _intent([1])):
        with pytest.raises(ValueError):
            parse_retrieval_request(malformed)


def test_canonical_intent_preserves_filters_and_years_as_structured_fields():
    intent = parse_retrieval_intent(_intent(
        ["road incidents"], entities=["incidents"], measures=["count"],
        filters=[{"field": "borough", "operator": "=", "value": "Queens"}],
        time_constraints=[{"field": "year", "operator": "=", "value": 2024}],
        group_by=["borough"], order_by=[{"field": "count", "direction": "desc"}],
        limit=5,
    ))
    assert intent.concepts == ["road incidents"]
    assert intent.time_constraints[0].value == 2024
    assert intent.filters[0].value == "Queens"
    assert "2024" not in intent.concepts


def test_unresolved_intent_is_structured_and_skips_retrieval(monkeypatch):
    calls = []
    response = _intent([], status="unresolved", missing_evidence=["dataset subject"])
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery._run_tool_free_turn",
        lambda **_kwargs: (response, "trace", 1),
    )
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery.prepare_discovery_context",
        lambda **kwargs: calls.append(kwargs),
    )
    result = run_unified_orchestrated_discovery(
        query="How many?", llm=object(), solr_client=object(), all_files=[],
        retrieval_config=RetrievalConfig(),
    )
    assert result.reasoning == "UNRESOLVED_RETRIEVAL_INTENT: dataset subject"
    assert result.selected_datasets == []
    assert calls == []


def test_unified_orchestrated_keeps_history_and_never_uses_phase1_or_tools(monkeypatch):
    calls = []
    responses = iter([
        _intent(["roads"]),
        'FINAL_PAYLOAD: {"tables":"table.csv","reasoning":"best"}',
    ])

    def fake_turn(**kwargs):
        calls.append(kwargs)
        return next(responses), "trace", 3

    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery._run_tool_free_turn", fake_turn
    )
    prepared = PreparedDiscoveryContext(
        query="question", retrieval_mode="keyword",
        candidates=[PreparedCandidate(
            retrieval_rank=1, prepared_position=1, dataset="table.csv",
            scores={"score": 1.0}, missing_signals=[], metadata={},
        )], retrieved_hit_count=1, prepared_candidate_count=1,
    )
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery.prepare_discovery_context",
        lambda **_kwargs: (prepared, {"table.csv": {}}),
    )
    result = run_unified_orchestrated_discovery(
        query="question", llm=object(), solr_client=object(),
        all_files=["table.csv"], retrieval_config=RetrievalConfig(),
    )
    assert result.agent_count == 1 and result.llm_invocations == 2
    assert calls[0]["agent_name"] == calls[1]["agent_name"]
    history = calls[1]["chat_history"]
    assert len(history) == 2
    assert "RETRIEVAL_INTENT" in history[1].content
    forbidden = re.compile(r"\b(keyword|semantic|hybrid|bm25|knn)\b", re.IGNORECASE)
    for call in calls:
        assert forbidden.search(call["system_prompt"]) is None
        assert forbidden.search(call["user_prompt"]) is None


def test_unified_empty_context_skips_second_turn(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery._run_tool_free_turn",
        lambda **kwargs: (calls.append(kwargs) or (_intent(["x"]), "", 1)),
    )
    prepared = PreparedDiscoveryContext(
        query="q", retrieval_mode="keyword", candidates=[],
        retrieved_hit_count=0, prepared_candidate_count=0,
    )
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery.prepare_discovery_context",
        lambda **_kwargs: (prepared, {}),
    )
    result = run_unified_orchestrated_discovery(
        query="q", llm=object(), solr_client=object(), all_files=[],
        retrieval_config=RetrievalConfig(),
    )
    assert len(calls) == 1
    assert result.retry_keywords is True
    assert result.selected_datasets == []


@pytest.mark.parametrize(
    "selector_response",
    [
        "REJECT_KEYWORDS: candidates are irrelevant",
        'FINAL_PAYLOAD: {"tables":"not-in-context.csv","reasoning":"none"}',
    ],
)
def test_unified_selector_without_valid_datasets_requests_retry(
    selector_response, monkeypatch
):
    responses = iter([
        _intent(["roads"]), selector_response,
    ])
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery._run_tool_free_turn",
        lambda **_kwargs: (next(responses), "", 1),
    )
    prepared = PreparedDiscoveryContext(
        query="q", retrieval_mode="keyword",
        candidates=[PreparedCandidate(
            retrieval_rank=1, prepared_position=1, dataset="table.csv",
            scores={}, missing_signals=[], metadata={},
        )], retrieved_hit_count=1, prepared_candidate_count=1,
    )
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery.prepare_discovery_context",
        lambda **_kwargs: (prepared, {}),
    )
    result = run_unified_orchestrated_discovery(
        query="q", llm=object(), solr_client=object(),
        all_files=["table.csv"], retrieval_config=RetrievalConfig(),
    )
    assert result.selected_datasets == []
    assert result.retry_keywords is True
    assert result.retry_reason.startswith("REJECT_KEYWORDS:")


def test_unified_errors_are_typed_by_stage(monkeypatch):
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery._run_tool_free_turn",
        lambda **_kwargs: ("bad request", "", 1),
    )
    with pytest.raises(RetrievalRequestProtocolError):
        run_unified_orchestrated_discovery(
            query="q", llm=object(), solr_client=object(), all_files=[],
            retrieval_config=RetrievalConfig(),
        )

    responses = iter([_intent(["x"])])
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery._run_tool_free_turn",
        lambda **_kwargs: (next(responses), "", 1),
    )
    monkeypatch.setattr(
        "lakegen.phases.orchestrated_discovery.prepare_discovery_context",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("solr failed")),
    )
    with pytest.raises(OrchestratedContextPreparationError):
        run_unified_orchestrated_discovery(
            query="q", llm=object(), solr_client=object(), all_files=[],
            retrieval_config=RetrievalConfig(),
        )


def test_cli_unified_empty_retries_write_exactly_one_terminal_log(monkeypatch, tmp_path):
    import src.cli as cli

    config = ExperimentConfig(
        discovery_architecture="unified", tool_access="orchestrated_context"
    )
    runtime = SimpleNamespace(
        experiment=config, model_name=config.model, solr_core=config.core,
        csv_dir=tmp_path, portal_name="NYC", retrieval=RetrievalConfig(),
        use_unified_agent=True,
    )
    empty = PreparedDiscoveryContext(
        query="q", retrieval_mode="keyword", candidates=[],
        retrieved_hit_count=0, prepared_candidate_count=0,
    )
    monkeypatch.setattr(cli, "get_llm", lambda _name: (object(), None))
    monkeypatch.setattr(cli, "get_solr", lambda _core: object())
    monkeypatch.setattr(cli, "get_prompt_manager", object)
    monkeypatch.setattr(cli, "get_all_table_files", lambda _path: ["table.csv"])
    monkeypatch.setattr(cli, "persist_manifest", lambda *_args: None)
    monkeypatch.setattr(
        cli, "run_unified_orchestrated_discovery",
        lambda **_kwargs: DiscoveryResult(
            selected_datasets=[], candidates=[], keywords=["x"], metadata={},
            reasoning="REJECT_KEYWORDS: empty", trace="", tokens=1,
            llm_invocations=1, agent_count=1, retry_keywords=True,
            retry_reason="REJECT_KEYWORDS: empty", prepared_context=empty,
        ),
    )
    monkeypatch.setattr(
        cli, "phase3_generate_and_execute",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("phase3 reached")),
    )
    logs = []
    monkeypatch.setattr(cli, "save_experiment_log", lambda **kwargs: logs.append(kwargs))

    cli.run_cli_workflow("q", runtime)

    assert len(logs) == 1
    assert logs[0]["status"] == "failed"
    run_trace = logs[0]["extra_fields"]["RUN_TRACE_JSON"]
    assert run_trace["phase_reached"] == "discovery"
    assert run_trace["tool_access"]["empty_context_retries"] == 3
    assert run_trace["tool_access"]["orchestrator_retrieval_calls"] == {"keyword": 3}


@pytest.mark.parametrize("architecture", ["unified", "divided"])
def test_service_dispatches_to_orchestrated_path_and_traces_actor(
    architecture, monkeypatch, tmp_path
):
    config = ExperimentConfig(
        discovery_architecture=architecture,
        tool_access="orchestrated_context",
        interaction_mode="autonomous",
    )
    (tmp_path / "table.csv").write_text("value\n42\n", encoding="utf-8")
    runtime = SimpleNamespace(
        model_name=config.model, solr_core=config.core, csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(top_k=1), experiment=config,
    )
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["table.csv"])
    monkeypatch.setattr("lakegen.service.persist_manifest", lambda *_args: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)
    monkeypatch.setattr(
        "lakegen.service.phase12_agent",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("agentic fallback")),
    )
    monkeypatch.setattr(
        "lakegen.service.phase2_select_tables",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("agentic fallback")),
    )
    monkeypatch.setattr(
        "lakegen.service.phase1_generate_keywords",
        lambda **_kwargs: (["value"], "raw", 1, "keywords"),
    )
    prepared = PreparedDiscoveryContext(
        query="question", retrieval_mode="keyword",
        candidates=[PreparedCandidate(
            retrieval_rank=1, prepared_position=1, dataset="table.csv",
            scores={}, missing_signals=[], metadata={},
        )], prepared_candidate_count=1, retrieved_hit_count=1,
    )
    monkeypatch.setattr(
        "lakegen.service.prepare_discovery_context", lambda **_kwargs: (prepared, {})
    )
    monkeypatch.setattr(
        "lakegen.service.select_from_prepared_context",
        lambda **_kwargs: (["table.csv"], "selected", "trace", 2),
    )
    monkeypatch.setattr(
        "lakegen.service.run_unified_orchestrated_discovery",
        lambda **_kwargs: DiscoveryResult(
            selected_datasets=["table.csv"], candidates=["table.csv"],
            keywords=["value"], metadata={}, reasoning="selected", trace="trace",
            tokens=3, llm_invocations=2, agent_count=1,
            prepared_context=prepared,
        ),
    )
    generated = SimpleNamespace(
        tokens=1, clean_code="print(42)", code_raw="print(42)",
        rejected_reason="", error=None, raw_result="42",
    )
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute", lambda *_a, **_kw: generated
    )
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("42", 1)
    )
    logged = {}
    monkeypatch.setattr(
        "lakegen.service.save_experiment_log", lambda **kwargs: logged.update(kwargs)
    )

    result = run_question("question", runtime)

    assert result.status == "completed"
    trace = logged["extra_fields"]["RUN_TRACE_JSON"]
    assert trace["tool_access"]["execution_path"] == "orchestrated_context"
    assert trace["tool_access"]["prepared_candidate_count"] == 1
    assert trace["tool_access"]["agent_direct_tools"] == []
    assert trace["tool_access"]["orchestrator_retrieval_calls"] == {"keyword": 1}


def test_empty_unified_context_retries_and_never_reaches_phase3(
    monkeypatch, tmp_path
):
    config = ExperimentConfig(
        discovery_architecture="unified", tool_access="orchestrated_context",
        interaction_mode="autonomous",
    )
    (tmp_path / "table.csv").write_text("value\n42\n", encoding="utf-8")
    runtime = SimpleNamespace(
        model_name=config.model, solr_core=config.core, csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(), experiment=config,
    )
    empty = PreparedDiscoveryContext(
        query="question", retrieval_mode="keyword", candidates=[],
        retrieved_hit_count=0, prepared_candidate_count=0,
    )
    calls = {"discovery": 0}

    def empty_discovery(**_kwargs):
        calls["discovery"] += 1
        return DiscoveryResult(
            selected_datasets=[], candidates=[], keywords=[f"try-{calls['discovery']}"],
            metadata={}, reasoning="REJECT_KEYWORDS: No datasets found in the prepared context",
            trace="", tokens=1, llm_invocations=1, agent_count=1,
            retry_keywords=True, retry_reason="empty", prepared_context=empty,
        )

    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["table.csv"])
    monkeypatch.setattr("lakegen.service.persist_manifest", lambda *_args: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.run_unified_orchestrated_discovery", empty_discovery)
    monkeypatch.setattr(
        "lakegen.service.phase1_generate_keywords",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unified used phase1")),
    )
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("empty reached phase3")),
    )
    logged = {}
    monkeypatch.setattr(
        "lakegen.service.save_experiment_log", lambda **kwargs: logged.update(kwargs)
    )

    result = run_question("question", runtime)

    assert result.status == "failed"
    assert calls["discovery"] == 3
    telemetry = logged["extra_fields"]["RUN_TRACE_JSON"]["tool_access"]
    assert telemetry["empty_context_retries"] == 3
    assert telemetry["orchestrator_retrieval_calls"] == {"keyword": 3}
    assert telemetry["llm_invocations"] == 3
