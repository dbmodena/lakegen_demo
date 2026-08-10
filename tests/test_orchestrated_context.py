from types import SimpleNamespace

import pytest

from lakegen.experiment_config import DiscoveryArchitecture
from lakegen.orchestrated_context import prepare_discovery_context
from lakegen.phases.orchestrated_discovery import select_from_prepared_context
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.models import RetrievalHit
from lakegen.experiment_config import ExperimentConfig
from lakegen.service import run_question


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
    assert context.total_candidates_before_limit == 2
    assert context.total_candidates_after_limit == 2
    assert metadata["b.csv"]["title"] == "Second"
    assert context.stable_json() == context.stable_json()


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
    candidate = SimpleNamespace(dataset="table.csv")
    prepared = SimpleNamespace(
        candidates=[candidate], total_candidates_after_limit=1,
        total_candidates_before_limit=1, stable_json=lambda: '{"candidate":1}',
    )
    monkeypatch.setattr(
        "lakegen.service.prepare_discovery_context", lambda **_kwargs: (prepared, {})
    )
    monkeypatch.setattr(
        "lakegen.service.select_from_prepared_context",
        lambda **_kwargs: (["table.csv"], "selected", "trace", 2),
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
    assert trace["tool_access"]["orchestrator_retrieval_calls"] == ["retrieval:keyword"]
