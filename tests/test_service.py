from types import SimpleNamespace

from lakegen.output_validation import AnswerDisposition, validate_answer
from lakegen.experiment_config import ExperimentConfig
from lakegen.retrieval import RetrievalConfig
from lakegen.service import (
    _record_semantic_plan_telemetry, _rejected_selection_signature,
    extract_questions, run_question,
)
from lakegen.service_models import QueryResult


def test_rejected_selection_signature_preserves_runtime_failure_requirements():
    signature = _rejected_selection_signature(
        ["B.parquet", "a.parquet"],
        {
            "category": "temporal_coverage_incompatible",
            "missing_requirements": ["records for 2024", "join column school_id"],
        },
        "Missing column school_id and incompatible temporal coverage.",
    )
    assert signature["tables"] == ["a.parquet", "b.parquet"]
    assert signature["category"] == "temporal_coverage_incompatible"
    assert signature["missing_requirements"] == [
        "join column school_id", "records for 2024"
    ]


def test_semantic_plan_telemetry_is_persisted_without_coder_sweep():
    result = QueryResult(question="q", status="failed")
    _record_semantic_plan_telemetry(result, {
        "semantic_plan_present": True, "semantic_plan_status": "verified",
        "semantic_plan_locked": True, "semantic_plan_revised": False,
        "semantic_plan_rejected": False, "validation_diagnostics": [],
        "evidence_count": 4, "coder_started_after_verified_plan": True,
    })
    payload = result.to_dict()
    assert payload["semantic_plan_status"] == "verified"
    assert payload["evidence_count"] == 4
    assert payload["coder_started_after_verified_plan"] is True
    assert payload["semantic_plan_initial_status"] == "verified"
    assert payload["semantic_plan_final_status"] == "verified"
    assert payload["semantic_plan_coder_start_status"] == "verified"


def test_coder_brief_telemetry_separates_selection_and_effective_status():
    result = QueryResult(question="q", status="completed")
    _record_semantic_plan_telemetry(result, {
        "coder_brief": {"source": "runtime_fallback"},
        "contract_type": "coder_brief",
        "selection_brief_status": "missing",
        "effective_coder_brief_status": "executable_with_obligations",
        "effective_coder_brief_source": "runtime_fallback",
        "semantic_plan_status": "executable_with_obligations",
        "coder_started_after_verified_plan": True,
        "semantic_plan_coder_start_status": "executable_with_obligations",
    })
    payload = result.to_dict()
    assert payload["selection_brief_status"] == "missing"
    assert payload["effective_coder_brief_status"] == "executable_with_obligations"
    assert payload["effective_coder_brief_source"] == "runtime_fallback"
    assert payload["coder_brief_status"] == "executable_with_obligations"


def test_extracts_queries_old_shape_and_preserves_metadata():
    payload = {
        "model": {
            "SQL": {
                "0": {
                    "data": {
                        "queries": [
                            {"id": 7, "question": "First question?", "code": "SELECT 1"},
                            {"id": None, "question": "Second question?"},
                        ]
                    }
                }
            }
        }
    }

    questions = extract_questions(payload)

    assert [item.question for item in questions] == ["First question?", "Second question?"]
    assert questions[0].source_id == 7
    assert questions[0].path == "$.model.SQL['0'].data.queries[0].question"
    assert questions[0].source_data["code"] == "SELECT 1"
    assert questions[0].log_fields()["SOURCE_CODE"] == "SELECT 1"


def test_extracts_simple_question_lists():
    questions = extract_questions({"questions": [" One? ", {"question": "Two?"}, ""]})

    assert [item.question for item in questions] == ["One?", "Two?"]


def test_extracts_top_level_string_list():
    questions = extract_questions(["One?", " Two? "])

    assert [item.question for item in questions] == ["One?", "Two?"]


def test_does_not_treat_unrelated_strings_as_questions():
    assert extract_questions({"description": "not a question", "tables": ["users"]}) == []


def test_answer_validation_distinguishes_valid_rejected_and_empty():
    assert validate_answer("42 schools").disposition == AnswerDisposition.VALID
    assert validate_answer("  ").disposition == AnswerDisposition.EMPTY
    refusal = validate_answer(
        "The data provided does not contain information about bandwidth."
    )
    assert refusal.disposition == AnswerDisposition.REJECTED


def test_run_question_does_not_mark_synthesized_refusal_completed(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        model_name="fake",
        solr_core="nyc",
        csv_dir=tmp_path,
        portal_name="NYC",
        retrieval=RetrievalConfig(),
    )
    (tmp_path / "gold.csv").write_text("value\n42\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["gold.csv"])
    monkeypatch.setattr(
        "lakegen.service.phase12_agent",
        lambda **_kwargs: (["gold.csv"], ["gold"], {}, "correct table", "trace", 0),
    )
    generated = SimpleNamespace(
        tokens=0,
        clean_code="print(42)",
        code_raw="print(42)",
        rejected_reason="",
        error=None,
        raw_result="42 schools",
    )
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute", lambda *_args, **_kwargs: generated
    )
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize",
        lambda *_args: ("The necessary data is not available.", 0),
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("How many schools?", runtime)

    assert result.status == "rejected"
    assert result.answer_disposition == "rejected"
    assert result.pipeline_stages["final_answer"] == "rejected"


def test_coder_revision_failure_does_not_restart_discovery(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        model_name="fake", solr_core="nyc", csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(),
    )
    (tmp_path / "selected.csv").write_text("value\n42\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr(
        "lakegen.service.get_all_table_files", lambda _path: ["selected.csv"]
    )
    discovery_calls = []

    def discover(**_kwargs):
        discovery_calls.append("called")
        return ["selected.csv"], ["selected"], {}, "verified selection", "trace", 0

    monkeypatch.setattr("lakegen.service.phase12_agent", discover)
    coder_calls = []

    def generate(*_args, **_kwargs):
        coder_calls.append("called")
        retryable = len(coder_calls) == 1
        return SimpleNamespace(
            tokens=0, clean_code="print(42)", code_raw="print(42)",
            rejected_reason="", error="result needs revision", raw_result=None,
            coder_runs=1, execution_error={
                "stage": "result_validation", "category": "result_needs_revision",
                "retryable": retryable,
            }, coder_context_audit=None,
        )

    monkeypatch.setattr("lakegen.service.phase3_generate_and_execute", generate)
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("What is the value?", runtime)

    assert result.status == "failed"
    assert discovery_calls == ["called"]
    assert coder_calls == ["called", "called"]
    assert result.discovery["selection_attempts"][0]["outcome"] == "selected"


def test_run_question_evaluates_structured_benchmark_code_result(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        model_name="fake",
        solr_core="nyc",
        csv_dir=tmp_path,
        portal_name="NYC",
        retrieval=RetrievalConfig(),
    )
    (tmp_path / "gold.csv").write_text("value\n42\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["gold.csv"])
    monkeypatch.setattr(
        "lakegen.service.phase12_agent",
        lambda **_kwargs: (["gold.csv"], ["gold"], {}, "correct table", "trace", 0),
    )
    captured = {}

    def generate(*_args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            tokens=0,
            clean_code="print(42)",
            code_raw="print(42)",
            rejected_reason="",
            error=None,
            raw_result='[{"total": 42}]',
            structured_result=42,
            structured_result_error="",
        )

    monkeypatch.setattr("lakegen.service.phase3_generate_and_execute", generate)
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("There are 42.", 0)
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question(
        "How many?",
        runtime,
        log_context={
            "SOURCE_EXPECTED_RESULT_TYPE": "number",
            "SOURCE_EXPECTED_RESULT_DESCRIPTION": "BENCHMARK_SECRET_DO_NOT_LEAK",
            "SOURCE_REFERENCE_RESULT": [{"total": 42}],
        },
    )

    # Benchmark result type remains evaluator-only; Phase 3 derives its generic
    # output shape from the question.
    assert captured["evaluation_result_type"] is None
    assert "expected_result_description" not in captured
    assert "SOURCE_REFERENCE_RESULT" in captured["source_field_names"]
    assert "BENCHMARK_SECRET_DO_NOT_LEAK" not in repr(captured)
    assert result.code_evaluation["exact_result_match"] is True
    assert result.code_evaluation["pass_at_1"] is True
    assert result.code_evaluation["success_within_3"] is True
    assert result.execution_outcome["code_evaluation"]["numeric_match"] is True
    assert result.execution_outcome["code_evaluation"]["numeric_absolute_error"] == 0.0


def test_automatic_coder_sweep_reuses_one_discovery_context(monkeypatch, tmp_path):
    experiment = ExperimentConfig(automatic_test_coder=True)
    runtime = SimpleNamespace(
        model_name=experiment.model,
        solr_core=experiment.core,
        csv_dir=tmp_path,
        portal_name="NYC",
        retrieval=RetrievalConfig(),
        experiment=experiment,
    )
    (tmp_path / "gold.csv").write_text("value\n42\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["gold.csv"])
    discovery_calls = []

    def discover(**_kwargs):
        discovery_calls.append("called")
        return ["gold.csv"], ["gold"], {}, "shared reasoning", "trace", 0

    monkeypatch.setattr("lakegen.service.phase12_agent", discover)
    coder_calls = []

    def generate(*_args, **kwargs):
        level = kwargs["coder_context_level"].value
        coder_calls.append(level)
        value = 41 if level == "minimal" else 42
        return SimpleNamespace(
            tokens=10,
            clean_code=f"print({value})",
            code_raw=f"print({value})",
            rejected_reason="",
            error=None,
            raw_result=str(value),
            structured_result=value,
            structured_result_error="",
        )

    monkeypatch.setattr("lakegen.service.phase3_generate_and_execute", generate)
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("There are 42.", 0)
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question(
        "How many?",
        runtime,
        log_context={
            "SOURCE_EXPECTED_RESULT_TYPE": "number",
            "SOURCE_REFERENCE_RESULT": [{"total": 42}],
        },
    )

    assert discovery_calls == ["called"]
    assert coder_calls == ["full", "schema_only", "minimal"]
    assert result.tokens["p3"] == 30
    assert result.code == "print(42)"
    assert result.status == "completed"
    experiment_result = result.coder_context_experiment
    assert experiment_result["shared_retrieval"] is True
    assert experiment_result["shared_tables"] == ["gold.csv"]
    assert experiment_result["variants"]["full"]["code_evaluation"][
        "exact_result_match"
    ] is True
    assert experiment_result["variants"]["schema_only"]["code_evaluation"][
        "exact_result_match"
    ] is True
    assert experiment_result["variants"]["minimal"]["code_evaluation"][
        "exact_result_match"
    ] is False


def test_automatic_coder_full_rejection_restarts_discovery_before_sweep(
    monkeypatch, tmp_path
):
    experiment = ExperimentConfig(automatic_test_coder=True)
    runtime = SimpleNamespace(
        model_name=experiment.model, solr_core=experiment.core,
        csv_dir=tmp_path, portal_name="NYC", retrieval=RetrievalConfig(),
        experiment=experiment,
    )
    (tmp_path / "first.csv").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "second.csv").write_text("value\n42\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr(
        "lakegen.service.get_all_table_files",
        lambda _path: ["first.csv", "second.csv"],
    )
    discovery_calls = []
    selection_state_ids = []

    def discover(**kwargs):
        discovery_calls.append("called")
        selection_state_ids.append(id(kwargs["state"]))
        # The second discovery turn ignores the feedback and repeats the exact
        # rejected set. Service must block it before invoking the coder.
        table = "first.csv" if len(discovery_calls) <= 2 else "second.csv"
        return [table], ["value"], {}, "selection", "trace", 0

    monkeypatch.setattr("lakegen.service.phase12_agent", discover)
    coder_calls = []

    def generate(*args, **kwargs):
        level = kwargs["coder_context_level"].value
        table = args[1][0]
        coder_calls.append((table, level))
        rejected = table == "first.csv" and level == "full"
        return SimpleNamespace(
            tokens=0, clean_code="print(42)", code_raw="print(42)",
            rejected_reason="wrong tables" if rejected else "",
            error=None, raw_result=None if rejected else "42",
            structured_result=None if rejected else 42,
            structured_result_error="", coder_runs=1,
            execution_error=None, coder_review=None,
        )

    monkeypatch.setattr("lakegen.service.phase3_generate_and_execute", generate)
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("There are 42.", 0)
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question(
        "How many?", runtime,
        log_context={
            "SOURCE_EXPECTED_RESULT_TYPE": "number",
            "SOURCE_REFERENCE_RESULT": 42,
        },
    )

    assert result.status == "completed"
    assert discovery_calls == ["called", "called", "called"]
    assert len(set(selection_state_ids)) == 3
    assert coder_calls == [
        ("first.csv", "full"),
        ("second.csv", "full"),
        ("second.csv", "schema_only"),
        ("second.csv", "minimal"),
    ]
    assert result.tables == ["second.csv"]
    assert [attempt["outcome"] for attempt in result.discovery["selection_attempts"]] == [
        "tables_rejected",
        "rejected_selection_excluded",
        "accepted",
    ]
