from types import SimpleNamespace

from lakegen.output_validation import AnswerDisposition, validate_answer
from lakegen.experiment_config import (
    DiscoveryArchitecture, ExperimentConfig, InteractionMode, ReviewerConfig, ToolAccess,
)
from lakegen.retrieval import RetrievalConfig
from lakegen.service import (
    _record_review_telemetry, _record_semantic_plan_telemetry, _rejected_selection_signature,
    _selection_plan_signature, _selection_retry_feedback,
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


def test_selection_plan_signature_ignores_mapping_order_but_detects_corrections():
    first = _selection_plan_signature(["B.csv", "a.csv"], {
        "coder_brief": {
            "filters": [{"column": "year", "value": 2024}],
            "measures": ["count rows"],
        }
    })
    reordered = _selection_plan_signature(["a.csv", "B.csv"], {
        "coder_brief": {
            "measures": ["count rows"],
            "filters": [{"value": 2024, "column": "year"}],
        }
    })
    corrected = _selection_plan_signature(["a.csv", "B.csv"], {
        "coder_brief": {
            "measures": ["count rows"],
            "filters": [{"value": 2025, "column": "year"}],
        }
    })
    assert first == reordered
    assert corrected != first


def test_selection_retry_feedback_preserves_tables_and_structures_diagnostics():
    feedback = _selection_retry_feedback(
        ["vision-zero.parquet"],
        {"status": "invalid", "validation_diagnostics": [{
            "category": "unknown_column",
            "table": "vision-zero.parquet",
            "evidence": {"unknown": ["Partner category"]},
        }]},
        "Selection contract was not verified.",
    )

    assert feedback["keep_current_tables"] is True
    assert feedback["previous_selection"] == ["vision-zero.parquet"]
    correction = feedback["required_corrections"][0]
    assert correction["category"] == "unknown_column"
    assert "inspect the runtime schema" in correction["action"]
    assert "Do not repeat" in feedback["instruction"]


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


def test_review_telemetry_is_a_noop_for_a_plain_non_reviewed_run():
    result = QueryResult(question="q", status="completed")
    generated = SimpleNamespace(review_trace=[])
    _record_review_telemetry(result, generated)
    assert result.review_pipeline_used is False
    assert result.review_trace == []


def test_review_telemetry_records_stage_attempts_and_summary():
    result = QueryResult(question="q", status="completed")
    trace = [
        {"stage": "plan", "attempt": 1, "approved": False, "max_attempts": 3},
        {"stage": "plan", "attempt": 2, "approved": True, "max_attempts": 3},
        {"stage": "validator", "attempt": 1, "approved": True, "max_attempts": 3},
        {"stage": "code_judge", "attempt": 1, "approved": False, "max_attempts": 3},
        {"stage": "code_judge", "attempt": 2, "approved": True, "max_attempts": 3},
        {"stage": "summary", "outcome": "validated", "reason": "", "generation_calls": 4, "wall_time": 12.3},
    ]
    generated = SimpleNamespace(review_trace=trace)
    _record_review_telemetry(result, generated)
    assert result.review_pipeline_used is True
    assert result.review_outcome == "validated"
    assert result.review_generation_calls == 4
    assert result.review_stage_attempts == {"plan": 2, "validator": 1, "code_judge": 2}
    assert result.review_trace == trace


def test_review_telemetry_records_a_decline():
    result = QueryResult(question="q", status="rejected")
    trace = [
        {"stage": "plan", "attempt": 1, "approved": False, "max_attempts": 3},
        {"stage": "plan", "attempt": 2, "approved": False, "max_attempts": 3},
        {"stage": "plan", "attempt": 3, "approved": False, "max_attempts": 3},
        {"stage": "summary", "outcome": "declined", "reason": "never approved",
         "generation_calls": 3, "wall_time": 9.1},
    ]
    generated = SimpleNamespace(review_trace=trace)
    _record_review_telemetry(result, generated)
    assert result.review_outcome == "declined"
    assert result.review_stage_attempts == {"plan": 3}


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


def test_reviewed_pipeline_is_used_once_instead_of_the_outer_retry_loop(monkeypatch, tmp_path):
    """reviewers.plan=True must route through
    phase3_generate_and_execute_reviewed exactly once -- never the plain
    phase3_generate_and_execute, and never MAX_CODE_ATTEMPTS times -- since
    the reviewed pipeline owns its own internal stage-scoped retrying."""
    experiment = ExperimentConfig().model_copy(update={
        "core": "nyc", "model": "fake",
        "interaction_mode": InteractionMode.AUTONOMOUS,
        "reviewers": ReviewerConfig(plan=True),
    })
    runtime = SimpleNamespace(
        model_name="fake", solr_core="nyc", csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(), experiment=experiment,
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

    def unexpected_plain_call(*_args, **_kwargs):
        raise AssertionError("plain phase3_generate_and_execute must not be called when reviewers.plan is on")

    reviewed_calls = []

    fake_review_trace = [
        {"stage": "plan", "attempt": 1, "approved": True, "max_attempts": 3},
        {"stage": "validator", "attempt": 1, "approved": True, "max_attempts": 3},
        {"stage": "summary", "outcome": "validated", "reason": "", "generation_calls": 1, "wall_time": 1.0},
    ]

    def fake_reviewed(*_args, **_kwargs):
        reviewed_calls.append(_kwargs)
        return SimpleNamespace(
            tokens=0, clean_code="print(42)", code_raw="print(42)",
            rejected_reason="", finalization_mode="", error=None,
            raw_result="42 schools", structured_result=None,
            execution_error=None, coder_runs=1, coder_context_audit=None,
            review_trace=fake_review_trace,
        )

    monkeypatch.setattr("lakegen.service.phase3_generate_and_execute", unexpected_plain_call)
    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute_reviewed", fake_reviewed)
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("42 schools", 0)
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("How many schools?", runtime)

    assert len(reviewed_calls) == 1
    assert reviewed_calls[0]["stage_max_retries"] == 3
    assert reviewed_calls[0]["enable_plan_review"] is True
    assert reviewed_calls[0]["enable_code_review"] is False
    assert result.status == "completed"
    assert result.review_pipeline_used is True
    assert result.review_outcome == "validated"
    assert result.review_stage_attempts == {"plan": 1, "validator": 1}


def test_review_declined_result_is_rejected_without_table_banning(monkeypatch, tmp_path):
    """A review-stage decline (finalization_mode == "review_declined") must
    set result.status == "rejected" with the review's own reason, and must
    NOT be routed through the table-banning/rediscovery machinery that a
    genuine coder-side reject_tables verdict uses."""
    experiment = ExperimentConfig().model_copy(update={
        "core": "nyc", "model": "fake",
        "interaction_mode": InteractionMode.AUTONOMOUS,
        "reviewers": ReviewerConfig(plan=True, code=True),
    })
    runtime = SimpleNamespace(
        model_name="fake", solr_core="nyc", csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(), experiment=experiment,
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

    def fake_reviewed(*_args, **_kwargs):
        return SimpleNamespace(
            tokens=0, clean_code="print(1)", code_raw="print(1)",
            rejected_reason="Code judge never approved after 3 tries: unclear result.",
            finalization_mode="review_declined", error=None,
            raw_result=None, structured_result=None,
            execution_error=None, coder_runs=1, coder_context_audit=None,
        )

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute_reviewed", fake_reviewed)
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("How many schools?", runtime)

    assert result.status == "rejected"
    assert result.error == "Code judge never approved after 3 tries: unclear result."
    assert result.pipeline_stages["code_execution"] == "declined_by_review"


def test_last_attempt_fallback_excludes_just_rejected_tables(monkeypatch, tmp_path):
    # Divided + agentic is the only architecture that carries a per-table ban
    # (excluded_tables) across discovery attempts; the last-attempt fallback
    # bug only surfaces there.
    experiment = ExperimentConfig().model_copy(update={
        "core": "nyc", "model": "fake",
        "discovery_architecture": DiscoveryArchitecture.DIVIDED,
        "tool_access": ToolAccess.AGENTIC,
        "interaction_mode": InteractionMode.AUTONOMOUS,
    })
    runtime = SimpleNamespace(
        model_name="fake", solr_core="nyc", csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(), experiment=experiment,
    )
    (tmp_path / "bad.parquet").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "good.parquet").write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr(
        "lakegen.service.get_all_table_files",
        lambda _path: ["bad.parquet", "good.parquet"],
    )
    monkeypatch.setattr(
        "lakegen.service.phase1_generate_keywords",
        lambda **_kwargs: (["k"], "raw", 0, "p1 reasoning"),
    )

    def fake_phase2(*, selection_state, **_kwargs):
        # Every attempt inspects both candidates and explicitly rules out
        # "bad.parquet"; the architect never finds a complete selection, so
        # every attempt ends in REJECT_KEYWORDS -- including the last one.
        selection_state.rejection_keep_tables = []
        selection_state.rejection_skip_tables = ["bad.parquet"]
        return (
            [], ["bad.parquet", "good.parquet"], {},
            "REJECT_KEYWORDS: bad.parquet has no matching rows", "trace", 0,
        )

    monkeypatch.setattr("lakegen.service.phase2_select_tables", fake_phase2)
    generated = SimpleNamespace(
        tokens=0, clean_code="print(1)", code_raw="print(1)",
        rejected_reason="", error=None, raw_result="1",
    )
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute", lambda *_args, **_kwargs: generated
    )
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("answer", 0)
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("What is the value?", runtime)

    # The last-attempt fallback must not force-select a table the architect
    # explicitly rejected in that very same attempt.
    assert "bad.parquet" not in result.tables
    assert result.tables == ["good.parquet"]


def test_unified_cross_round_exclusion_persists_across_attempts(monkeypatch, tmp_path):
    # The unified architecture previously started every discovery attempt
    # from a blank P12State, so a table proven insufficient in attempt 1
    # could silently resurface in attempt 2 or 3. This locks in that
    # excluded_tables (seeded onto the fresh P12State each attempt) actually
    # carries the ban forward, mirroring the divided architecture's test
    # above.
    experiment = ExperimentConfig().model_copy(update={
        "core": "nyc", "model": "fake",
        "discovery_architecture": DiscoveryArchitecture.UNIFIED,
        "tool_access": ToolAccess.AGENTIC,
        "interaction_mode": InteractionMode.AUTONOMOUS,
    })
    runtime = SimpleNamespace(
        model_name="fake", solr_core="nyc", csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(), experiment=experiment,
    )
    (tmp_path / "bad.parquet").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "good.parquet").write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr(
        "lakegen.service.get_all_table_files",
        lambda _path: ["bad.parquet", "good.parquet"],
    )

    seen_excluded_tables = []

    def fake_phase12(*, state, **_kwargs):
        # Records what excluded_tables looked like when this attempt
        # started, then behaves like a real retrieval: "bad.parquet" is
        # never a candidate once excluded. Every attempt rejects, and every
        # attempt that inspects "bad.parquet" bans it, so attempt 2 onward
        # must never see it again.
        seen_excluded_tables.append(set(state.excluded_tables))
        state.all_candidates = [
            table for table in ["bad.parquet", "good.parquet"]
            if table.casefold() not in state.excluded_tables
        ]
        state.rejection_keep_tables = []
        state.rejection_skip_tables = (
            ["bad.parquet"] if "bad.parquet" in state.all_candidates else []
        )
        return (
            list(state.all_candidates), ["k"], {},
            "REJECT_KEYWORDS: still missing the required organisation",
            "trace", 0,
        )

    monkeypatch.setattr("lakegen.service.phase12_agent", fake_phase12)
    generated = SimpleNamespace(
        tokens=0, clean_code="print(1)", code_raw="print(1)",
        rejected_reason="", error=None, raw_result="1",
    )
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute", lambda *_args, **_kwargs: generated
    )
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("answer", 0)
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("What is the value?", runtime)

    assert seen_excluded_tables[0] == set()
    assert seen_excluded_tables[1] == {"bad.parquet"}
    assert seen_excluded_tables[2] == {"bad.parquet"}
    assert "bad.parquet" not in result.tables
    assert result.tables == ["good.parquet"]


def test_coder_rejection_feeds_discovery_exclusion_and_carry(monkeypatch, tmp_path):
    # The coder's own per-table judgment (reject_tables' ban_tables) must
    # feed the same excluded_tables/carried_tables memory an architect
    # rejection does: a table it proved unusable stays banned from the next
    # discovery attempt, and one it found partially useful is carried
    # forward instead of being lost with the rest of the selected set.
    experiment = ExperimentConfig().model_copy(update={
        "core": "nyc", "model": "fake",
        "discovery_architecture": DiscoveryArchitecture.DIVIDED,
        "tool_access": ToolAccess.AGENTIC,
        "interaction_mode": InteractionMode.AUTONOMOUS,
    })
    runtime = SimpleNamespace(
        model_name="fake", solr_core="nyc", csv_dir=tmp_path,
        portal_name="NYC", retrieval=RetrievalConfig(), experiment=experiment,
    )
    (tmp_path / "good.parquet").write_text("value\n1\n", encoding="utf-8")
    (tmp_path / "bad.parquet").write_text("value\n1\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr(
        "lakegen.service.get_all_table_files",
        lambda _path: ["good.parquet", "bad.parquet"],
    )
    monkeypatch.setattr(
        "lakegen.service.phase1_generate_keywords",
        lambda **_kwargs: (["k"], "raw", 0, "p1 reasoning"),
    )

    seen_excluded_tables = []
    seen_carried_tables = []

    def fake_phase2(*, excluded_tables, carried_tables, **_kwargs):
        seen_excluded_tables.append(set(excluded_tables))
        seen_carried_tables.append(list(carried_tables))
        selected = list(carried_tables) or ["good.parquet", "bad.parquet"]
        return (
            selected, selected, {}, "the selection covers the question", "trace", 0,
        )

    monkeypatch.setattr("lakegen.service.phase2_select_tables", fake_phase2)

    phase3_calls = []

    def fake_phase3(*_args, **_kwargs):
        phase3_calls.append(1)
        if len(phase3_calls) == 1:
            return SimpleNamespace(
                tokens=0, clean_code="print(1)", code_raw="print(1)",
                rejected_reason="good.parquet covers part of the answer.",
                rejection_details={}, error=None, raw_result=None,
                rejection_keep_tables=["good.parquet"],
                rejection_skip_tables=["bad.parquet"],
            )
        return SimpleNamespace(
            tokens=0, clean_code="print(1)", code_raw="print(1)",
            rejected_reason="", rejection_details={}, error=None,
            raw_result="1", rejection_keep_tables=[], rejection_skip_tables=[],
        )

    monkeypatch.setattr("lakegen.service.phase3_generate_and_execute", fake_phase3)
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("answer", 0)
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("What is the value?", runtime)

    assert seen_excluded_tables[0] == set()
    assert seen_carried_tables[0] == []
    assert seen_excluded_tables[1] == {"bad.parquet"}
    assert seen_carried_tables[1] == ["good.parquet"]
    assert "bad.parquet" not in result.tables


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
