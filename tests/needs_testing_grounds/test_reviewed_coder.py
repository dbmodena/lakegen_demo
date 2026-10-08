"""Direct port of the scratchpad-validated scenarios (four rounds of live
100-question testing) as fake-LLM/fake-generator pytest cases -- mirrors
tests/test_semantic_code_judge.py's fake pattern. The validator stage
(Stage B) uses the REAL value_grounding/query_validator checks against real
CSV data (not mocked) since that's the deterministic, directly bug-motivated
part; only phase3_generate_and_execute and the two LLM judges are faked."""
from __future__ import annotations

from lakegen.analysis_plan import AnalysisPlan
from lakegen.analysis_planner import PlannerResult
from lakegen.coding.agent import Phase3Result
from lakegen.reviewed_coder import phase3_generate_and_execute_reviewed


def _write_table(tmp_path, name="t.csv"):
    # Fiscal-year-span-coded column, exactly like the motivating bug's table.
    (tmp_path / name).write_text(
        "Financial Year,Amount,Postcode\n"
        "200001,10.0,LS1 1AA\n"
        "200102,20.0,LS2 2BB\n"
        "201516,40.0,LS4 4DD\n"
        "201617,5.86,LS8 5NA\n",
        encoding="utf-8",
    )
    return name


def _result(code, columns, answer, used_tables=None):
    return Phase3Result(
        code_raw=code, tokens=10, clean_code=code,
        structured_result=answer,
        operation_trace={"used_tables": used_tables or ["t.csv"], "used_columns": columns},
    )


def _valid_plan(table):
    return AnalysisPlan.model_validate({
        "summary": "Sum Amount by Postcode.",
        "table_roles": {table: "source"},
        "combination": "single_table",
        "measures": [{"output": "total", "operation": "sum", "columns": ["Amount"],
                      "tables": [table], "requirement": "total amount"}],
        "dimensions": [{"output": "Postcode", "column": "Postcode", "tables": [table],
                        "requirement": "by postcode"}],
        "output_columns": ["Postcode", "total"],
        "question_coverage": {"total amount": "measures[0]", "by postcode": "dimensions[0]"},
    })


def _planner(table):
    return lambda **_kwargs: PlannerResult(_valid_plan(table))


def test_plan_rejected_twice_then_approved_on_third_try(tmp_path, monkeypatch):
    table = _write_table(tmp_path)
    generate_calls = []

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        generate_calls.append(retries)
        return _result(f"df.groupby('Postcode')['Amount'].sum()  # call {retries}",
                        ["Postcode", "Amount"], [{"total": 75.86}])

    plan_calls = []

    def fake_plan_judge(**_kwargs):
        plan_calls.append(1)
        approved = len(plan_calls) >= 3
        return ({
            "approves_question_fidelity": approved, "confidence": 0.7,
            "rationale": "retry" if not approved else "matches", "missing_requirements": [],
            "unjustified_steps": [], "judge_error": "",
        }, 5)

    def fake_code_judge(**_kwargs):
        return ({
            "plan_compliance_approval": True, "present_result_approval": True,
            "violated_criteria": [], "feedback": "ok", "suggestions": [],
            "judge_error": "", "approved": True,
        }, 5)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
    )

    assert result.rejected_reason == ""
    assert result.structured_result == [{"total": 75.86}]
    assert len(plan_calls) == 3
    assert len(generate_calls) == 1  # plan retries happen before code is generated
    plan_events = [e for e in result.review_trace if e.get("stage") == "plan"]
    assert len(plan_events) == 3
    assert plan_events[-1]["approved"] is True


def test_plan_stage_exhausted_declines_without_a_guess(tmp_path, monkeypatch):
    table = _write_table(tmp_path)

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        return _result("df.sum()", ["Postcode"], [{"total": 1.0}])

    def fake_plan_judge(**_kwargs):
        return ({
            "approves_question_fidelity": False, "confidence": 0.9,
            "rationale": "never matches", "missing_requirements": ["Amount"],
            "unjustified_steps": [], "judge_error": "",
        }, 5)

    def fake_code_judge(**_kwargs):
        raise AssertionError("code judge must never be called when the plan stage never approves")

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
    )

    assert result.rejected_reason != ""
    assert result.finalization_mode == "review_declined"
    assert result.structured_result is None
    assert result.raw_result is None
    stages_seen = {e.get("stage") for e in result.review_trace}
    assert "validator" not in stages_seen
    assert "code_judge" not in stages_seen


def test_validator_flags_on_first_try_and_clears_on_regeneration(tmp_path, monkeypatch):
    """Reproduces the exact motivating bug shape with REAL checks: a bare
    4-digit-year filter against a fiscal-year-span column produces a zero
    result and a HIGH grounding violation on attempt 1; the corrected code
    on attempt 2 clears validation."""
    table = _write_table(tmp_path)
    attempts = {"n": 0}

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            code = "df[df['Financial Year'] == 2016]['Amount'].sum()"
            return _result(code, ["Financial Year", "Amount"], [{"total": 0.0}])
        code = "df[df['Financial Year'] == 201617]['Amount'].sum()"
        return _result(code, ["Financial Year", "Amount"], [{"total": 5.86}])

    def fake_plan_judge(**_kwargs):
        return ({"approves_question_fidelity": True, "confidence": 0.9, "rationale": "ok",
                  "missing_requirements": [], "unjustified_steps": [], "judge_error": ""}, 5)

    def fake_code_judge(**_kwargs):
        return ({"plan_compliance_approval": True, "present_result_approval": True,
                  "violated_criteria": [], "feedback": "ok", "suggestions": [],
                  "judge_error": "", "approved": True}, 5)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    result = phase3_generate_and_execute_reviewed(
        "Total amount written off in 2016/17?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
    )

    assert result.rejected_reason == ""
    assert result.structured_result == [{"total": 5.86}]
    validator_events = [e for e in result.review_trace if e.get("stage") == "validator"]
    assert len(validator_events) == 2
    assert validator_events[0]["approved"] is False
    assert validator_events[1]["approved"] is True


def test_code_judge_rejects_twice_then_approves_on_third(tmp_path, monkeypatch):
    table = _write_table(tmp_path)
    judge_calls = []

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        return _result(f"df.groupby('Postcode')['Amount'].sum()  # {retries}",
                        ["Postcode", "Amount"], [{"total": 75.86}])

    def fake_plan_judge(**_kwargs):
        return ({"approves_question_fidelity": True, "confidence": 0.9, "rationale": "ok",
                  "missing_requirements": [], "unjustified_steps": [], "judge_error": ""}, 5)

    def fake_code_judge(**_kwargs):
        judge_calls.append(1)
        approved = len(judge_calls) >= 3
        return ({
            "plan_compliance_approval": approved, "present_result_approval": approved,
            "violated_criteria": [] if approved else ["partial_implementation"],
            "feedback": "ok" if approved else "retry", "suggestions": [],
            "judge_error": "", "approved": approved,
        }, 5)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
    )

    assert result.rejected_reason == ""
    assert result.structured_result == [{"total": 75.86}]
    assert len(judge_calls) == 3
    code_judge_events = [e for e in result.review_trace if e.get("stage") == "code_judge"]
    assert len(code_judge_events) == 3
    assert code_judge_events[-1]["approved"] is True


def test_code_judge_stage_exhausted_declines_without_a_guess(tmp_path, monkeypatch):
    table = _write_table(tmp_path)

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        return _result("df.groupby('Postcode')['Amount'].sum()",
                        ["Postcode", "Amount"], [{"total": 75.86}])

    def fake_plan_judge(**_kwargs):
        return ({"approves_question_fidelity": True, "confidence": 0.9, "rationale": "ok",
                  "missing_requirements": [], "unjustified_steps": [], "judge_error": ""}, 5)

    def fake_code_judge(**_kwargs):
        return ({"plan_compliance_approval": False, "present_result_approval": False,
                  "violated_criteria": ["unclear_result"], "feedback": "never good enough",
                  "suggestions": [], "judge_error": "", "approved": False}, 5)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
    )

    assert result.rejected_reason != ""
    assert result.finalization_mode == "review_declined"
    assert result.structured_result is None
    assert result.raw_result is None


def test_reviewers_off_still_runs_validator_and_a_single_generation(tmp_path, monkeypatch):
    """enable_plan_review=False, enable_code_review=False: the validator
    still runs (it's cheap/deterministic and the direct bug fix), but only
    one generation call happens since neither judge loop drives a retry."""
    table = _write_table(tmp_path)
    generate_calls = []

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        generate_calls.append(retries)
        return _result("df.groupby('Postcode')['Amount'].sum()",
                        ["Postcode", "Amount"], [{"total": 75.86}])

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
        enable_plan_review=False, enable_code_review=False,
    )

    assert result.rejected_reason == ""
    assert len(generate_calls) == 1
    stages_seen = {e.get("stage") for e in result.review_trace}
    assert stages_seen == {"validator", "summary"}


def test_on_stage_event_callback_is_invoked_for_every_trace_entry(tmp_path, monkeypatch):
    table = _write_table(tmp_path)

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        return _result("df.groupby('Postcode')['Amount'].sum()",
                        ["Postcode", "Amount"], [{"total": 75.86}])

    def fake_plan_judge(**_kwargs):
        return ({"approves_question_fidelity": True, "confidence": 0.9, "rationale": "ok",
                  "missing_requirements": [], "unjustified_steps": [], "judge_error": ""}, 5)

    def fake_code_judge(**_kwargs):
        return ({"plan_compliance_approval": True, "present_result_approval": True,
                  "violated_criteria": [], "feedback": "ok", "suggestions": [],
                  "judge_error": "", "approved": True}, 5)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    events = []
    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3, on_stage_event=events.append,
    )

    assert events == result.review_trace
    assert events[-1]["stage"] == "summary"
    assert events[-1]["outcome"] == "validated"


def test_coder_own_rejection_short_circuits_and_is_not_marked_review_declined(tmp_path, monkeypatch):
    """When the underlying coder itself gives up on the tables (its own
    reject_tables verdict), the reviewed pipeline must pass that Phase3Result
    through untouched -- same rejected_reason/rejection_details shape as a
    plain (non-reviewed) run -- and must NOT tag it finalization_mode=
    "review_declined", since that's a materially different situation
    (data/table insufficiency, not a review failure) that service.py's
    existing table-banning/rediscovery handling needs to keep recognizing."""
    table = _write_table(tmp_path)
    generate_calls = []

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        generate_calls.append(retries)
        return Phase3Result(
            code_raw="", tokens=5, rejected_reason="The selected table lacks the requested field.",
            rejection_details={"missing_requirements": ["Amount"]},
            rejection_skip_tables=[table],
        )

    def fake_plan_judge(**_kwargs):
        # Planning is deliberately first; the coder can only decline after
        # its plan has been deterministically checked and approved.
        return ({"approves_question_fidelity": True, "confidence": 1.0, "rationale": "ok",
                 "missing_requirements": [], "unjustified_steps": [], "judge_error": ""}, 1)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
    )

    assert result.rejected_reason == "The selected table lacks the requested field."
    assert result.rejection_details == {"missing_requirements": ["Amount"]}
    assert result.rejection_skip_tables == [table]
    assert result.finalization_mode != "review_declined"
    assert len(generate_calls) == 1  # short-circuited on the very first attempt


def test_on_stage_event_exception_does_not_break_the_pipeline(tmp_path, monkeypatch):
    table = _write_table(tmp_path)

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                       retries=0, error_msg="", previous_code="", **_kwargs):
        return _result("df.groupby('Postcode')['Amount'].sum()",
                        ["Postcode", "Amount"], [{"total": 75.86}])

    def fake_plan_judge(**_kwargs):
        return ({"approves_question_fidelity": True, "confidence": 0.9, "rationale": "ok",
                  "missing_requirements": [], "unjustified_steps": [], "judge_error": ""}, 5)

    def fake_code_judge(**_kwargs):
        return ({"plan_compliance_approval": True, "present_result_approval": True,
                  "violated_criteria": [], "feedback": "ok", "suggestions": [],
                  "judge_error": "", "approved": True}, 5)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_plan_question_fidelity", fake_plan_judge)
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    def broken_callback(_event):
        raise RuntimeError("UI is down")

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3, on_stage_event=broken_callback,
    )

    assert result.rejected_reason == ""
    assert result.structured_result == [{"total": 75.86}]


def test_execution_failure_retries_code_before_any_code_judge(tmp_path, monkeypatch):
    """A preflight/execution failure is actionable coder feedback, not a
    plan-compliance question. It must not consume the code-judge budget."""
    table = _write_table(tmp_path)
    generated = []

    def fake_generate(query, tables, candidates, solr_meta, reasoning, llm, pm, csv_dir,
                      retries=0, error_msg="", previous_code="", **_kwargs):
        generated.append((retries, error_msg))
        if len(generated) == 1:
            return Phase3Result(
                code_raw="out[['Quarter', 'Total_Spend']]", tokens=10,
                clean_code="out[['Quarter', 'Total_Spend']]",
                error="Column preflight failed: output aliases were treated as source columns.",
            )
        return _result("df.groupby('Postcode')['Amount'].sum()", ["Postcode", "Amount"],
                       [{"total": 75.86}])

    judged = []

    def fake_code_judge(**_kwargs):
        judged.append(1)
        return ({"plan_compliance_approval": True, "present_result_approval": True,
                 "violated_criteria": [], "feedback": "ok", "suggestions": [],
                 "judge_error": "", "approved": True}, 1)

    monkeypatch.setattr("lakegen.reviewed_coder.phase3_generate_and_execute", fake_generate)
    monkeypatch.setattr("lakegen.reviewed_coder.plan_analysis", _planner(table))
    monkeypatch.setattr(
        "lakegen.reviewed_coder.judge_plan_question_fidelity",
        lambda **_kwargs: ({"approves_question_fidelity": True, "confidence": 1.0,
                            "rationale": "ok", "missing_requirements": [],
                            "unjustified_steps": [], "judge_error": ""}, 1),
    )
    monkeypatch.setattr("lakegen.reviewed_coder.judge_production_code_result", fake_code_judge)

    result = phase3_generate_and_execute_reviewed(
        "Total amount by postcode?", [table], [table], {}, "", llm=object(), pm=object(),
        csv_dir=tmp_path, stage_max_retries=3,
    )

    assert result.rejected_reason == ""
    assert len(generated) == 2
    assert "Execution/preflight error" in generated[1][1]
    assert len(judged) == 1
