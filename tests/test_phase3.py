from lakegen.column_resolution import (
    resolve_column_name,
    resolve_generated_code_columns,
)
import json
import pandas as pd

from lakegen.phases.phase3 import (
    _build_coder_tables_info,
    _detect_tabpfn_intent,
    _exact_column_labels,
    _execute_code,
    _recover_fenced_agent_code,
    _tabpfn_enabled,
)
from lakegen.experiment_config import CoderContextLevel
from lakegen.agent_tools.tools_p3 import (
    CoderLifecycle,
    P3State,
    Phase3ToolsManager,
    classify_execution_error,
)


def test_execute_code_treats_reported_missing_columns_as_error(tmp_path):
    output, error, _code = _execute_code(
        'print("Missing required columns: {\'district\'}")',
        run_dir=tmp_path,
    )

    assert output is None
    assert error == "Missing required columns: {'district'}"


def test_execute_code_accepts_normal_stdout(tmp_path):
    output, error, _code = _execute_code(
        'print("district 1: 42 removals")',
        run_dir=tmp_path,
    )

    assert output == "district 1: 42 removals"
    assert error is None


def test_execute_code_identifies_the_forbidden_fragment(tmp_path):
    output, error, _code = _execute_code(
        "import sys\nprint('unused import')",
        run_dir=tmp_path,
    )

    assert output is None
    assert "'import sys'" in error
    assert "Remove it completely" in error


def _agentic_tools(tmp_path, *, execute=None, question=""):
    state = P3State()
    manager = Phase3ToolsManager(
        state,
        tables=["table.csv"],
        csv_dir=tmp_path,
        run_dir=tmp_path / "run",
        evaluation_result_type=None,
        question=question,
        resolve_code=lambda code, _tables, _csv_dir: (code, None),
        execute_code=execute or (lambda code, **_kwargs: ("value: 42", None, code)),
        extract_payload=lambda raw: (raw, None, None),
    )
    return state, manager


def test_agentic_coder_requires_inspection_and_self_review(tmp_path):
    state, manager = _agentic_tools(tmp_path)

    assert '"ok": true' in manager.run_code("print(42)")
    try:
        manager.finish_code(
            "verified", "verified", "not_applicable", "not_applicable",
            "verified", {}, "The output contains the requested numeric measure.",
        )
    except ValueError as exc:
        assert "inspect_result" in str(exc)
    else:
        raise AssertionError("finish_code should require result inspection")

    assert '"ok": true' in manager.inspect_result()
    payload = manager.finish_code(
        "verified", "verified", "not_applicable", "not_applicable",
        "verified", {}, "The output contains the requested numeric measure.",
    )
    assert payload.startswith("FINAL_PAYLOAD:")
    assert state.finished is True


def test_agentic_coder_returns_structured_execution_error(tmp_path):
    state, manager = _agentic_tools(
        tmp_path,
        execute=lambda code, **_kwargs: (
            None, "KeyError: 'missing_total'", code
        ),
    )

    response = manager.run_code("print(df['missing_total'])")

    assert '"category": "missing_column"' in response
    assert state.execution_error["stage"] == "execution"
    assert state.execution_error["column"] == "missing_total"


def test_fenced_code_without_tool_call_uses_normal_run_and_inspection(tmp_path):
    state, manager = _agentic_tools(
        tmp_path,
        execute=lambda code, **_kwargs: (
            '__LAKEGEN_EVAL_JSON__42', None, code
        ),
    )
    manager.evaluation_result_type = "number"
    manager.extract_payload = lambda raw: ("42", 42, None)
    response = """assistant: ```python
import json
evaluation_value = 42
print('__LAKEGEN_EVAL_JSON__' + json.dumps(evaluation_value))
```"""

    recovered = _recover_fenced_agent_code(response, manager, state)

    assert recovered is True
    assert state.run_count == 1
    assert state.inspected_version == state.result_version == 1
    assert state.lifecycle == CoderLifecycle.READY_TO_FINISH
    assert state.stop_reason == "recovered_fenced_code_without_run_code_call"


def test_execution_error_classifier_marks_security_failures_non_retryable():
    error = classify_execution_error(
        "Security Error: forbidden code fragment 'import os'."
    )

    assert error["category"] == "security_error"
    assert error["retryable"] is False


def test_finalization_recovery_requires_structured_latest_inspection():
    state = P3State(
        raw_result="42", structured_result=42, result_version=1,
        inspected_version=0,
    )
    assert state.ready_for_finalization() is False

    state.inspected_version = 1
    state.lifecycle = CoderLifecycle.READY_TO_FINISH
    assert state.ready_for_finalization() is True

    state.error = "execution failed"
    assert state.ready_for_finalization() is False

    state.error = None
    state.coverage_warnings = ["coverage_shortfall"]
    assert state.ready_for_finalization() is False


def test_recovery_finalizes_only_warning_free_latest_inspection(tmp_path):
    state, manager = _agentic_tools(tmp_path)
    manager.evaluation_result_type = "number"
    manager.extract_payload = lambda raw: ("42", 42, None)

    assert '"ok": true' in manager.run_code("print(42)")
    inspection = json.loads(manager.inspect_result())
    assert inspection["state"] == "ready_to_finish"

    manager.recover_finish("Agent exhausted its protocol budget after inspection.")

    assert state.finished is True
    assert state.lifecycle == CoderLifecycle.FINISHED
    assert state.finalization_mode == "system_recovery"
    assert state.review["semantic_self_review"] == "not_available"


def test_recovery_does_not_override_explicit_needs_revision(tmp_path):
    state, manager = _agentic_tools(tmp_path)
    manager.evaluation_result_type = "table"
    manager.question = "Show the top 3 agencies"
    manager.extract_payload = lambda raw: (raw, [{"agency": "A"}], None)

    manager.run_code("print('one item')")
    inspection = json.loads(manager.inspect_result())

    assert inspection["state"] == "needs_revision"
    try:
        manager.recover_finish("Must not recover a warned result.")
    except ValueError as exc:
        assert "warning-free" in str(exc)
    else:
        raise AssertionError("Recovery must reject a result that needs revision")
    assert state.finished is False


def test_coverage_counts_nested_top_five_as_five_semantic_items(tmp_path):
    _state, manager = _agentic_tools(tmp_path, question="Show the top 5 agencies")
    requirements, warnings, facts = manager._coverage(
        {"top_5": ["A", "B", "C", "D", "E"]}
    )

    assert facts["semantic_item_count"] == 5
    assert "return 5 ranked semantic items" in requirements
    assert warnings == []


def test_top_n_ignores_unrelated_number_words(tmp_path):
    _state, manager = _agentic_tools(
        tmp_path,
        question=(
            "Which top five districts have the highest combined values across "
            "the two disciplinary measures?"
        ),
    )
    requirements, warnings, facts = manager._coverage(
        [{"district": item} for item in range(5)]
    )

    assert facts["expected_ranked_items"] == 5
    assert requirements == ["return 5 ranked semantic items"]
    assert warnings == []


def test_coverage_recognizes_nyc_borough_codes(tmp_path):
    _state, manager = _agentic_tools(
        tmp_path, question="Show the result for each borough"
    )
    requirements, warnings, facts = manager._coverage([
        {"BORO": code, "value": 1} for code in ["K", "M", "Q", "R", "X"]
    ])

    assert "cover all five NYC boroughs, including valid zero-count groups" in requirements
    assert set(facts["boroughs_present"]) == {
        "bronx", "brooklyn", "manhattan", "queens", "staten island"
    }
    assert warnings == []


def test_coverage_problem_blocks_finish_but_allows_corrected_run(tmp_path):
    outputs = iter([
        ('[{"name": "A"}]', [{"name": "A"}]),
        ('[{"name": "A"}, {"name": "B"}, {"name": "C"}]',
         [{"name": "A"}, {"name": "B"}, {"name": "C"}]),
    ])

    def execute(code, **_kwargs):
        raw, _structured = next(outputs)
        return "__PAYLOAD__" + raw, None, code

    state, manager = _agentic_tools(
        tmp_path, execute=execute, question="Which three agencies have the most?"
    )
    manager.evaluation_result_type = "table"
    manager.extract_payload = lambda raw: (
        raw.removeprefix("__PAYLOAD__"),
        json.loads(raw.removeprefix("__PAYLOAD__")),
        None,
    )
    manager.run_code("print('first')")
    first = json.loads(manager.inspect_result())
    assert first["profile"]["correction_required"] is True
    try:
        manager.finish_code(
            "verified", "verified", "verified", "verified", "verified",
            {}, "Only one item was returned and requires correction.",
        )
    except ValueError as exc:
        assert "correct the code" in str(exc)
    else:
        raise AssertionError("coverage warning should block finish_code")

    assert '"ok": true' in manager.run_code("print('corrected')")
    second = json.loads(manager.inspect_result())
    assert second["profile"]["correction_required"] is False
    requirement = state.coverage_requirements[0]
    payload = manager.finish_code(
        "verified", "verified", "verified", "verified", "verified",
        {requirement: "Three items are present."},
        "The corrected result contains all three ranked agencies.",
    )
    assert payload.startswith("FINAL_PAYLOAD:")


def test_inspect_table_is_limited_and_returns_exact_load_command(tmp_path):
    (tmp_path / "table.csv").write_text(
        "Borough,Created Date\nManhattan,2024-01-01\nBrooklyn,2024-02-01\n",
        encoding="utf-8",
    )
    _state, manager = _agentic_tools(tmp_path)

    first = json.loads(manager.inspect_table("table.csv", "Borough,Created Date"))
    second = json.loads(manager.inspect_table("table.csv", "Borough"))

    assert first["ok"] is True
    assert first["columns"] == ["Borough", "Created Date"]
    assert "pd.read_csv" in first["load_command"]
    assert second["ok"] is True
    assert second["cached_sample"] is True
    assert "run_code" in second["next_action"]


def test_inspect_table_accepts_exact_selected_absolute_path(tmp_path):
    table = tmp_path / "table.csv"
    table.write_text("Borough,Value\nBrooklyn,1\n", encoding="utf-8")
    state, manager = _agentic_tools(tmp_path)

    response = json.loads(manager.inspect_table(str(table), "Borough"))

    assert response["ok"] is True
    assert response["table"] == "table.csv"
    assert state.inspected_tables == {"table.csv"}


def test_inspect_table_profiles_new_columns_from_same_cached_sample(tmp_path):
    (tmp_path / "table.csv").write_text(
        "Borough,Year,Value\nManhattan,2023,1\nBrooklyn,2024,2\n",
        encoding="utf-8",
    )
    state, manager = _agentic_tools(tmp_path)

    first = json.loads(manager.inspect_table("table.csv", "Borough"))
    second = json.loads(manager.inspect_table("table.csv", "Year"))

    assert first["cached_sample"] is False
    assert second["cached_sample"] is True
    assert "Year" in second["requested_column_profiles"]
    assert state.table_profiled_columns["table.csv"] == {"Borough", "Year"}


def test_reject_tables_requires_evidence_and_returns_structured_marker(tmp_path):
    (tmp_path / "table.csv").write_text("Year\n2020\n", encoding="utf-8")
    state, manager = _agentic_tools(tmp_path)
    manager.inspect_table("table.csv", "Year")

    response = manager.reject_tables(
        "The selected table cannot answer the requested 2023 question.",
        ["records for year 2023"],
        "The cached Year profile contains only the value 2020.",
    )

    assert response.startswith("REJECT_TABLES:")
    assert state.rejected_reason.startswith("The selected table")
    assert state.rejection_details["missing_requirements"] == ["records for year 2023"]


def test_reject_tables_blocks_year_column_claim_when_metadata_has_period(tmp_path):
    (tmp_path / "table.csv").write_text("Value\n1\n", encoding="utf-8")
    state = P3State()
    manager = Phase3ToolsManager(
        state,
        tables=["table.csv"],
        csv_dir=tmp_path,
        run_dir=tmp_path / "run",
        evaluation_result_type=None,
        question="How many records were there in fiscal year 2023?",
        table_metadata={"table.csv": {"title": "Fiscal Year 2023 records"}},
        resolve_code=lambda code, _tables, _csv_dir: (code, None),
        execute_code=lambda code, **_kwargs: ("1", None, code),
        extract_payload=lambda raw: (raw, None, None),
    )
    manager.inspect_table("table.csv")

    try:
        manager.reject_tables(
            "The selected table cannot answer the fiscal year 2023 question.",
            ["a fiscal year column for 2023"],
            "The inspected schema does not contain a dedicated fiscal year column.",
        )
    except ValueError as exc:
        assert "rejection_not_proven" in str(exc)
    else:
        raise AssertionError("Metadata-backed periods must block table rejection")
    assert state.rejected_reason == ""
    assert state.lifecycle == CoderLifecycle.NEEDS_REVISION


def test_reject_tables_blocks_missing_borough_rows_as_insufficient_evidence(tmp_path):
    (tmp_path / "table.csv").write_text(
        "Borough,Value\nBrooklyn,1\nQueens,2\n", encoding="utf-8"
    )
    state, manager = _agentic_tools(tmp_path, question="Show each borough")
    manager.inspect_table("table.csv", "Borough")

    try:
        manager.reject_tables(
            "The selected table contains only two boroughs rather than all five.",
            ["rows for all five NYC boroughs"],
            "The Borough profile contains only Brooklyn and Queens in the sample.",
        )
    except ValueError as exc:
        assert "rejection_not_proven" in str(exc)
    else:
        raise AssertionError("Missing borough rows alone must not reject tables")
    assert state.rejected_reason == ""


def test_diagnostic_output_is_retryable_and_not_inspectable_as_result(tmp_path):
    state, manager = _agentic_tools(
        tmp_path,
        execute=lambda code, **_kwargs: ("Columns: ['A', 'B']", None, code),
    )
    manager.evaluation_result_type = "table"
    manager.extract_payload = lambda raw: (raw, None, "missing payload")

    response = json.loads(manager.run_code("print('columns')"))

    assert response["ok"] is False
    assert response["error"]["category"] == "diagnostic_output"
    assert state.raw_result is None
    assert json.loads(manager.inspect_result())["ok"] is False


def test_run_code_rejects_hallucinated_table_path_with_allowed_command(tmp_path):
    (tmp_path / "table.csv").write_text("value\n42\n", encoding="utf-8")
    state, manager = _agentic_tools(tmp_path)

    response = json.loads(manager.run_code(
        "import pandas as pd\ndf = pd.read_csv('/wrong/table.csv')\nprint(df)"
    ))

    assert response["error"]["category"] == "invalid_table_path"
    assert "table.csv" in response["error"]["allowed_load_commands"][0]
    assert state.finished is False


def test_column_resolver_preserves_exact_names_and_normalizes_generated_aliases():
    columns = [
        "AGENCY EXPENDITURES",
        "Construction_Year",
        "FISCAL YEAR",
        "AGENCY NAME",
        "TOTAL AMOUNT",
    ]

    assert resolve_column_name("agency_expenditures", columns) == "AGENCY EXPENDITURES"
    assert resolve_column_name("construction year", columns) == "Construction_Year"
    assert resolve_column_name("fiscal_year", columns) == "FISCAL YEAR"
    assert resolve_column_name("agency_name", columns) == "AGENCY NAME"
    assert resolve_column_name("total_amount", columns) == "TOTAL AMOUNT"
    assert resolve_column_name("agy_nm", columns) is None


def test_generated_code_preflight_rewrites_column_contexts_and_validates_required():
    result = resolve_generated_code_columns(
        "required_col = 'construction_year'\n"
        "required = {'agency_name', 'total_amount'}\n"
        "out = df.groupby('agency_name')['total_amount'].sum()\n",
        ["Construction_Year", "AGENCY NAME", "TOTAL AMOUNT"],
    )

    assert "Construction_Year" in result.code
    assert "AGENCY NAME" in result.code
    assert "TOTAL AMOUNT" in result.code
    assert result.unresolved_required == ()

    invalid = resolve_generated_code_columns(
        "required_cols = {'invented_metric'}", ["Real Metric"]
    )
    assert invalid.unresolved_required == ("invented_metric",)


def test_historical_trend_does_not_trigger_tabpfn_forecasting():
    assert _detect_tabpfn_intent("Show the allocation trend over time") is None
    assert _detect_tabpfn_intent("Forecast allocation for next year") == "forecasting"


def test_tabpfn_is_disabled_by_default_and_can_be_enabled(monkeypatch):
    monkeypatch.delenv("LAKEGEN_ENABLE_TABPFN", raising=False)
    assert _tabpfn_enabled() is False

    monkeypatch.setenv("LAKEGEN_ENABLE_TABPFN", "true")
    assert _tabpfn_enabled() is True


def test_code_generator_schema_uses_exact_file_columns_not_solr_field_aliases():
    frame = pd.DataFrame({"Mbps Bandwidth": [10], "Construction Year": [2020]})

    labels = _exact_column_labels(frame)

    assert labels[0].startswith("Mbps Bandwidth(")
    assert labels[1].startswith("Construction Year(")
    assert all("mbps_bandwidth" not in label for label in labels)


def test_coder_context_levels_control_table_metadata(tmp_path):
    table = tmp_path / "table.csv"
    table.write_text("City,Population\nBologna,390000\n", encoding="utf-8")

    full = _build_coder_tables_info(
        [table.name], tmp_path, CoderContextLevel.FULL
    )
    schema_only = _build_coder_tables_info(
        [table.name], tmp_path, CoderContextLevel.SCHEMA_ONLY
    )
    minimal = _build_coder_tables_info(
        [table.name], tmp_path, CoderContextLevel.MINIMAL
    )

    assert "LOAD:" in full and "Columns (2):" in full and "Sample:" in full
    assert "Bologna" in full
    assert "LOAD:" in schema_only and "Columns (2):" in schema_only
    assert "Sample:" not in schema_only and "Bologna" not in schema_only
    assert "LOAD:" in minimal
    assert "Columns" not in minimal and "Sample:" not in minimal
