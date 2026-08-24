from lakegen.column_resolution import (
    resolve_column_name,
    resolve_generated_code_columns,
)
import pandas as pd

from lakegen.phases.phase3 import (
    _build_coder_tables_info,
    _detect_tabpfn_intent,
    _exact_column_labels,
    _execute_code,
    _tabpfn_enabled,
)
from lakegen.experiment_config import CoderContextLevel


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
