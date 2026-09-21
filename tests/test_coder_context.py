import json

from lakegen.coder_context import CoderContext, infer_output_shape


SECRET = "BENCHMARK_SECRET_DO_NOT_LEAK"


def test_coder_context_is_explicit_allowlist_and_drops_benchmark_secret():
    context = CoderContext.build(
        question="Show the top 3 boroughs by count.",
        selected_tables=["runtime.parquet"],
        table_metadata={
            "runtime.parquet": {
                "title": "Runtime title",
                "publisher": "Runtime publisher",
                "resource_name": "Runtime resource",
                "description": "Observed resource",
                "columns": [{"name": "borough", "description": "Borough"}],
                "reference_result": SECRET,
                "benchmark_label": SECRET,
            }
        },
        selection_plan={
            "semantic_plan": {
                "filters": [], "dimensions": [], "measures": [],
                "output_columns": ["borough", "count"],
                "reference_code": SECRET,
                "evidence_map": [{"evidence": {"nested": SECRET}}],
            },
            "gold_tables": SECRET,
        },
        source_payload={
            "question": "Show the top 3 boroughs by count.",
            "SOURCE_REFERENCE_RESULT": SECRET,
            "SOURCE_REFERENCE_CODE": SECRET,
            "SOURCE_EXPECTED_RESULT_DESCRIPTION": SECRET,
            "SOURCE_TABLE_ALIASES": {"Table_0": SECRET},
            "SOURCE_GOLD_VALIDATION": SECRET,
        },
        execution_error={"category": "runtime_error"},
    )

    serialized = json.dumps(context.__dict__, default=str)
    assert SECRET not in serialized
    assert "reference_result" not in context.table_metadata["runtime.parquet"]
    assert context.table_metadata["runtime.parquet"]["publisher"] == "Runtime publisher"
    assert context.table_metadata["runtime.parquet"]["resource_name"] == "Runtime resource"
    assert "reference_code" not in context.selection_plan["semantic_plan"]
    assert "evidence_map" not in context.selection_plan["semantic_plan"]
    assert context.audit()["reference_accessed_by_coder"] is False
    assert context.audit()["context_filtered"] is True


def test_output_shape_is_question_derived_only():
    assert infer_output_shape("Show the top 3 boroughs by count") == {
        "result_type": "table", "ordered": True, "row_limit": 3,
        "source": "derived_from_question",
    }
    assert infer_output_shape("How many records are there?")["result_type"] == "number"


def test_coder_context_preserves_requirement_ledger_evidence():
    ledger = [
        {
            "kind": "filter", "request": "year = 2022", "status": "bound",
            "evidence": {"table": "crimes.csv", "columns": ["Year"]},
            "role": "final", "reference_answer": SECRET,
        },
    ]
    context = CoderContext.build(
        question="How many crimes in 2022?", selected_tables=["crimes.csv"],
        table_metadata={}, selection_plan={"requirement_ledger": ledger},
    )
    assert context.selection_plan["requirement_ledger"] == [{
        "kind": "filter", "request": "year = 2022", "status": "bound",
        "evidence": {"table": "crimes.csv", "columns": ["Year"]},
        "role": "final",
    }]
    assert SECRET not in json.dumps(context.__dict__)


def test_coder_context_drops_non_list_requirement_ledger():
    context = CoderContext.build(
        question="How many crimes in 2022?", selected_tables=["crimes.csv"],
        table_metadata={}, selection_plan={"requirement_ledger": "not-a-list"},
    )
    assert context.selection_plan["requirement_ledger"] == []


def test_coder_context_preserves_explicit_requirements_for_fallback():
    requirements = {
        "filters": [{"table": "runtime.csv", "column": "status", "value": "completed"}],
        "grouping": ["borough"], "measures": ["count rows"],
        "result_type": "table", "reference_result": SECRET,
    }
    context = CoderContext.build(
        question="Count completed records by borough", selected_tables=["runtime.csv"],
        table_metadata={}, selection_plan={"requirements": requirements},
    )
    assert context.selection_plan["requirements"] == {
        key: value for key, value in requirements.items() if key != "reference_result"
    }
    assert SECRET not in json.dumps(context.__dict__)


def test_recursive_sanitizer_removes_secret_without_source_payload_hint():
    context = CoderContext.build(
        question="Count completed records",
        selected_tables=["runtime.csv"],
        table_metadata={"runtime.csv": {
            "title": SECRET,
            "columns": [{"name": "status", "description": SECRET}],
        }},
        selection_plan={"semantic_plan": {
            "filters": [{
                "table": "runtime.csv", "column": "status",
                "operator": "equals", "value": SECRET,
                "evidence": {"nested": SECRET},
            }],
            "evidence_map": [{"nested": {"value": SECRET}}],
            "measures": [], "dimensions": [], "joins": [],
        }},
    )
    assert SECRET not in json.dumps(context.__dict__, default=str)


def test_semantic_plan_is_preserved_except_for_unauthorized_fields():
    semantic_plan = {
        "filters": [{
            "requirement": "completed records", "table": "runtime.csv",
            "column": "status", "operator": "equals", "value": "completed",
            "evidence": "observed runtime category",
        }],
        "temporal_filters": [],
        "dimensions": [{
            "output": "borough_name", "table": "runtime.csv",
            "column": "borough", "evidence": "observed runtime column",
        }],
        "measures": [{
            "output": "average_value", "operation": "mean",
            "table": "runtime.csv", "columns": ["value"], "distinct": False,
            "evidence": "observed numeric column",
        }],
        "joins": [], "ordering": [{"output": "average_value", "direction": "descending"}],
        "limit": 5, "output_columns": ["borough_name", "average_value"],
        "null_policy": "exclude null measures", "table_roles": {"runtime.csv": "facts"},
        "reference_result": SECRET,
    }
    context = CoderContext.build(
        question="Top 5 completed records by borough",
        selected_tables=["runtime.csv"], table_metadata={},
        selection_plan={"semantic_plan": semantic_plan},
    )
    expected = dict(semantic_plan)
    expected.pop("reference_result")
    assert context.selection_plan["semantic_plan"] == expected


def test_coder_brief_is_allowlisted_without_benchmark_fields():
    context = CoderContext.build(
        question="Count rows by borough",
        selected_tables=["runtime.csv"], table_metadata={},
        selection_plan={"coder_brief": {
            "tables": ["runtime.csv"],
            "selected_columns": {"runtime.csv": ["borough"]},
            "task": {"grouping": ["borough"]}, "operations": ["count rows"],
            "filters": [], "result_type": "table", "joins": [],
            "reference_result": SECRET,
        }},
    )
    assert context.selection_plan["coder_brief"]["selected_columns"] == {
        "runtime.csv": ["borough"]
    }
    assert "reference_result" not in context.selection_plan["coder_brief"]
    assert SECRET not in json.dumps(context.__dict__, default=str)
