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
    assert "reference_code" not in context.selection_plan["semantic_plan"]
    assert context.audit()["reference_accessed_by_coder"] is False
    assert context.audit()["context_filtered"] is True


def test_output_shape_is_question_derived_only():
    assert infer_output_shape("Show the top 3 boroughs by count") == {
        "result_type": "table", "ordered": True, "row_limit": 3,
        "source": "derived_from_question",
    }
    assert infer_output_shape("How many records are there?")["result_type"] == "number"
