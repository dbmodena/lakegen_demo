from lakegen.column_lineage import (
    compose_feedback, derive_plan_from_operation_trace, resolve_plan_columns,
)


def test_derive_plan_from_operation_trace_builds_one_flat_step():
    op_trace = {"used_tables": ["t.parquet"], "used_columns": ["Postcode", "Amount"]}
    steps = derive_plan_from_operation_trace(op_trace)
    assert len(steps) == 1
    assert steps[0].tables == ["t.parquet"]
    assert steps[0].reads == ["Postcode", "Amount"]
    assert steps[0].produces == []


def test_derive_plan_from_operation_trace_empty_when_nothing_used():
    assert derive_plan_from_operation_trace({}) == []


def test_valid_plan_has_no_violations():
    op_trace = {"used_tables": ["t.parquet"], "used_columns": ["Postcode", "Amount written-off"]}
    schemas = {"t.parquet": ["Postcode", "Amount written-off", "Financial Year"]}
    violations = resolve_plan_columns(derive_plan_from_operation_trace(op_trace), schemas)
    assert violations == []


def test_hallucinated_column_is_a_violation():
    op_trace = {"used_tables": ["t.parquet"], "used_columns": ["TotallyMadeUpColumn"]}
    schemas = {"t.parquet": ["Postcode", "Amount written-off"]}
    violations = resolve_plan_columns(derive_plan_from_operation_trace(op_trace), schemas)
    assert len(violations) == 1
    assert violations[0].category == "unknown_column"
    assert violations[0].column == "TotallyMadeUpColumn"


def test_suggestion_offered_for_a_near_miss():
    op_trace = {"used_tables": ["t.parquet"], "used_columns": ["postcode"]}  # wrong case
    schemas = {"t.parquet": ["Postcode"]}
    violations = resolve_plan_columns(derive_plan_from_operation_trace(op_trace), schemas)
    # case-insensitive fuzzy match should resolve cleanly, not violate
    assert violations == []


def test_compose_feedback_empty_when_no_violations():
    assert compose_feedback([]) == ""


def test_compose_feedback_names_the_column_and_category():
    op_trace = {"used_tables": ["t.parquet"], "used_columns": ["Bogus"]}
    schemas = {"t.parquet": ["Real"]}
    violations = resolve_plan_columns(derive_plan_from_operation_trace(op_trace), schemas)
    feedback = compose_feedback(violations)
    assert "Bogus" in feedback
    assert "hallucinated" in feedback
