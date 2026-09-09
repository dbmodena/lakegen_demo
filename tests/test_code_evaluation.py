from lakegen.code_evaluation import (
    EVALUATION_MARKER,
    evaluate_code_result,
    extract_evaluation_payload,
    summarize_code_evaluations,
)


def test_extracts_structured_evaluation_payload_and_cleans_marker():
    clean, value, error = extract_evaluation_payload(
        f'{EVALUATION_MARKER}[{{"borough":"Bronx","count":2}}]'
    )

    assert value == [{"borough": "Bronx", "count": 2}]
    assert clean == '[{"borough": "Bronx", "count": 2}]'
    assert error is None


def test_structured_payload_stays_complete_while_synthesis_output_is_bounded():
    values = list(range(12))
    clean, value, error = extract_evaluation_payload(
        EVALUATION_MARKER + "[" + ",".join(map(str, values)) + "]"
    )

    assert value == values
    assert clean.startswith("Total items: 12\n[0, 1, 2, 3, 4]")
    assert error is None


def test_number_evaluation_accepts_scalar_with_numeric_tolerance():
    evaluation = evaluate_code_result(
        expected_result_type="number",
        reference_result=[{"correlation": 0.123456789}],
        actual_result=0.12345679,
    )

    assert evaluation["result_type_match"] is True
    assert evaluation["exact_result_match"] is True
    assert evaluation["numeric_match"] is True
    assert evaluation["numeric_absolute_error"] < 1e-8
    assert "cell_accuracy" not in evaluation


def test_number_evaluation_accepts_dynamic_scalar_reference():
    evaluation = evaluate_code_result(
        expected_result_type="number",
        reference_result=638904,
        actual_result=638904,
    )

    assert evaluation["applicable"] is True
    assert evaluation["result_type_match"] is True
    assert evaluation["exact_result_match"] is True


def test_number_evaluation_compares_scalar_reference_to_legacy_record_result():
    evaluation = evaluate_code_result(
        expected_result_type="number",
        reference_result=158.0,
        actual_result=[{"total_adult_shelter_beds": 158}],
    )

    assert evaluation["numeric_match"] is True
    assert evaluation["exact_result_match"] is True


def test_number_evaluation_accepts_equivalent_percentage_text():
    evaluation = evaluate_code_result(
        expected_result_type="number",
        reference_result=[{"percentage": 82}],
        actual_result="82%",
    )

    assert evaluation["numeric_match"] is True
    assert evaluation["exact_result_match"] is True


def test_number_evaluation_normalizes_fractional_percentage_and_currency():
    percentage = evaluate_code_result(
        expected_result_type="number",
        reference_result=0.82,
        actual_result="82%",
    )
    currency = evaluate_code_result(
        expected_result_type="number",
        reference_result=-1234.5,
        actual_result="($1,234.50)",
    )

    assert percentage["representation_equivalent_match"] is True
    assert percentage["numeric_absolute_error"] == 0.0
    assert currency["representation_equivalent_match"] is True


def test_table_single_cell_scalar_is_equivalent_but_not_exact_shape():
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=[{"total": 42}],
        actual_result=42,
    )

    assert evaluation["result_type_match"] is False
    assert evaluation["exact_result_match"] is False
    assert evaluation["representation_equivalent_match"] is True


def test_list_accepts_column_oriented_mapping_as_equivalent_representation():
    evaluation = evaluate_code_result(
        expected_result_type="list",
        reference_result=[{"borough": "Bronx"}, {"borough": "Queens"}],
        actual_result={"borough": ["Bronx", "Queens"]},
    )

    assert evaluation["result_type_match"] is False
    assert evaluation["exact_result_match"] is False
    assert evaluation["representation_equivalent_match"] is True


def test_table_evaluation_ignores_row_order_unless_required():
    reference = [
        {"borough": "Bronx", "count": 2},
        {"borough": "Queens", "count": 4},
    ]
    actual = list(reversed(reference))

    unordered = evaluate_code_result(
        expected_result_type="table",
        reference_result=reference,
        actual_result=actual,
    )
    ordered = evaluate_code_result(
        expected_result_type="table",
        reference_result=reference,
        actual_result=actual,
        expected_description="Rows sorted by count descending.",
    )

    assert unordered["exact_result_match"] is True
    assert unordered["cell_accuracy"] == 1.0
    assert ordered["exact_result_match"] is False
    assert ordered["order_correct"] is False


def test_table_evaluation_accepts_unique_value_equivalent_column_aliases():
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=[
            {"program": "A", "avg_count": 10.0},
            {"program": "B", "avg_count": 20.0},
        ],
        actual_result=[
            {"program": "B", "Average Monthly Case Count": 20.0},
            {"program": "A", "Average Monthly Case Count": 10.0},
        ],
    )

    assert evaluation["exact_result_match"] is True
    assert evaluation["column_aliases"] == {
        "avg_count": "Average Monthly Case Count"
    }


def test_table_evaluation_rejects_ambiguous_value_based_column_aliases():
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=[{"left": 1, "right": 1}],
        actual_result=[{"first": 1, "second": 1}],
    )

    assert evaluation["exact_result_match"] is False
    assert evaluation["column_recall"] == 0.0


def test_partial_table_metrics_penalize_missing_columns_and_wrong_values():
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=[{"borough": "Bronx", "count": 2}],
        actual_result=[{"borough": "Bronx", "count": 3, "extra": "x"}],
    )

    assert evaluation["exact_result_match"] is False
    assert evaluation["column_recall"] == 1.0
    assert evaluation["column_precision"] < 1.0
    assert evaluation["row_f1"] == 0.0
    assert evaluation["cell_accuracy"] == 0.5


def test_table_treats_non_conflicting_extra_columns_as_equivalent_not_exact():
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=[
            {"borough": "Bronx", "community_plaza_count": 2},
            {"borough": "Queens", "community_plaza_count": 1},
        ],
        actual_result=[
            {"BoroName": "Queens", "plaza_count": 1, "source": "DOT"},
            {"BoroName": "Bronx", "plaza_count": 2, "source": "DOT"},
        ],
    )

    assert evaluation["exact_result_match"] is False
    assert evaluation["representation_equivalent_match"] is True
    assert evaluation["column_aliases"] == {
        "borough": "BoroName",
        "community_plaza_count": "plaza_count",
    }
    assert evaluation["ignored_actual_columns"] == ["source"]


def test_table_extra_columns_do_not_hide_wrong_requested_values():
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=[{"borough": "Bronx", "count": 2}],
        actual_result=[{"borough": "Bronx", "count": 99, "source": "DOT"}],
    )

    assert evaluation["exact_result_match"] is False
    assert evaluation["representation_equivalent_match"] is False


def test_contract_declares_order_keys_columns_and_limit():
    reference = [
        {"borough": "Queens", "count": 4},
        {"borough": "Bronx", "count": 2},
    ]
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=reference,
        actual_result=list(reversed(reference)),
        evaluation_contract={
            "result_kind": "table",
            "required_dimensions": ["borough"],
            "required_measures": ["count"],
            "key_columns": ["borough"],
            "ordering": [{"field": "count", "direction": "desc"}],
            "limit": 2,
        },
    )

    assert evaluation["key_columns"] == ["borough"]
    assert evaluation["requirement_checks"] == {
        "result_type": True,
        "required_columns": True,
        "row_count": True,
        "ordering": False,
        "limit": True,
    }
    assert evaluation["requirement_pass_rate"] == 0.8
    assert evaluation["representation_equivalent_match"] is False


def test_contract_applies_per_column_numeric_tolerance():
    evaluation = evaluate_code_result(
        expected_result_type="table",
        reference_result=[{"group": "A", "average": 10.0}],
        actual_result=[{"group": "A", "average": 10.05}],
        evaluation_contract={
            "key_columns": ["group"],
            "numeric_tolerances": {"average": {"abs": 0.1, "rel": 0}},
        },
    )

    assert evaluation["exact_result_match"] is True
    assert evaluation["cell_accuracy"] == 1.0


def test_summary_reports_robust_numeric_error_statistics():
    results = [
        {"result": {"code_evaluation": {
            "applicable": True, "numeric_absolute_error": value,
            "numeric_relative_error": value, "requirement_pass_rate": 0.5,
        }}}
        for value in (1.0, 2.0, 1e200)
    ]

    summary = summarize_code_evaluations(results)

    assert summary["numeric_absolute_error_robust"]["median"] == 2.0
    assert summary["numeric_absolute_error_robust"]["p95"] > 1e100
    assert summary["mean_requirement_pass_rate"] == 0.5


def test_list_uses_item_metrics_instead_of_table_metrics():
    evaluation = evaluate_code_result(
        expected_result_type="list",
        reference_result=[{"rotation": 10}, {"rotation": 20}],
        actual_result=[20, 10],
    )

    assert evaluation["exact_result_match"] is True
    assert evaluation["item_f1"] == 1.0
    assert "column_f1" not in evaluation
    assert "cell_accuracy" not in evaluation


def test_batch_summary_aggregates_only_applicable_code_evaluations():
    results = [
        {"result": {"code_evaluation": {
            "applicable": True,
            "generation_success": True,
            "execution_success": True,
            "structured_output_valid": True,
            "result_type_match": True,
            "exact_result_match": True,
            "pass_at_1": True,
            "success_within_3": True,
            "attempt_count": 1,
            "generation_attempt_count": 1,
            "execution_attempt_count": 2,
            "any_attempt_execution_success": True,
            "final_execution_success": True,
            "column_f1": 1.0,
            "row_f1": 1.0,
            "cell_accuracy": 1.0,
        }}},
        {"result": {"code_evaluation": {
            "applicable": True,
            "generation_success": True,
            "execution_success": False,
            "structured_output_valid": False,
            "result_type_match": False,
            "exact_result_match": False,
            "pass_at_1": False,
            "success_within_3": False,
            "attempt_count": 3,
            "generation_attempt_count": 2,
            "execution_attempt_count": 3,
            "any_attempt_execution_success": True,
            "final_execution_success": False,
            "error_category": "execution_error",
        }}},
        {"result": {"code_evaluation": {"applicable": False}}},
    ]

    summary = summarize_code_evaluations(results)

    assert summary["batch_case_count"] == 3
    assert summary["applicable_case_count"] == 2
    assert summary["execution_success_rate"] == 0.5
    assert summary["exact_result_match_rate"] == 0.5
    assert summary["supported_result_rate"] == 0.5
    assert summary["ambiguous_result_rate"] == 0.0
    assert summary["evaluation_dispositions"] == {
        "correct": 1,
        "incorrect": 1,
    }
    assert summary["mean_attempts"] == 2.0
    assert summary["mean_generation_attempts"] == 1.5
    assert summary["mean_execution_attempts"] == 2.5
    assert summary["any_attempt_execution_success_rate"] == 1.0
    assert summary["final_execution_success_rate"] == 0.5
    assert summary["error_categories"] == {"execution_error": 1}


def test_batch_summary_separates_blocked_and_not_evaluated():
    results = [
        {"result": {"code_evaluation": {
            "applicable": True, "attempt_count": 0,
            "evaluation_disposition": "blocked",
        }}},
        {"result": {"code_evaluation": {
            "applicable": True, "attempt_count": 0,
            "evaluation_disposition": "not_evaluated",
        }}},
    ]
    summary = summarize_code_evaluations(results)
    assert summary["evaluation_dispositions"] == {
        "blocked": 1, "not_evaluated": 1,
    }
