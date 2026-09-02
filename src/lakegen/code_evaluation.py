"""Deterministic functional evaluation for generated data-analysis code."""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


EVALUATION_MARKER = "__LAKEGEN_EVAL_JSON__"
_FLOAT_REL_TOL = 1e-6
_FLOAT_ABS_TOL = 1e-8
_ORDER_WORDS = re.compile(
    r"\b(sorted|ordered|order|ascending|descending|top|highest|lowest|rank(?:ed)?)\b",
    re.IGNORECASE,
)
_ORDER_NOT_REQUIRED = re.compile(
    r"\border\s+(?:is\s+)?not\s+required\b", re.IGNORECASE
)


def evaluation_output_instruction(expected_result_type: str) -> str:
    """Return the benchmark-only structured-output contract for the coder."""

    return (
        "\n\n[FUNCTIONAL EVALUATION OUTPUT — REQUIRED]\n"
        f"The expected result type is `{expected_result_type}`. This run is part of "
        "an end-to-end API experiment. Compute the answer normally, but preserve the "
        "COMPLETE, untruncated answer in a JSON-serializable value named "
        "`evaluation_value`. For a table use "
        "`dataframe.to_dict(orient='records')`; for a number use the scalar; for a "
        "list use the complete list. Convert NumPy scalar values with `.item()` when "
        "needed. Your final and only print must be exactly equivalent to:\n"
        f"`print('{EVALUATION_MARKER}' + json.dumps(evaluation_value, default=str))`\n"
        "Import `json`. Do not truncate, summarize, round, or hardcode "
        "`evaluation_value`.\n"
    )


def extract_evaluation_payload(stdout: str) -> tuple[str, Any | None, str | None]:
    """Extract the marked JSON value and return clean output, value, and error."""

    marker_position = stdout.rfind(EVALUATION_MARKER)
    if marker_position < 0:
        return stdout, None, "structured evaluation marker was not emitted"

    prefix = stdout[:marker_position].rstrip()
    encoded = stdout[marker_position + len(EVALUATION_MARKER):].strip()
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as exc:
        return stdout, None, f"invalid structured evaluation JSON: {exc}"

    if isinstance(value, list) and len(value) > 10:
        clean = (
            f"Total items: {len(value)}\n"
            + json.dumps(value[:5], ensure_ascii=False, default=str)
        )
    else:
        clean = json.dumps(value, ensure_ascii=False, default=str)
    if prefix:
        clean = f"{prefix}\n{clean}"
    return clean, value, None


def _normalized_column(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _order_required(description: str) -> bool:
    return bool(
        _ORDER_WORDS.search(description)
        and not _ORDER_NOT_REQUIRED.search(description)
    )


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or _is_null(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        negative = stripped.startswith("(") and stripped.endswith(")")
        if negative:
            stripped = stripped[1:-1].strip()
        stripped = re.sub(r"^[€£$]\s*", "", stripped)
        if stripped.endswith("%"):
            stripped = stripped[:-1].strip()
        try:
            number = float(stripped)
            return -number if negative else number
        except ValueError:
            return None
    return None


def _numeric_candidates(value: Any) -> list[float]:
    """Return lossless numeric interpretations for common result formats."""

    number = _numeric(value)
    if number is None:
        return []
    candidates = [number]
    if isinstance(value, str) and value.strip().endswith("%"):
        candidates.append(number / 100.0)
    return candidates


def _values_equal(
    left: Any,
    right: Any,
    *,
    rel_tol: float = _FLOAT_REL_TOL,
    abs_tol: float = _FLOAT_ABS_TOL,
) -> bool:
    if _is_null(left) or _is_null(right):
        return _is_null(left) and _is_null(right)
    left_numbers = _numeric_candidates(left)
    right_numbers = _numeric_candidates(right)
    if left_numbers and right_numbers:
        return any(
            math.isclose(
                left_number,
                right_number,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            for left_number in left_numbers
            for right_number in right_numbers
        )
    return str(left).strip() == str(right).strip()


def _records(value: Any, expected_columns: Sequence[str]) -> list[dict[str, Any]] | None:
    if isinstance(value, Mapping):
        if value and all(isinstance(item, (list, tuple)) for item in value.values()):
            lengths = {len(item) for item in value.values()}
            if len(lengths) == 1:
                return [
                    {str(column): values[index] for column, values in value.items()}
                    for index in range(next(iter(lengths)))
                ]
        return [dict(value)]
    if not isinstance(value, list):
        if len(expected_columns) == 1:
            return [{expected_columns[0]: value}]
        return None
    if not value:
        return []
    if all(isinstance(item, Mapping) for item in value):
        return [dict(item) for item in value]
    if len(expected_columns) == 1:
        return [{expected_columns[0]: item} for item in value]
    return None


def _single_value(value: Any) -> tuple[Any, bool]:
    """Return a scalar candidate and whether it is unambiguously single-valued."""

    if isinstance(value, Mapping):
        if len(value) == 1:
            return _single_value(next(iter(value.values())))
        return value, False
    if isinstance(value, list):
        if len(value) == 1:
            return _single_value(value[0])
        return value, False
    return value, True


def _column_map(
    expected_columns: Sequence[str], actual_columns: Sequence[str]
) -> dict[str, str]:
    actual_by_normalized = {
        _normalized_column(column): column for column in actual_columns
    }
    return {
        expected: actual_by_normalized[_normalized_column(expected)]
        for expected in expected_columns
        if _normalized_column(expected) in actual_by_normalized
    }


def _value_multisets_equal(left: Sequence[Any], right: Sequence[Any]) -> bool:
    """Compare column values without relying on row order."""

    if len(left) != len(right):
        return False
    unmatched = set(range(len(right)))
    for left_value in left:
        match = next(
            (
                index for index in sorted(unmatched)
                if _values_equal(left_value, right[index])
            ),
            None,
        )
        if match is None:
            return False
        unmatched.remove(match)
    return True


def _semantic_column_map(
    expected_rows: Sequence[Mapping[str, Any]],
    actual_rows: Sequence[Mapping[str, Any]],
    expected_columns: Sequence[str],
    actual_columns: Sequence[str],
) -> dict[str, str]:
    """Resolve exact names first, then unique value-equivalent aliases."""

    mapping = _column_map(expected_columns, actual_columns)
    remaining_expected = [
        column for column in expected_columns if column not in mapping
    ]
    used_actual = set(mapping.values())
    remaining_actual = [
        column for column in actual_columns if column not in used_actual
    ]
    if not remaining_expected or len(remaining_expected) > len(remaining_actual):
        return mapping

    candidates: dict[str, list[str]] = {}
    for expected_column in remaining_expected:
        expected_values = [row.get(expected_column) for row in expected_rows]
        candidates[expected_column] = [
            actual_column
            for actual_column in remaining_actual
            if _value_multisets_equal(
                expected_values,
                [row.get(actual_column) for row in actual_rows],
            )
        ]
        if not candidates[expected_column]:
            return mapping

    solutions: list[dict[str, str]] = []

    def search(index: int, current: dict[str, str], used: set[str]) -> None:
        if len(solutions) > 1:
            return
        if index == len(remaining_expected):
            solutions.append(dict(current))
            return
        expected_column = remaining_expected[index]
        for actual_column in candidates[expected_column]:
            if actual_column in used:
                continue
            current[expected_column] = actual_column
            used.add(actual_column)
            search(index + 1, current, used)
            used.remove(actual_column)
            current.pop(expected_column, None)

    search(0, {}, set())
    if len(solutions) == 1:
        mapping.update(solutions[0])
    return mapping


def _rows_equal(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    columns: Mapping[str, str],
    tolerances: Mapping[str, tuple[float, float]] | None = None,
) -> bool:
    return all(
        _values_equal(
            expected.get(expected_column),
            actual.get(actual_column),
            rel_tol=(tolerances or {}).get(expected_column, (_FLOAT_REL_TOL, _FLOAT_ABS_TOL))[0],
            abs_tol=(tolerances or {}).get(expected_column, (_FLOAT_REL_TOL, _FLOAT_ABS_TOL))[1],
        )
        for expected_column, actual_column in columns.items()
    )


def _contract_tolerances(contract: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    configured = contract.get("numeric_tolerances", {})
    if not isinstance(configured, Mapping):
        return {}
    result: dict[str, tuple[float, float]] = {}
    for column, value in configured.items():
        if not isinstance(value, Mapping):
            continue
        try:
            result[str(column)] = (
                float(value.get("rel", _FLOAT_REL_TOL)),
                float(value.get("abs", _FLOAT_ABS_TOL)),
            )
        except (TypeError, ValueError):
            continue
    return result


def _requirement_checks(
    *,
    type_match: bool,
    expected_columns: Sequence[str],
    mapped_columns: Mapping[str, str],
    expected_count: int,
    actual_count: int,
    order_required: bool,
    order_correct: bool,
    contract: Mapping[str, Any],
) -> dict[str, bool]:
    """Return stable, machine-readable structural requirement checks."""

    required = contract.get("required_columns")
    if not isinstance(required, list):
        dimensions = contract.get("required_dimensions", [])
        measures = contract.get("required_measures", [])
        required = [
            *(dimensions if isinstance(dimensions, list) else []),
            *(measures if isinstance(measures, list) else []),
        ]
    required_columns = [str(column) for column in required] or list(expected_columns)
    checks = {
        "result_type": type_match,
        "required_columns": all(column in mapped_columns for column in required_columns),
        "row_count": expected_count == actual_count,
    }
    if order_required:
        checks["ordering"] = order_correct
    limit = contract.get("limit")
    if isinstance(limit, int) and limit >= 0:
        checks["limit"] = actual_count == min(limit, expected_count)
    return checks


def evaluate_code_result(
    *,
    expected_result_type: str,
    reference_result: Any,
    actual_result: Any,
    expected_description: str = "",
    evaluation_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a structured generated result with the benchmark reference."""

    contract = dict(evaluation_contract or {})
    result_type = str(contract.get("result_kind") or expected_result_type).casefold()
    if result_type not in {"number", "table", "list"}:
        return {
            "applicable": False,
            "reason": f"unsupported expected_result_type {expected_result_type!r}",
        }

    if result_type == "number":
        expected_value, _reference_scalar_shaped = _single_value(reference_result)
        expected_numbers = _numeric_candidates(expected_value)
        if not expected_numbers:
            return {
                "applicable": False,
                "reason": "number reference_result must contain one numeric value",
            }
        actual_value, scalar_shaped = _single_value(actual_result)
        actual_numbers = _numeric_candidates(actual_value)
        actual_number = actual_numbers[0] if actual_numbers else None
        scalar_tolerance = _contract_tolerances(contract).get(
            "value", (_FLOAT_REL_TOL, _FLOAT_ABS_TOL)
        )
        numeric_match = _values_equal(
            expected_value, actual_value,
            rel_tol=scalar_tolerance[0], abs_tol=scalar_tolerance[1],
        )
        numeric_pairs = [
            (expected, actual)
            for expected in expected_numbers
            for actual in actual_numbers
        ]
        best_pair = min(
            numeric_pairs,
            key=lambda pair: abs(pair[1] - pair[0]),
            default=None,
        )
        absolute_error = (
            abs(best_pair[1] - best_pair[0]) if best_pair is not None else None
        )
        relative_error = (
            absolute_error / abs(best_pair[0])
            if absolute_error is not None and best_pair is not None and best_pair[0] != 0
            else absolute_error
        )
        type_match = scalar_shaped and actual_number is not None
        requirement_checks = {"result_type": type_match, "numeric_value": numeric_match}
        return {
            "applicable": True,
            "expected_result_type": result_type,
            "result_type_match": type_match,
            "numeric_match": numeric_match,
            "exact_result_match": type_match and numeric_match,
            "representation_equivalent_match": numeric_match,
            "numeric_absolute_error": (
                round(absolute_error, 12) if absolute_error is not None else None
            ),
            "numeric_relative_error": (
                round(relative_error, 12) if relative_error is not None else None
            ),
            "requirement_checks": requirement_checks,
            "requirement_pass_rate": round(
                sum(requirement_checks.values()) / len(requirement_checks), 6
            ),
        }

    reference_rows = _records(reference_result, [])
    if reference_rows is None or not reference_rows:
        return {
            "applicable": False,
            "reason": "reference_result is empty or not record-shaped",
        }
    expected_columns = list(reference_rows[0])

    actual_rows = _records(actual_result, expected_columns)
    if actual_rows is None:
        if result_type == "list":
            return {
                "applicable": True,
                "expected_result_type": result_type,
                "result_type_match": False,
                "exact_result_match": False,
                "representation_equivalent_match": False,
                "item_precision": 0.0,
                "item_recall": 0.0,
                "item_f1": 0.0,
                "order_required": _order_required(expected_description),
                "order_correct": False,
                "expected_item_count": len(reference_rows),
                "actual_item_count": None,
            }
        return {
            "applicable": True,
            "expected_result_type": result_type,
            "result_type_match": False,
            "exact_result_match": False,
            "representation_equivalent_match": False,
            "column_precision": 0.0,
            "column_recall": 0.0,
            "column_f1": 0.0,
            "row_precision": 0.0,
            "row_recall": 0.0,
            "row_f1": 0.0,
            "cell_accuracy": 0.0,
            "order_required": _order_required(expected_description),
            "order_correct": False,
            "expected_row_count": len(reference_rows),
            "actual_row_count": None,
        }

    actual_columns = list(dict.fromkeys(
        column for row in actual_rows for column in row
    ))
    columns = _semantic_column_map(
        reference_rows, actual_rows, expected_columns, actual_columns
    )
    key_columns = contract.get("key_columns", [])
    if not isinstance(key_columns, list):
        key_columns = []
    key_columns = [str(column) for column in key_columns]
    tolerances = _contract_tolerances(contract)
    matched_column_count = len(columns)
    column_precision = (
        matched_column_count / len(actual_columns) if actual_columns else 0.0
    )
    column_recall = matched_column_count / len(expected_columns)
    column_f1 = (
        2 * column_precision * column_recall / (column_precision + column_recall)
        if column_precision + column_recall
        else 0.0
    )

    unmatched_actual = set(range(len(actual_rows)))
    matched_pairs: list[tuple[int, int]] = []
    if columns:
        for expected_index, expected_row in enumerate(reference_rows):
            candidates = sorted(unmatched_actual)
            mapped_keys = {
                column: columns[column]
                for column in key_columns if column in columns
            }
            if mapped_keys:
                candidates = [
                    index for index in candidates
                    if _rows_equal(expected_row, actual_rows[index], mapped_keys, tolerances)
                ]
            for actual_index in candidates:
                if _rows_equal(
                    expected_row, actual_rows[actual_index], columns, tolerances
                ):
                    matched_pairs.append((expected_index, actual_index))
                    unmatched_actual.remove(actual_index)
                    break

    matched_rows = len(matched_pairs)
    row_precision = matched_rows / len(actual_rows) if actual_rows else 0.0
    row_recall = matched_rows / len(reference_rows)
    row_f1 = (
        2 * row_precision * row_recall / (row_precision + row_recall)
        if row_precision + row_recall
        else 0.0
    )

    if result_type == "list":
        order_required = bool(contract.get("ordering")) or _order_required(expected_description)
        order_correct = (
            len(reference_rows) == len(actual_rows)
            and bool(columns)
            and all(
                _rows_equal(expected, actual, columns, tolerances)
                for expected, actual in zip(reference_rows, actual_rows)
            )
        )
        type_match = isinstance(actual_result, list)
        strict_schema_match = len(expected_columns) == len(actual_columns) == len(columns)
        representation_equivalent = (
            len(expected_columns) == len(columns)
            and len(reference_rows) == len(actual_rows) == matched_rows
            and (order_correct or not order_required)
        )
        exact_match = (
            type_match
            and strict_schema_match
            and representation_equivalent
        )
        checks = _requirement_checks(
            type_match=type_match, expected_columns=expected_columns,
            mapped_columns=columns, expected_count=len(reference_rows),
            actual_count=len(actual_rows), order_required=order_required,
            order_correct=order_correct, contract=contract,
        )
        return {
            "applicable": True,
            "expected_result_type": result_type,
            "result_type_match": type_match,
            "exact_result_match": exact_match,
            "representation_equivalent_match": representation_equivalent,
            "item_precision": round(row_precision, 6),
            "item_recall": round(row_recall, 6),
            "item_f1": round(row_f1, 6),
            "order_required": order_required,
            "order_correct": order_correct,
            "expected_item_count": len(reference_rows),
            "actual_item_count": len(actual_rows),
            "requirement_checks": checks,
            "requirement_pass_rate": round(sum(checks.values()) / len(checks), 6),
        }

    comparable_cells = len(reference_rows) * len(expected_columns)
    matching_cells = matched_rows * len(columns)
    matched_expected = {expected for expected, _actual in matched_pairs}
    matched_actual = {actual for _expected, actual in matched_pairs}
    remaining_actual = set(range(len(actual_rows))) - matched_actual
    for expected_index, expected_row in enumerate(reference_rows):
        if expected_index in matched_expected or not remaining_actual:
            continue
        best_actual = None
        best_score = -1
        for actual_index in remaining_actual:
            score = sum(
                _values_equal(
                    expected_row.get(expected_column),
                    actual_rows[actual_index].get(actual_column),
                    rel_tol=tolerances.get(expected_column, (_FLOAT_REL_TOL, _FLOAT_ABS_TOL))[0],
                    abs_tol=tolerances.get(expected_column, (_FLOAT_REL_TOL, _FLOAT_ABS_TOL))[1],
                )
                for expected_column, actual_column in columns.items()
            )
            if score > best_score:
                best_actual = actual_index
                best_score = score
        if best_actual is not None:
            matching_cells += best_score
            remaining_actual.remove(best_actual)
    cell_accuracy = matching_cells / comparable_cells if comparable_cells else 0.0

    order_required = bool(contract.get("ordering")) or _order_required(expected_description)
    order_correct = (
        len(reference_rows) == len(actual_rows)
        and bool(columns)
        and all(
            _rows_equal(expected, actual, columns, tolerances)
            for expected, actual in zip(reference_rows, actual_rows)
        )
    )
    all_expected_columns_match = len(expected_columns) == matched_column_count
    strict_schema_match = len(expected_columns) == len(actual_columns)
    all_rows_match = (
        len(reference_rows) == len(actual_rows) == matched_rows
    )
    representation_equivalent = all_expected_columns_match and all_rows_match and (
        order_correct or not order_required
    )

    type_match = isinstance(actual_result, (list, Mapping))
    exact_match = type_match and strict_schema_match and representation_equivalent

    checks = _requirement_checks(
        type_match=type_match, expected_columns=expected_columns,
        mapped_columns=columns, expected_count=len(reference_rows),
        actual_count=len(actual_rows), order_required=order_required,
        order_correct=order_correct, contract=contract,
    )
    return {
        "applicable": True,
        "expected_result_type": result_type,
        "result_type_match": type_match,
        "exact_result_match": exact_match,
        "representation_equivalent_match": representation_equivalent,
        "column_precision": round(column_precision, 6),
        "column_recall": round(column_recall, 6),
        "column_f1": round(column_f1, 6),
        "row_precision": round(row_precision, 6),
        "row_recall": round(row_recall, 6),
        "row_f1": round(row_f1, 6),
        "cell_accuracy": round(cell_accuracy, 6),
        "order_required": order_required,
        "order_correct": order_correct,
        "expected_row_count": len(reference_rows),
        "actual_row_count": len(actual_rows),
        "column_aliases": {
            expected: actual
            for expected, actual in columns.items()
            if _normalized_column(expected) != _normalized_column(actual)
        },
        "ignored_actual_columns": [
            column for column in actual_columns if column not in set(columns.values())
        ],
        "key_columns": key_columns,
        "requirement_checks": checks,
        "requirement_pass_rate": round(sum(checks.values()) / len(checks), 6),
    }


def unavailable_code_evaluation(reason: str) -> dict[str, Any]:
    return {"applicable": False, "reason": reason, "attempts": []}


def summarize_code_evaluations(
    results: Sequence[Mapping[str, Any]],
    *,
    coder_context_level: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-query code evaluation records for a completed API batch."""

    evaluations = []
    for entry in results:
        query_result = entry.get("result", {})
        if coder_context_level is None:
            evaluation = query_result.get("code_evaluation", {})
        else:
            evaluation = (
                query_result.get("coder_context_experiment", {})
                .get("variants", {})
                .get(coder_context_level, {})
                .get("code_evaluation", {})
            )
        evaluations.append(evaluation)
    applicable = [item for item in evaluations if item.get("applicable")]
    total = len(evaluations)
    count = len(applicable)

    def rate(key: str) -> float:
        return round(
            sum(bool(item.get(key)) for item in applicable) / count, 6
        ) if count else 0.0

    def mean(key: str) -> float | None:
        values = [
            float(item[key]) for item in applicable if item.get(key) is not None
        ]
        return round(sum(values) / len(values), 6) if values else None

    def robust_stats(key: str) -> dict[str, float | None]:
        values = sorted(
            float(item[key]) for item in applicable
            if item.get(key) is not None and math.isfinite(float(item[key]))
        )
        if not values:
            return {"median": None, "p90": None, "p95": None}
        def percentile(fraction: float) -> float:
            index = fraction * (len(values) - 1)
            low, high = math.floor(index), math.ceil(index)
            if low == high:
                return values[low]
            return values[low] + (values[high] - values[low]) * (index - low)
        return {
            "median": round(statistics.median(values), 6),
            "p90": round(percentile(0.90), 6),
            "p95": round(percentile(0.95), 6),
        }

    def exact_rate_for_type(result_type: str) -> float | None:
        typed = [
            item for item in applicable
            if item.get("expected_result_type") == result_type
        ]
        if not typed:
            return None
        return round(
            sum(bool(item.get("exact_result_match")) for item in typed) / len(typed),
            6,
        )

    errors = Counter(
        str(item.get("error_category"))
        for item in applicable
        if item.get("error_category")
    )
    type_counts = Counter(
        str(item.get("expected_result_type"))
        for item in applicable
        if item.get("expected_result_type")
    )
    def disposition(item: Mapping[str, Any]) -> str:
        explicit = item.get("evaluation_disposition")
        if explicit:
            return str(explicit)
        if not int(item.get("attempt_count") or 0):
            return "not_evaluated"
        return "correct" if item.get("exact_result_match") else "incorrect"

    dispositions = Counter(disposition(item) for item in applicable)
    supported_count = sum(
        bool(item.get("supported_correct", item.get("exact_result_match")))
        for item in applicable
    )
    return {
        "batch_case_count": total,
        "applicable_case_count": count,
        "non_applicable_case_count": total - count,
        "generation_success_rate": rate("generation_success"),
        "execution_success_rate": rate("execution_success"),
        "any_attempt_execution_success_rate": rate(
            "any_attempt_execution_success"
        ),
        "final_execution_success_rate": rate("final_execution_success"),
        "structured_output_rate": rate("structured_output_valid"),
        "result_type_match_rate": rate("result_type_match"),
        "exact_result_match_rate": rate("exact_result_match"),
        "representation_equivalent_match_rate": rate(
            "representation_equivalent_match"
        ),
        "supported_result_rate": round(supported_count / count, 6) if count else 0.0,
        "ambiguous_result_rate": round(
            dispositions.get("completed_with_warnings", 0) / count, 6
        ) if count else 0.0,
        "pass_at_1": rate("pass_at_1"),
        "success_within_3": rate("success_within_3"),
        "mean_attempts": mean("attempt_count"),
        "mean_generation_attempts": mean("generation_attempt_count"),
        "mean_execution_attempts": mean("execution_attempt_count"),
        "mean_column_f1": mean("column_f1"),
        "mean_row_f1": mean("row_f1"),
        "mean_cell_accuracy": mean("cell_accuracy"),
        "mean_item_f1": mean("item_f1"),
        "mean_numeric_absolute_error": mean("numeric_absolute_error"),
        "mean_numeric_relative_error": mean("numeric_relative_error"),
        "numeric_absolute_error_robust": robust_stats("numeric_absolute_error"),
        "numeric_relative_error_robust": robust_stats("numeric_relative_error"),
        "mean_requirement_pass_rate": mean("requirement_pass_rate"),
        "case_count_by_result_type": dict(sorted(type_counts.items())),
        "exact_result_match_rate_by_type": {
            result_type: exact_rate_for_type(result_type)
            for result_type in ("number", "table", "list")
        },
        "error_categories": dict(sorted(errors.items())),
        "evaluation_dispositions": dict(sorted(dispositions.items())),
        "semantic_judge_case_count": sum(
            bool(item.get("semantic_judge_used")) for item in applicable
        ),
        "semantic_judge_total_tokens": sum(
            int(item.get("semantic_judge_tokens") or 0) for item in applicable
        ),
        "semantic_judge_parse_success_rate": (
            round(
                sum(bool(item.get("semantic_judge_parse_success")) for item in applicable
                    if item.get("semantic_judge_used"))
                / sum(bool(item.get("semantic_judge_used")) for item in applicable),
                6,
            )
            if any(item.get("semantic_judge_used") for item in applicable)
            else None
        ),
    }
