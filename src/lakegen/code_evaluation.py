"""Deterministic functional evaluation for generated data-analysis code."""

from __future__ import annotations

import json
import math
import re
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
        if stripped.endswith("%"):
            stripped = stripped[:-1].strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _values_equal(left: Any, right: Any) -> bool:
    if _is_null(left) or _is_null(right):
        return _is_null(left) and _is_null(right)
    left_number = _numeric(left)
    right_number = _numeric(right)
    if left_number is not None and right_number is not None:
        return math.isclose(
            left_number,
            right_number,
            rel_tol=_FLOAT_REL_TOL,
            abs_tol=_FLOAT_ABS_TOL,
        )
    return str(left).strip() == str(right).strip()


def _records(value: Any, expected_columns: Sequence[str]) -> list[dict[str, Any]] | None:
    if isinstance(value, Mapping):
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
    """Return a scalar candidate and whether it was already scalar-shaped."""

    if isinstance(value, Mapping):
        if len(value) == 1:
            return next(iter(value.values())), False
        return value, False
    if isinstance(value, list):
        if len(value) == 1:
            scalar, _scalar_shaped = _single_value(value[0])
            return scalar, False
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
    if not remaining_expected or len(remaining_expected) != len(remaining_actual):
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
) -> bool:
    return all(
        _values_equal(expected.get(expected_column), actual.get(actual_column))
        for expected_column, actual_column in columns.items()
    )


def evaluate_code_result(
    *,
    expected_result_type: str,
    reference_result: Any,
    actual_result: Any,
    expected_description: str = "",
) -> dict[str, Any]:
    """Compare a structured generated result with the benchmark reference."""

    result_type = str(expected_result_type).casefold()
    if result_type not in {"number", "table", "list"}:
        return {
            "applicable": False,
            "reason": f"unsupported expected_result_type {expected_result_type!r}",
        }

    reference_rows = _records(reference_result, [])
    if reference_rows is None or not reference_rows:
        return {
            "applicable": False,
            "reason": "reference_result is empty or not record-shaped",
        }
    expected_columns = list(reference_rows[0])

    if result_type == "number":
        if len(reference_rows) != 1 or len(expected_columns) != 1:
            return {
                "applicable": False,
                "reason": "number reference_result must contain exactly one value",
            }
        expected_value = reference_rows[0][expected_columns[0]]
        actual_value, scalar_shaped = _single_value(actual_result)
        expected_number = _numeric(expected_value)
        actual_number = _numeric(actual_value)
        numeric_match = _values_equal(expected_value, actual_value)
        absolute_error = (
            abs(actual_number - expected_number)
            if expected_number is not None and actual_number is not None
            else None
        )
        relative_error = (
            absolute_error / abs(expected_number)
            if absolute_error is not None and expected_number not in {None, 0.0}
            else absolute_error
        )
        type_match = scalar_shaped and actual_number is not None
        return {
            "applicable": True,
            "expected_result_type": result_type,
            "result_type_match": type_match,
            "numeric_match": numeric_match,
            "exact_result_match": type_match and numeric_match,
            "numeric_absolute_error": (
                round(absolute_error, 12) if absolute_error is not None else None
            ),
            "numeric_relative_error": (
                round(relative_error, 12) if relative_error is not None else None
            ),
        }

    actual_rows = _records(actual_result, expected_columns)
    if actual_rows is None:
        if result_type == "list":
            return {
                "applicable": True,
                "expected_result_type": result_type,
                "result_type_match": False,
                "exact_result_match": False,
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
            for actual_index in sorted(unmatched_actual):
                if _rows_equal(expected_row, actual_rows[actual_index], columns):
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
        order_required = _order_required(expected_description)
        order_correct = (
            len(reference_rows) == len(actual_rows)
            and bool(columns)
            and all(
                _rows_equal(expected, actual, columns)
                for expected, actual in zip(reference_rows, actual_rows)
            )
        )
        type_match = isinstance(actual_result, list)
        exact_match = (
            type_match
            and len(expected_columns) == len(actual_columns) == len(columns)
            and len(reference_rows) == len(actual_rows) == matched_rows
            and (order_correct or not order_required)
        )
        return {
            "applicable": True,
            "expected_result_type": result_type,
            "result_type_match": type_match,
            "exact_result_match": exact_match,
            "item_precision": round(row_precision, 6),
            "item_recall": round(row_recall, 6),
            "item_f1": round(row_f1, 6),
            "order_required": order_required,
            "order_correct": order_correct,
            "expected_item_count": len(reference_rows),
            "actual_item_count": len(actual_rows),
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

    order_required = _order_required(expected_description)
    order_correct = (
        len(reference_rows) == len(actual_rows)
        and bool(columns)
        and all(
            _rows_equal(expected, actual, columns)
            for expected, actual in zip(reference_rows, actual_rows)
        )
    )
    all_columns_match = (
        len(expected_columns) == len(actual_columns) == matched_column_count
    )
    all_rows_match = (
        len(reference_rows) == len(actual_rows) == matched_rows
    )
    exact_match = all_columns_match and all_rows_match and (
        order_correct or not order_required
    )

    type_match = isinstance(actual_result, (list, Mapping))

    return {
        "applicable": True,
        "expected_result_type": result_type,
        "result_type_match": type_match,
        "exact_result_match": exact_match,
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
        return "gold_correct" if item.get("exact_result_match") else "incorrect"

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
        "structured_output_rate": rate("structured_output_valid"),
        "result_type_match_rate": rate("result_type_match"),
        "exact_result_match_rate": rate("exact_result_match"),
        "supported_result_rate": round(supported_count / count, 6) if count else 0.0,
        "ambiguous_result_rate": round(
            dispositions.get("indeterminate", 0) / count, 6
        ) if count else 0.0,
        "pass_at_1": rate("pass_at_1"),
        "success_within_3": rate("success_within_3"),
        "mean_attempts": mean("attempt_count"),
        "mean_column_f1": mean("column_f1"),
        "mean_row_f1": mean("row_f1"),
        "mean_cell_accuracy": mean("cell_accuracy"),
        "mean_item_f1": mean("item_f1"),
        "mean_numeric_absolute_error": mean("numeric_absolute_error"),
        "mean_numeric_relative_error": mean("numeric_relative_error"),
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
    }
