"""Small deterministic recovery policy for coder failures and result gaps."""

from __future__ import annotations

from typing import Literal


RevisionAction = Literal[
    "CONTINUE", "REPAIR_CODE", "REVISE_RESULT", "RETRY_DISCOVERY"
]


_TECHNICAL_ERRORS = {
    "column_resolution_error",
    "diagnostic_output",
    "dataset_edition_row_filter",
    "forbidden_import",
    "join_error",
    "manifest_invalid",
    "manifest_missing",
    "missing_column",
    "missing_file",
    "runtime_error",
    "structured_output_error",
    "syntax_error",
    "type_error",
}


def classify_revision(
    execution_error: dict[str, object] | None = None,
    coverage_warnings: list[str] | None = None,
    rejection_details: dict[str, object] | None = None,
) -> dict[str, object]:
    """Choose one generic next action from existing deterministic evidence."""
    error = execution_error or {}
    warnings = [str(item) for item in coverage_warnings or [] if str(item).strip()]
    rejection = rejection_details or {}
    category = str(error.get("category") or "").strip()

    if rejection:
        missing = rejection.get("missing_requirements")
        evidence = str(rejection.get("inspected_evidence") or "").strip()
        if isinstance(missing, list) and missing and evidence:
            return {
                "action": "RETRY_DISCOVERY",
                "reason": "fundamental_data_requirement_proven_missing",
                "retryable": True,
            }
    if category == "security_error":
        return {
            "action": "REPAIR_CODE",
            "reason": category,
            "retryable": False,
        }
    if category in {"result_needs_revision", "pre_execution_contract_gap"} or warnings:
        return {
            "action": "REVISE_RESULT",
            "reason": category or "semantic_requirement_gap",
            "retryable": bool(error.get("retryable", True)),
        }
    if category in _TECHNICAL_ERRORS or str(error.get("stage")) == "preflight":
        return {
            "action": "REPAIR_CODE",
            "reason": category or "preflight_error",
            "retryable": bool(error.get("retryable", True)),
        }
    return {"action": "CONTINUE", "reason": "no_blocking_issue", "retryable": False}


def technical_repair_eligible(error: dict[str, object] | None) -> bool:
    """Return whether one bounded technical repair credit may be granted."""
    decision = classify_revision(execution_error=error)
    return bool(
        decision["action"] == "REPAIR_CODE"
        and decision["retryable"]
        and str((error or {}).get("stage") or "") == "execution"
    )
