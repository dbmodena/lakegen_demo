"""From what the phases return to the outcomes the engine compares.

The phases keep their return types; these adapters read them once, at the
boundary, into ``lakegen.domain.outcomes``. The status rules are the
engine's own, moved here: a coder attempt that the selection-contract gate
stopped is blocked, one that rejected its tables is a rejection, one that
produced a result without an error succeeded, anything else failed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lakegen.coder_experiment import serialize_retry_error
from lakegen.domain.outcomes import CodeOutcome, CodeStatus

CONTRACT_GATE_STAGE = "selection_contract_gate"
CONTRACT_NOT_VERIFIED = "The selection contract was not verified."


def _gated(execution_error: Mapping[str, Any] | None, require_semantic_plan: bool) -> bool:
    return require_semantic_plan and (execution_error or {}).get("stage") == CONTRACT_GATE_STAGE


def code_outcome(generated: Any, *, require_semantic_plan: bool) -> CodeOutcome:
    """The outcome of one ``phase3_generate_and_execute`` call (a ``Phase3Result``)."""
    execution_error = getattr(generated, "execution_error", None)
    common = dict(
        generated=generated,
        code=generated.clean_code or generated.code_raw,
        tokens=generated.tokens,
        raw_result=generated.raw_result,
        rejection_details=dict(getattr(generated, "rejection_details", None) or {}),
        keep_tables=list(getattr(generated, "rejection_keep_tables", None) or []),
        skip_tables=list(getattr(generated, "rejection_skip_tables", None) or []),
        audit=getattr(generated, "coder_context_audit", None),
        execution_error=execution_error,
        retryable=bool((execution_error or {}).get("retryable")),
        coder_runs=int(getattr(generated, "coder_runs", 0) or 0),
        coder_review=getattr(generated, "coder_review", None),
    )
    if _gated(execution_error, require_semantic_plan):
        return CodeOutcome(CodeStatus.BLOCKED_BY_SELECTION, error=generated.error or CONTRACT_NOT_VERIFIED, **common)
    if generated.rejected_reason:
        return CodeOutcome(CodeStatus.TABLES_REJECTED, error=generated.rejected_reason, **common)
    if generated.error is None and generated.raw_result is not None:
        return CodeOutcome(CodeStatus.SUCCESS, error="", **common)
    return CodeOutcome(CodeStatus.EXECUTION_ERROR, error=serialize_retry_error(generated), **common)


def sweep_outcome(variant: Mapping[str, Any], *, require_semantic_plan: bool) -> CodeOutcome:
    """The outcome of one coder-context variant of ``run_coder_context_sweep``.

    The sweep ran its own attempts and settled on a status word; the
    variant's error is its final one.
    """
    execution_error = variant.get("execution_error")
    raw_status = str(variant.get("status") or "")
    if _gated(execution_error, require_semantic_plan):
        status = CodeStatus.BLOCKED_BY_SELECTION
    elif raw_status == "tables_rejected":
        status = CodeStatus.TABLES_REJECTED
    elif raw_status == "completed":
        status = CodeStatus.SUCCESS
    else:
        status = CodeStatus.EXECUTION_ERROR
    return CodeOutcome(
        status,
        generated=variant,
        code=str(variant.get("code") or ""),
        tokens=int(variant.get("tokens") or 0),
        raw_result=variant.get("raw_result"),
        error=str(variant.get("error") or ""),
        raw_status=raw_status,
        rejection_details=dict(variant.get("rejection_details") or {}),
        audit=variant.get("coder_context_audit"),
        execution_error=execution_error,
        coder_runs=int(variant.get("coder_runs") or 0),
        coder_review=variant.get("coder_review"),
    )
