"""What the coder's run knows: its lifecycle, the attempts, the best result so far, the contract and its validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class CoderLifecycle(str, Enum):
    NEEDS_CODE = "needs_code"
    NEEDS_INSPECTION = "needs_inspection"
    NEEDS_REVISION = "needs_revision"
    READY_TO_FINISH = "ready_to_finish"
    TABLES_INSUFFICIENT = "tables_insufficient"
    FINISHED = "finished"


@dataclass
class P3State:
    max_runs: int = 2
    run_count: int = 0
    result_version: int = 0
    inspected_version: int = 0
    code_raw: str = ""
    clean_code: str = ""
    raw_result: str | None = None
    structured_result: object | None = None
    structured_result_error: str = ""
    result_adaptations: list[str] = field(default_factory=list)
    result_adaptation_notes: list[str] = field(default_factory=list)
    preflight_revision_used: bool = False
    column_repair_credit_used: bool = False
    revision_decisions: list[dict[str, object]] = field(default_factory=list)
    error: str | None = None
    execution_error: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    review: dict[str, Any] = field(default_factory=dict)
    coverage_requirements: list[str] = field(default_factory=list)
    coverage_warnings: list[str] = field(default_factory=list)
    inspected_tables: set[str] = field(default_factory=set)
    table_samples: dict[str, Any] = field(default_factory=dict)
    table_profiled_columns: dict[str, set[str]] = field(default_factory=dict)
    rejected_reason: str = ""
    rejection_details: dict[str, Any] = field(default_factory=dict)
    rejection_keep_tables: list[str] = field(default_factory=list)
    rejection_skip_tables: list[str] = field(default_factory=list)
    lifecycle: CoderLifecycle = CoderLifecycle.NEEDS_CODE
    stop_reason: str = ""
    finalization_mode: str = ""
    analysis_contract: dict[str, Any] = field(default_factory=dict)
    contract_code_warnings: list[str] = field(default_factory=list)
    contract_advisories: list[str] = field(default_factory=list)
    contract_evidence: list[dict[str, Any]] = field(default_factory=list)
    contract_evidence_advisories: list[str] = field(default_factory=list)
    missing_column_failures: dict[str, int] = field(default_factory=dict)
    architect_contract_locked: bool = False
    operation_trace: dict[str, Any] = field(default_factory=dict)
    plan_validation: dict[str, Any] = field(default_factory=dict)
    analysis_manifest: dict[str, Any] = field(default_factory=dict)
    analysis_manifest_validation: dict[str, Any] = field(default_factory=dict)
    execution_attempts: list[dict[str, Any]] = field(default_factory=list)
    best_result_snapshot: dict[str, Any] = field(default_factory=dict)
    data_inspection_used: bool = False

    def has_unusable_result(self) -> bool:
        return any(warning.startswith((
            "diagnostic_output:", "contract_result_all_non_finite:",
            "contract_result_non_finite_numeric_value", "empty_result:",
            "final_contract_correlation_requires_one_numeric_value",
        )) for warning in self.coverage_warnings)

    def ready_for_finalization(self) -> bool:
        """Return whether a closure-only turn is safe and meaningful."""
        return (
            not self.finished
            and self.error is None
            and self.structured_result is not None
            and self.result_version > 0
            and self.inspected_version == self.result_version
            and not self.coverage_warnings
            and self.lifecycle == CoderLifecycle.READY_TO_FINISH
        )

    def ready_for_degraded_finalization(self, *, allow_unusable: bool = False) -> bool:
        """Return whether a computed result can be preserved with advisories."""
        return (
            not self.finished
            and (allow_unusable or not self.has_unusable_result())
            and self.error is None
            and self.structured_result is not None
            and self.result_version > 0
            and self.inspected_version == self.result_version
            and self.lifecycle in {
                CoderLifecycle.NEEDS_REVISION,
                CoderLifecycle.READY_TO_FINISH,
            }
        )


ReviewStatus = Literal["verified", "not_applicable", "needs_revision"]
