"""Finishing the coder's run: the validated result, the degraded one, the best one remembered."""

from __future__ import annotations

import copy
import json

from lakegen.coding.tools.state import CoderLifecycle, ReviewStatus


class FinishMixin:
    """Part of ``Phase3ToolsManager``; the manager supplies the attributes it reads."""

    def finish_code(
        self,
        filters: ReviewStatus,
        measures: ReviewStatus,
        grouping: ReviewStatus,
        ordering: ReviewStatus,
        output_shape: ReviewStatus,
        requirement_reviews: dict[str, str],
        review: str,
    ) -> str:
        """Finish only after inspecting the latest result and completing every review dimension."""
        checks = {
            "filters": filters,
            "measures": measures,
            "grouping": grouping,
            "ordering": ordering,
            "output_shape": output_shape,
        }
        if self.state.error is not None or self.state.raw_result is None:
            raise ValueError("Finish blocked: there is no successful execution result.")
        if self.state.inspected_version != self.state.result_version:
            raise ValueError("Finish blocked: inspect_result must inspect the latest successful run.")
        if self.state.coverage_warnings:
            raise ValueError(
                "Finish blocked: correct the code and call run_code again. "
                + "; ".join(self.state.coverage_warnings)
            )
        unresolved = [name for name, status in checks.items() if status == "needs_revision"]
        if unresolved:
            self.state.lifecycle = CoderLifecycle.NEEDS_REVISION
            raise ValueError("Finish blocked: revise code for " + ", ".join(unresolved) + ".")
        if len(review.strip()) < 20:
            raise ValueError("Finish blocked: provide a concrete review of at least 20 characters.")
        reviewed = {
            str(requirement).strip(): str(evidence).strip()
            for requirement, evidence in requirement_reviews.items()
            if str(requirement).strip()
        }
        missing_requirements = [
            requirement for requirement in self.state.coverage_requirements
            if requirement not in reviewed
        ]
        if missing_requirements:
            raise ValueError(
                "Finish blocked: review every inspect_result requirement: "
                + "; ".join(missing_requirements)
            )
        if any(len(evidence) < 8 for evidence in reviewed.values()):
            raise ValueError("Finish blocked: provide evidence for every requirement.")
        self.state.finished = True
        self.state.lifecycle = CoderLifecycle.FINISHED
        self.state.finalization_mode = "agent"
        self.state.review = {
            **checks,
            "requirement_reviews": reviewed,
            "review": review.strip(),
        }
        return "FINAL_PAYLOAD: " + json.dumps({"status": "finished", "review": self.state.review})

    def recover_finish(self, reason: str) -> None:
        """Close a valid inspected result when only the agent protocol failed."""
        if not self.state.ready_for_finalization():
            raise ValueError("Recovery finalization requires the latest warning-free inspected result.")
        self.state.finished = True
        self.state.lifecycle = CoderLifecycle.FINISHED
        self.state.finalization_mode = "system_recovery"
        self.state.review = {
            "finalization_mode": "system_recovery",
            "coverage_checks": "verified",
            "semantic_self_review": "not_available",
            "requirement_reviews": {
                requirement: "Verified by inspect_result coverage checks."
                for requirement in self.state.coverage_requirements
            },
            "review": reason,
        }

    def finalize_validated_result(self, reason: str) -> None:
        """Finalize deterministically after the latest result passes all gates."""
        self.recover_finish(reason)
        self.state.finalization_mode = "deterministic_validation"
        self.state.review.update({
            "finalization_mode": "deterministic_validation",
            "semantic_self_review": "deterministic_contract_checks",
        })

    def recover_degraded_finish(
        self, reason: str, *, allow_unusable: bool = False
    ) -> None:
        """Preserve an inspected structured result without declaring it correct."""
        if not self.state.ready_for_degraded_finalization(
            allow_unusable=allow_unusable
        ):
            raise ValueError("Degraded recovery requires the latest inspected structured result.")
        self.state.finished = True
        self.state.lifecycle = CoderLifecycle.FINISHED
        self.state.finalization_mode = "system_recovery_with_advisories"
        self.state.review = {
            "finalization_mode": "system_recovery_with_advisories",
            "coverage_checks": "advisories_preserved",
            "semantic_self_review": "incomplete",
            "coverage_warnings": list(self.state.coverage_warnings),
            "contract_advisories": list(dict.fromkeys([
                *self.state.contract_advisories,
                *self.state.contract_code_warnings,
            ])),
            "review": reason,
        }

    def _remember_best_result(self) -> None:
        if self.state.structured_result is None or self.state.has_unusable_result():
            return
        snapshot = {
            "score": (
                -len(self.state.coverage_warnings),
                -len(self.state.contract_advisories),
            ),
            "raw_result": self.state.raw_result,
            "structured_result": copy.deepcopy(self.state.structured_result),
            "structured_result_error": self.state.structured_result_error,
            "code_raw": self.state.code_raw,
            "clean_code": self.state.clean_code,
            "coverage_requirements": list(self.state.coverage_requirements),
            "coverage_warnings": list(self.state.coverage_warnings),
            "contract_advisories": list(self.state.contract_advisories),
            "operation_trace": copy.deepcopy(self.state.operation_trace),
            "result_adaptations": list(self.state.result_adaptations),
        }
        if not self.state.best_result_snapshot or snapshot["score"] > self.state.best_result_snapshot["score"]:
            self.state.best_result_snapshot = snapshot

    def _restore_best_result(self) -> None:
        snapshot = self.state.best_result_snapshot
        if not snapshot:
            return
        for key in (
            "raw_result", "structured_result", "structured_result_error",
            "code_raw", "clean_code", "coverage_requirements", "coverage_warnings",
            "contract_advisories", "operation_trace", "result_adaptations",
        ):
            setattr(self.state, key, copy.deepcopy(snapshot[key]))
        self.state.error = None
        self.state.execution_error = {}
        self.state.result_version += 1
        self.state.inspected_version = self.state.result_version
        self.state.lifecycle = CoderLifecycle.NEEDS_REVISION
