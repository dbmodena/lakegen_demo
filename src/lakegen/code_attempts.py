"""Deterministic bookkeeping for generated-code attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lakegen.code_evaluation import evaluate_code_result, unavailable_code_evaluation


@dataclass(frozen=True)
class CodeAttemptEvaluator:
    expected_result_type: str
    reference_result: object
    expected_description: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.expected_result_type and self.reference_result is not None)

    def initial_evaluation(self) -> dict[str, Any]:
        if not self.enabled:
            return unavailable_code_evaluation(
                "expected_result_type and reference_result are required"
            )
        return {
            "applicable": True,
            "expected_result_type": self.expected_result_type,
            "generation_success": False,
            "execution_success": False,
            "structured_output_valid": False,
            "result_type_match": False,
            "exact_result_match": False,
            "pass_at_1": False,
            "success_within_3": False,
            "attempt_count": 0,
            "attempts": [],
            "error_category": None,
        }

    def evaluate(self, generated: Any, attempt_number: int) -> dict[str, Any]:
        generation_success = not (
            generated.code_raw.startswith("__GENERATION_ERROR__:")
            or bool(generated.rejected_reason)
        )
        execution_success = generated.error is None and generated.raw_result is not None
        structured_result = getattr(generated, "structured_result", None)
        structured_error = str(getattr(generated, "structured_result_error", "") or "")
        evaluation: dict[str, Any] = {
            "attempt": attempt_number,
            "generation_success": generation_success,
            "execution_success": execution_success,
            "structured_output_valid": structured_result is not None,
            "error": generated.error or structured_error,
        }
        if structured_result is not None:
            evaluation.update(evaluate_code_result(
                expected_result_type=self.expected_result_type,
                reference_result=self.reference_result,
                actual_result=structured_result,
                expected_description=self.expected_description,
            ))
        else:
            evaluation["exact_result_match"] = False
        return evaluation

    def summarize(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        summary = {
            "applicable": True,
            "expected_result_type": self.expected_result_type,
            "generation_success": any(item["generation_success"] for item in attempts),
            "execution_success": any(item["execution_success"] for item in attempts),
            "structured_output_valid": any(item["structured_output_valid"] for item in attempts),
            "result_type_match": False,
            "exact_result_match": False,
            "pass_at_1": bool(attempts and attempts[0].get("exact_result_match")),
            "success_within_3": any(item.get("exact_result_match") for item in attempts[:3]),
            "attempt_count": len(attempts),
            "attempts": attempts,
            "error_category": None,
            "evaluation_disposition": "incorrect",
            "supported_correct": False,
            "semantic_judge_used": False,
            "semantic_judge_tokens": 0,
        }
        if not attempts:
            return summary
        latest = attempts[-1]
        summary.update({
            key: value for key, value in latest.items()
            if key not in {"attempt", "error", "applicable"}
        })
        if latest.get("exact_result_match") or latest.get("representation_equivalent_match"):
            summary.update({
                "error_category": None,
                "evaluation_disposition": "gold_correct",
                "supported_correct": True,
            })
        elif not latest["generation_success"]:
            summary["error_category"] = "generation_error"
        elif not latest["execution_success"]:
            summary["error_category"] = "execution_error"
        elif not latest["structured_output_valid"]:
            summary["error_category"] = "structured_output_error"
        elif not latest.get("result_type_match"):
            summary["error_category"] = "result_type_mismatch"
        else:
            summary["error_category"] = "wrong_result"
        if (
            latest["execution_success"] and latest["structured_output_valid"]
            and not latest.get("exact_result_match")
            and not latest.get("representation_equivalent_match")
        ):
            summary["evaluation_disposition"] = "pending_semantic_review"
        return summary
