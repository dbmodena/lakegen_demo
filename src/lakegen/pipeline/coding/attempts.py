"""Deterministic bookkeeping for generated-code attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lakegen.pipeline.evaluation.code import evaluate_code_result, unavailable_code_evaluation


def _revision_outcome(first: dict[str, Any], latest: dict[str, Any]) -> str:
    """How the latest attempt compares with the first: regression, improvement, stable or not comparable."""
    first_correct = bool(
        first.get("exact_result_match") or first.get("representation_equivalent_match")
    )
    latest_correct = bool(
        latest.get("exact_result_match") or latest.get("representation_equivalent_match")
    )
    comparable = all(
        item.get("execution_success") and item.get("structured_output_valid")
        for item in (first, latest)
    )
    if first_correct and not latest_correct:
        return "regression"
    if not first_correct and latest_correct:
        return "improvement"
    if first_correct and latest_correct:
        return "stable_correct"
    if comparable:
        return "stable_incorrect"
    return "not_comparable"


@dataclass(frozen=True)
class CodeAttemptEvaluator:
    """Scores the coder's attempts against the benchmark reference and sums them up per question.

    Attributes:
        expected_result_type: The reference's result type (number, table, list).
        reference_result: The benchmark's reference result.
        expected_description: The benchmark's description of the expected result.
        evaluation_contract: The benchmark's evaluation contract, if any.
    """

    expected_result_type: str
    reference_result: object
    expected_description: str = ""
    evaluation_contract: dict[str, Any] | None = None

    @property
    def enabled(self) -> bool:
        """Whether there is a result type and a reference to evaluate against."""
        return bool(self.expected_result_type and self.reference_result is not None)

    def initial_evaluation(self) -> dict[str, Any]:
        """The record before any attempt: every success flag false, or the reason it is unavailable."""
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
            "format_compliance": False,
            "exact_result_match": False,
            "semantic_correctness": "not_evaluated",
            # A pass@1 metric flag, not a password.
            "semantic_pass_at_1": False,  # nosec B105
            "semantic_success_within_3": False,
            # A pass@1 metric flag, not a password.
            "pass_at_1": False,  # nosec B105
            "success_within_3": False,
            "attempt_count": 0,
            "execution_attempt_count": 0,
            "generation_attempt_count": 0,
            "successful_execution_attempt_count": 0,
            "any_attempt_execution_success": False,
            "final_execution_success": False,
            "attempts": [],
            "error_category": None,
            "evaluation_disposition": "not_evaluated",
        }

    def evaluate(self, generated: Any, attempt_number: int) -> dict[str, Any]:
        """Evaluate one coder turn as a single attempt.

        Args:
            generated: The coder turn's result (a ``CoderResult``).
            attempt_number: The attempt's 1-based number.

        Returns:
            The attempt record: generation, execution and structure flags, the
            code and error, and the comparison with the reference when the
            turn produced a structured result.
        """
        code_raw = getattr(generated, "code_raw", "")
        generation_success = (
            isinstance(code_raw, str)
            and not code_raw.startswith("__GENERATION_ERROR__:")
            and not bool(generated.rejected_reason)
        )
        execution_success = generated.error is None and generated.raw_result is not None
        structured_result = getattr(generated, "structured_result", None)
        structured_error = str(getattr(generated, "structured_result_error", "") or "")
        evaluation: dict[str, Any] = {
            "attempt": attempt_number,
            "generation_success": generation_success,
            "execution_success": execution_success,
            "structured_output_valid": structured_result is not None,
            "code": str(code_raw or ""),
            "error": generated.error or structured_error,
        }
        if structured_result is not None:
            evaluation.update(
                evaluate_code_result(
                    expected_result_type=self.expected_result_type,
                    reference_result=self.reference_result,
                    actual_result=structured_result,
                    expected_description=self.expected_description,
                    evaluation_contract=self.evaluation_contract,
                )
            )
        else:
            evaluation["exact_result_match"] = False
        return evaluation

    def evaluate_runtime_attempt(
        self, runtime_attempt: dict[str, Any], attempt_number: int
    ) -> dict[str, Any]:
        """Evaluate one actual run_code call rather than one outer agent turn.

        Args:
            runtime_attempt: The execution attempt from the coder's attempt trace.
            attempt_number: The attempt's 1-based number.

        Returns:
            The attempt record, shaped as ``evaluate`` shapes it.
        """
        structured_result = runtime_attempt.get("structured_result")
        evaluation: dict[str, Any] = {
            "attempt": attempt_number,
            "generation_success": bool(runtime_attempt.get("generation_success", True)),
            "execution_success": bool(runtime_attempt.get("execution_success")),
            "structured_output_valid": structured_result is not None,
            "code": str(runtime_attempt.get("code") or ""),
            "error": runtime_attempt.get("error") or "",
        }
        if structured_result is not None:
            evaluation.update(
                evaluate_code_result(
                    expected_result_type=self.expected_result_type,
                    reference_result=self.reference_result,
                    actual_result=structured_result,
                    expected_description=self.expected_description,
                    evaluation_contract=self.evaluation_contract,
                )
            )
        else:
            evaluation["exact_result_match"] = False
        return evaluation

    def evaluate_generated_attempts(
        self, generated: Any, first_attempt_number: int
    ) -> list[dict[str, Any]]:
        """Evaluate every execution of a coder turn, or the turn itself when it has no attempt trace.

        Args:
            generated: The coder turn's result (a ``CoderResult``).
            first_attempt_number: The number of the turn's first attempt.

        Returns:
            One record per attempt.
        """
        trace = getattr(generated, "coder_attempt_trace", None)
        if isinstance(trace, list) and trace:
            return [
                self.evaluate_runtime_attempt(item, first_attempt_number + index)
                for index, item in enumerate(trace)
                if isinstance(item, dict)
            ]
        return [self.evaluate(generated, first_attempt_number)]

    def summarize(self, attempts: list[dict[str, Any]]) -> dict[str, Any]:
        """Sum up a question's attempts into its code evaluation.

        The latest attempt's fields carry over; pass@1 and success within 3
        come from the attempts in order. A latest result that executed and is
        structured but matches neither exactly nor by representation is left
        pending semantic review.

        Args:
            attempts: The attempt records, in order.

        Returns:
            The question's code evaluation, with its error category, its
            disposition and, over several attempts, how the revision went.
        """
        summary = {
            "applicable": True,
            "expected_result_type": self.expected_result_type,
            "generation_success": any(item["generation_success"] for item in attempts),
            "execution_success": any(item["execution_success"] for item in attempts),
            "structured_output_valid": any(item["structured_output_valid"] for item in attempts),
            "result_type_match": False,
            "format_compliance": False,
            "exact_result_match": False,
            "pass_at_1": bool(attempts and attempts[0].get("exact_result_match")),
            "success_within_3": any(item.get("exact_result_match") for item in attempts[:3]),
            "semantic_pass_at_1": bool(
                attempts
                and (
                    attempts[0].get("exact_result_match")
                    or attempts[0].get("representation_equivalent_match")
                )
            ),
            "semantic_success_within_3": any(
                item.get("exact_result_match") or item.get("representation_equivalent_match")
                for item in attempts[:3]
            ),
            "semantic_correctness": "not_evaluated",
            "attempt_count": len(attempts),
            "execution_attempt_count": len(attempts),
            "successful_execution_attempt_count": sum(
                bool(item.get("execution_success")) for item in attempts
            ),
            "any_attempt_execution_success": any(
                bool(item.get("execution_success")) for item in attempts
            ),
            "attempts": attempts,
            "error_category": None,
            "evaluation_disposition": "not_evaluated",
            "supported_correct": False,
            "semantic_judge_used": False,
            "semantic_judge_tokens": 0,
        }
        if not attempts:
            return summary
        latest = attempts[-1]
        summary.update(
            {
                key: value
                for key, value in latest.items()
                if key not in {"attempt", "error", "applicable"}
            }
        )
        summary["format_compliance"] = bool(latest.get("result_type_match"))
        if len(attempts) > 1:
            summary["revision_outcome"] = _revision_outcome(attempts[0], latest)
        else:
            summary["revision_outcome"] = "single_attempt"
        if latest.get("exact_result_match") or latest.get("representation_equivalent_match"):
            summary.update(
                {
                    "error_category": None,
                    "evaluation_disposition": "correct",
                    "semantic_correctness": "correct",
                    "supported_correct": True,
                }
            )
        elif not latest["generation_success"]:
            summary["error_category"] = "generation_error"
            summary["evaluation_disposition"] = "not_evaluated"
        elif not latest["execution_success"]:
            summary["error_category"] = "execution_error"
            summary["evaluation_disposition"] = "not_evaluated"
        elif not latest["structured_output_valid"]:
            summary["error_category"] = "structured_output_error"
            summary["evaluation_disposition"] = "not_evaluated"
        elif not latest.get("result_type_match"):
            summary["error_category"] = "result_type_mismatch"
            summary["evaluation_disposition"] = "incorrect"
        else:
            summary["error_category"] = "wrong_result"
            summary["evaluation_disposition"] = "incorrect"
        if (
            latest["execution_success"]
            and latest["structured_output_valid"]
            and not latest.get("exact_result_match")
            and not latest.get("representation_equivalent_match")
        ):
            summary["evaluation_disposition"] = "pending_semantic_review"
            summary["semantic_correctness"] = "pending_semantic_review"
        return summary
