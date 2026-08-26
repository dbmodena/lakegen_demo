"""Execution of the shared-retrieval coder-context experiment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from lakegen.code_attempts import CodeAttemptEvaluator
from lakegen.code_evaluation import unavailable_code_evaluation
from lakegen.experiment_config import CoderContextLevel
from lakegen.output_validation import AnswerDisposition, validate_answer


def run_coder_context_sweep(
    *,
    question: str,
    selected: list[str],
    solr_meta: dict[str, Any],
    reasoning: str,
    llm: Any,
    prompt_manager: Any,
    csv_dir: Path,
    run_dir: Path,
    seed: int,
    record_seed_instruction: Callable[[], None],
    expected_result_type: str,
    evaluation_enabled: bool,
    evaluator: CodeAttemptEvaluator,
    adjudicate: Callable[[dict[str, Any], Any], dict[str, Any]],
    generate_and_execute: Callable[..., Any],
    result: Any,
    phase_invocation_counts: dict[str, int],
    max_attempts: int,
    context_levels: list[CoderContextLevel] | None = None,
) -> dict[str, Any]:
    """Test every metadata level while reusing the same selected tables."""
    variants: dict[str, Any] = {}
    levels = context_levels or list(CoderContextLevel)
    for context_level in levels:
        started = time.monotonic()
        error = ""
        previous_code = ""
        generated = None
        attempts: list[dict[str, Any]] = []
        tokens = 0
        status = "failed"
        attempt_index = 0

        for attempt_index in range(max_attempts):
            code_started = time.monotonic()
            generated = generate_and_execute(
                question, selected, selected, solr_meta, reasoning, llm,
                prompt_manager, csv_dir, retries=attempt_index,
                error_msg=error, previous_code=previous_code,
                run_dir=run_dir / "coder_context" / context_level.value,
                seed=seed, seed_instruction_recorder=record_seed_instruction,
                coder_context_level=context_level,
                evaluation_result_type=(expected_result_type if evaluation_enabled else None),
            )
            phase_invocation_counts["code"] += 1
            tokens += generated.tokens
            result.tokens["p3"] += generated.tokens
            metric = result.phase_metrics.setdefault(
                "code", {"latency_seconds": 0.0, "retries": 0}
            )
            metric["latency_seconds"] = round(
                metric["latency_seconds"] + (time.monotonic() - code_started), 6
            )
            result.retries += int(attempt_index > 0)
            previous_code = generated.clean_code or generated.code_raw
            if evaluation_enabled:
                attempts.append(evaluator.evaluate(generated, attempt_index + 1))
            if generated.rejected_reason:
                status, error = "tables_rejected", generated.rejected_reason
                break
            if generated.error is None and generated.raw_result is not None:
                validation = validate_answer(generated.raw_result)
                if validation.disposition == AnswerDisposition.EMPTY:
                    status, error = "empty", validation.reason
                    continue
                if validation.disposition == AnswerDisposition.REJECTED:
                    status, error = "rejected", validation.reason
                    break
                status, error = "completed", ""
                break
            error = generated.error or "Code execution returned no output."
            if getattr(generated, "coder_runs", 0):
                break

        if evaluation_enabled:
            evaluation = evaluator.summarize(attempts)
            if generated is not None and status == "completed":
                evaluation = adjudicate(evaluation, generated)
        else:
            evaluation = unavailable_code_evaluation(
                "expected_result_type and reference_result are required"
            )
        variants[context_level.value] = {
            "coder_context_level": context_level.value,
            "status": status,
            "code": previous_code,
            "raw_result": (
                generated.raw_result
                if generated is not None and generated.raw_result is not None else ""
            ),
            "error": error,
            "execution_error": getattr(generated, "execution_error", None),
            "coder_review": getattr(generated, "coder_review", None),
            "coder_runs": getattr(generated, "coder_runs", 0),
            "tokens": tokens,
            "attempts": len(attempts) if evaluation_enabled else attempt_index + 1,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "code_evaluation": evaluation,
        }
    result.phase_metrics["code"]["retries"] = result.retries
    return variants
