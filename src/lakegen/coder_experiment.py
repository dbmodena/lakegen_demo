"""Execution of the shared-retrieval coder-context experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from lakegen.code_attempts import CodeAttemptEvaluator
from lakegen.code_evaluation import unavailable_code_evaluation
from lakegen.experiment_config import CoderContextLevel
from lakegen.output_validation import AnswerDisposition, validate_answer


def serialize_retry_error(generated: Any) -> str:
    """Build bounded, actionable retry context from a generated-code failure."""

    message = str(getattr(generated, "error", "") or "Code execution returned no output.")
    structured = getattr(generated, "execution_error", None)
    if not isinstance(structured, dict) or not structured:
        return message
    allowed = {
        "stage", "category", "column", "retryable", "closest_columns",
        "source_columns", "rename_hints", "repair_hint", "next_actions",
        "coverage_warnings",
    }
    payload = {key: structured[key] for key in allowed if key in structured}
    if isinstance(payload.get("source_columns"), list):
        payload["source_columns"] = payload["source_columns"][:50]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return (
        f"{message[-1600:]}\n\n"
        "STRUCTURED_REPAIR_CONTEXT (authoritative JSON):\n"
        f"{encoded}\n"
        "Apply repair_hint and rename_hints exactly when present. Do not repeat "
        "the failing downstream column label."
    )


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
        total_coder_runs = 0
        repair_attempts = 0
        # A model turn that never executes code must not consume the scarce
        # execution/repair budget. Keep both budgets bounded independently.
        max_generation_turns = max_attempts * 2

        for attempt_index in range(max_generation_turns):
            remaining_runs = max_attempts - total_coder_runs
            if remaining_runs <= 0:
                break
            code_started = time.monotonic()
            generated = generate_and_execute(
                question, selected, selected, solr_meta, reasoning, llm,
                prompt_manager, csv_dir, retries=attempt_index,
                error_msg=error, previous_code=previous_code,
                run_dir=run_dir / "coder_context" / context_level.value,
                seed=seed, seed_instruction_recorder=record_seed_instruction,
                coder_context_level=context_level,
                evaluation_result_type=(expected_result_type if evaluation_enabled else None),
                max_run_calls=remaining_runs,
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
            total_coder_runs += int(getattr(generated, "coder_runs", 0) or 0)
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
            error = serialize_retry_error(generated)
            execution_error = getattr(generated, "execution_error", None) or {}
            retryable = bool(execution_error.get("retryable"))
            if total_coder_runs >= max_attempts:
                break
            if getattr(generated, "coder_runs", 0) and not retryable:
                break
            if getattr(generated, "coder_runs", 0) and retryable:
                repair_attempts += 1

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
            "coder_runs": total_coder_runs,
            "generation_turns": len(attempts) if evaluation_enabled else attempt_index + 1,
            "repair_attempts": repair_attempts,
            "coder_lifecycle": getattr(generated, "coder_lifecycle", ""),
            "stop_reason": getattr(generated, "stop_reason", ""),
            "finalization_mode": getattr(generated, "finalization_mode", ""),
            "tokens": tokens,
            "attempts": len(attempts) if evaluation_enabled else attempt_index + 1,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "code_evaluation": evaluation,
        }
    result.phase_metrics["code"]["retries"] = result.retries
    return variants
