from types import SimpleNamespace

from lakegen.coder_experiment import run_coder_context_sweep
from lakegen.experiment_config import CoderContextLevel


def _generated(*, error, coder_runs, retryable=True, raw_result=None):
    return SimpleNamespace(
        tokens=1,
        clean_code="print(42)",
        code_raw="print(42)",
        rejected_reason="",
        error=error,
        raw_result=raw_result,
        structured_result=None,
        structured_result_error="",
        execution_error=(
            {
                "stage": "execution",
                "category": "missing_column",
                "column": "EXPULSIONS",
                "retryable": retryable,
                "rename_hints": [{
                    "renamed_from": "EXPULSIONS", "renamed_to": "Expulsions"
                }],
                "repair_hint": "Use Expulsions downstream.",
            }
            if error else None
        ),
        coder_runs=coder_runs,
        coder_review=None,
        coder_lifecycle="needs_revision" if error else "finished",
        stop_reason="",
        finalization_mode="",
    )


def test_non_execution_turns_do_not_consume_execution_repair_budget(tmp_path):
    generated = iter([
        _generated(error="Coder stopped without terminal state", coder_runs=0),
        _generated(error="Coder stopped without terminal state", coder_runs=0),
        _generated(error="KeyError: 'EXPULSIONS'", coder_runs=1),
        _generated(error=None, coder_runs=1, raw_result="42"),
    ])
    calls = []
    retry_messages = []

    def generate(*_args, **kwargs):
        calls.append(kwargs["max_run_calls"])
        retry_messages.append(kwargs["error_msg"])
        return next(generated)

    result = SimpleNamespace(
        tokens={"p3": 0}, phase_metrics={}, retries=0,
    )
    variants = run_coder_context_sweep(
        question="Question?", selected=["table.parquet"], solr_meta={},
        reasoning="", llm=None, prompt_manager=None, csv_dir=tmp_path,
        run_dir=tmp_path, seed=0, record_seed_instruction=lambda: None,
        expected_result_type="", evaluation_enabled=False, evaluator=None,
        adjudicate=lambda evaluation, _generated: evaluation,
        generate_and_execute=generate, result=result,
        phase_invocation_counts={"code": 0}, max_attempts=3,
        context_levels=[CoderContextLevel.FULL],
    )

    full = variants["full"]
    assert full["status"] == "completed"
    assert full["generation_turns"] == 4
    assert full["coder_runs"] == 2
    assert full["repair_attempts"] == 1
    assert calls == [3, 3, 3, 2]
    assert "STRUCTURED_REPAIR_CONTEXT" in retry_messages[3]
    assert '"renamed_to": "Expulsions"' in retry_messages[3]
    assert "Use Expulsions downstream." in retry_messages[3]
