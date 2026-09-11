from types import SimpleNamespace

from lakegen.code_attempts import CodeAttemptEvaluator


def test_non_string_generated_code_is_a_generation_failure_not_a_crash():
    evaluator = CodeAttemptEvaluator("number", 42)
    generated = SimpleNamespace(
        code_raw=["print(42)"], rejected_reason="", error="invalid code payload",
        raw_result=None, structured_result=None, structured_result_error="",
    )

    result = evaluator.evaluate(generated, 1)

    assert result["generation_success"] is False
    assert result["execution_success"] is False


def test_runtime_attempt_trace_drives_pass_metrics_and_dispositions():
    evaluator = CodeAttemptEvaluator("number", 42)
    generated = SimpleNamespace(coder_attempt_trace=[
        {
            "generation_success": True, "execution_success": False,
            "structured_result": None, "error": "temporary failure",
        },
        {
            "generation_success": True, "execution_success": True,
            "structured_result": 42, "error": "",
        },
    ])

    attempts = evaluator.evaluate_generated_attempts(generated, 1)
    summary = evaluator.summarize(attempts)

    assert summary["attempt_count"] == 2
    assert summary["pass_at_1"] is False
    assert summary["success_within_3"] is True
    assert summary["evaluation_disposition"] == "correct"


def test_execution_failure_is_not_classified_as_incorrect():
    evaluator = CodeAttemptEvaluator("number", 42)
    summary = evaluator.summarize([{
        "attempt": 1, "generation_success": True,
        "execution_success": False, "structured_output_valid": False,
        "exact_result_match": False, "error": "boom",
    }])

    assert summary["error_category"] == "execution_error"
    assert summary["evaluation_disposition"] == "not_evaluated"
