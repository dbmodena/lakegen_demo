"""The phases' returns become typed outcomes once, at the boundary."""

from __future__ import annotations

from types import SimpleNamespace

from lakegen.domain.evaluation import EvaluationContext
from lakegen.domain.outcomes import CodeStatus
from lakegen.pipeline.adapters import CONTRACT_NOT_VERIFIED, code_outcome, sweep_outcome


def _generated(**overrides) -> SimpleNamespace:
    base = dict(
        tokens=6, clean_code="print(42)", code_raw="```python\nprint(42)\n```", rejected_reason="",
        rejection_details=None, error=None, raw_result="42", coder_runs=1, execution_error=None,
        coder_review=None, coder_context_audit=None, rejection_keep_tables=[], rejection_skip_tables=[],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_result_without_an_error_succeeded():
    outcome = code_outcome(_generated(), require_semantic_plan=True)
    assert outcome.status is CodeStatus.SUCCESS and outcome.error == "" and outcome.raw_result == "42"
    assert outcome.code == "print(42)" and outcome.tokens == 6 and outcome.coder_runs == 1


def test_the_contract_gate_blocks_only_when_a_plan_is_required():
    generated = _generated(error="gate", raw_result=None, execution_error={"stage": "selection_contract_gate"})
    blocked = code_outcome(generated, require_semantic_plan=True)
    assert blocked.status is CodeStatus.BLOCKED_BY_SELECTION and blocked.error == "gate"
    assert code_outcome(_generated(error=None, raw_result=None, execution_error={"stage": "selection_contract_gate"}),
                        require_semantic_plan=True).error == CONTRACT_NOT_VERIFIED
    assert code_outcome(generated, require_semantic_plan=False).status is CodeStatus.EXECUTION_ERROR


def test_a_rejection_carries_its_details_and_the_tables_to_keep_or_skip():
    generated = _generated(rejected_reason="no join key", rejection_details={"category": "x"},
                           rejection_keep_tables=["a.csv"], rejection_skip_tables=["b.csv"], raw_result=None)
    outcome = code_outcome(generated, require_semantic_plan=True)
    assert outcome.status is CodeStatus.TABLES_REJECTED and outcome.error == "no join key"
    assert outcome.rejection_details == {"category": "x"}
    assert outcome.keep_tables == ["a.csv"] and outcome.skip_tables == ["b.csv"]


def test_an_execution_error_serialises_its_structure_and_says_whether_to_retry():
    generated = _generated(error="KeyError: x", raw_result=None,
                           execution_error={"stage": "execute", "category": "missing_column", "retryable": True})
    outcome = code_outcome(generated, require_semantic_plan=True)
    assert outcome.status is CodeStatus.EXECUTION_ERROR and outcome.retryable
    assert outcome.error.startswith("KeyError: x") and "missing_column" in outcome.error
    plain = code_outcome(_generated(error="boom", raw_result=None, coder_runs=0), require_semantic_plan=True)
    assert plain.error == "boom" and not plain.retryable and plain.coder_runs == 0
    assert code_outcome(_generated(clean_code="", raw_result=None, error="e"), require_semantic_plan=True).code.startswith("```")


def test_sweep_variants_map_their_status_words():
    base = {"code": "c", "raw_result": "", "error": "", "tokens": 3, "coder_runs": 2, "execution_error": None,
            "coder_context_audit": {"status": "verified"}, "rejection_details": None, "coder_review": None}
    completed = sweep_outcome({**base, "status": "completed"}, require_semantic_plan=True)
    assert completed.status is CodeStatus.SUCCESS and completed.raw_status == "completed" and completed.audit == {"status": "verified"}
    rejected = sweep_outcome({**base, "status": "tables_rejected", "error": "bad", "rejection_details": {"k": 1}}, require_semantic_plan=True)
    assert rejected.status is CodeStatus.TABLES_REJECTED and rejected.error == "bad" and rejected.rejection_details == {"k": 1}
    gated = sweep_outcome({**base, "status": "failed", "execution_error": {"stage": "selection_contract_gate"}}, require_semantic_plan=True)
    assert gated.status is CodeStatus.BLOCKED_BY_SELECTION
    other = sweep_outcome({**base, "status": "rejected", "error": "empty"}, require_semantic_plan=True)
    assert other.status is CodeStatus.EXECUTION_ERROR and other.raw_status == "rejected"


def test_evaluation_context_parses_the_source_fields_and_keeps_every_field():
    fields = {
        "SOURCE_ID": "q1", "SOURCE_EXPECTED_RESULT_TYPE": "number", "SOURCE_REFERENCE_RESULT": [{"n": 1}],
        "SOURCE_EXPECTED_RESULT_DESCRIPTION": "a count", "SOURCE_EVALUATION_CONTRACT": {"tolerance": 0.1},
        "SOURCE_RELEVANT_TABLE_IDS": ["a", "b"], "JOB_ID": "j",
    }
    context = EvaluationContext.from_log_context(fields)
    assert context.source_id == "q1" and context.expected_result_type == "number"
    assert context.reference_result == [{"n": 1}] and context.expected_description == "a count"
    assert context.evaluation_contract == {"tolerance": 0.1} and context.relevant_tables == ["a", "b"]
    assert context.field_names == list(fields) and context.to_log_fields() == fields
    assert context.to_log_fields() is not context.fields

    empty = EvaluationContext.from_log_context(None)
    assert empty.fields == {} and empty.expected_result_type == "" and empty.relevant_tables == []
    odd = EvaluationContext.from_log_context({"SOURCE_EVALUATION_CONTRACT": "x", "SOURCE_RELEVANT_TABLE_IDS": "a"})
    assert odd.evaluation_contract == {} and odd.relevant_tables == []
