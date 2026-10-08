"""The phases' returns become typed outcomes once, at the boundary."""

from __future__ import annotations

from types import SimpleNamespace

from lakegen.domain.evaluation import EvaluationContext
from lakegen.domain.outcomes import CodeStatus, DiscoveryStatus, SelectorKind
from lakegen.phases.jev_selector import TRACE_HEADER as JEV_TRACE_HEADER
from lakegen.pipeline.adapters import (
    CONTRACT_NOT_VERIFIED,
    NO_PREPARED_DATASETS,
    code_outcome,
    divided_agentic_outcome,
    divided_orchestrated_outcome,
    sweep_outcome,
    unified_agentic_outcome,
    unified_orchestrated_outcome,
)


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


# --- discovery -------------------------------------------------------------


def _state(**overrides) -> SimpleNamespace:
    base = dict(all_candidates=["a.csv", "b.csv"], rejection_keep_tables=[], rejection_skip_tables=[], inspection_cache={})
    base.update(overrides)
    return SimpleNamespace(**base)


def test_the_unified_agent_round_is_selected_or_rejected_by_its_reasoning():
    raw = (["a.csv"], ["parks"], {"a.csv": {"title": "a"}}, "The table covers it.", "trace", 5)
    outcome = unified_agentic_outcome(raw, _state())
    assert outcome.status is DiscoveryStatus.SELECTED and not outcome.rejected
    assert outcome.selected == ["a.csv"] and outcome.candidates == ["a.csv", "b.csv"] and outcome.tokens == 5
    assert outcome.phase_tokens == (5, 0) and outcome.selector is SelectorKind.AGENT

    state = _state(rejection_keep_tables=["a.csv"], rejection_skip_tables=["b.csv"], inspection_cache={"a.csv": "cols"})
    rejected = unified_agentic_outcome(([], ["parks"], {}, "REJECT_KEYWORDS: nothing fits", "t", 2), state)
    assert rejected.rejected and rejected.keep_tables == ["a.csv"] and rejected.skip_tables == ["b.csv"]
    assert rejected.inspection_cache == {"a.csv": "cols"} and rejected.reasoning.startswith("REJECT_KEYWORDS:")


def test_the_divided_rounds_compose_their_trace_and_name_the_selector():
    keyword_round = (["parks"], "raw", 2, "phase 1 reasoning")
    selection = (["a.csv"], ["a.csv", "b.csv"], {}, "Selected.", "phase 2 trace", 3)
    outcome = divided_agentic_outcome(keyword_round, selection, _state())
    assert outcome.status is DiscoveryStatus.SELECTED and outcome.tokens == 5 and outcome.phase_tokens == (2, 3)
    assert outcome.trace == "--- Divided Phase 1 ---\nphase 1 reasoning\n--- Divided Phase 2 ---\nphase 2 trace"
    assert outcome.selector is SelectorKind.LLM and outcome.candidates == ["a.csv", "b.csv"]
    jev = divided_agentic_outcome(keyword_round, (["a.csv"], ["a.csv"], {}, "ok", JEV_TRACE_HEADER + " x", 0), _state())
    assert jev.selector is SelectorKind.JEV


def test_the_unified_orchestrated_result_rejects_when_it_asks_for_keywords():
    result = SimpleNamespace(selected_datasets=["a.csv"], candidates=["a.csv"], keywords=["k"], metadata={},
                             reasoning="Selected.", trace="", tokens=4, retry_keywords=False, prepared_context="ctx")
    outcome = unified_orchestrated_outcome(result)
    assert outcome.status is DiscoveryStatus.SELECTED and outcome.prepared == "ctx" and outcome.phase_tokens == (4, 0)
    assert unified_orchestrated_outcome(SimpleNamespace(**{**vars(result), "retry_keywords": True})).rejected


def test_the_divided_orchestrated_round_without_candidates_or_with_an_empty_selection_rejects():
    keyword_round = (["k"], "raw", 2, "r")
    empty = SimpleNamespace(candidates=[])
    outcome = divided_orchestrated_outcome(keyword_round, empty, {}, None)
    assert outcome.rejected and outcome.reasoning == NO_PREPARED_DATASETS and outcome.selector is SelectorKind.NONE
    assert outcome.tokens == 2 and outcome.phase_tokens == (2, 0) and outcome.candidates == []

    prepared = SimpleNamespace(candidates=[SimpleNamespace(dataset="a.csv"), SimpleNamespace(dataset="b.csv")])
    chosen = divided_orchestrated_outcome(keyword_round, prepared, {"a.csv": {}}, (["a.csv"], "ok", "trace", 3))
    assert chosen.status is DiscoveryStatus.SELECTED and chosen.selected == ["a.csv"] and chosen.tokens == 5
    assert chosen.candidates == ["a.csv", "b.csv"] and chosen.selector is SelectorKind.LLM and chosen.prepared is prepared
    nothing = divided_orchestrated_outcome(keyword_round, prepared, {}, ([], "I chose nothing", "trace", 3))
    assert nothing.rejected and nothing.selected == [] and nothing.reasoning.startswith("REJECT_KEYWORDS:")
    asked = divided_orchestrated_outcome(keyword_round, prepared, {}, (["a.csv"], "REJECT_KEYWORDS: try again", "t", 1))
    assert asked.rejected and asked.selected == [] and asked.reasoning == "REJECT_KEYWORDS: try again"
