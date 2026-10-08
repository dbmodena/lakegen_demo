"""Batch metrics pair each question with its own result, not with the result in its position."""

from __future__ import annotations

from lakegen.analysis.metrics import batch_cases


def _question(source_id: str, gold: list[str]) -> dict:
    return {"question": f"q{source_id}", "source_id": source_id, "source_path": f"$.q[{source_id}]", "log_fields": {"SOURCE_RELEVANT_TABLE_IDS": gold}}


def _result(source_id: str, tables: list[str], **extra) -> dict:
    return {
        "source_id": source_id, "source_path": f"$.q[{source_id}]", "question": f"q{source_id}",
        "result": {"tables": tables, "ranking": [{"resource_id": t, "rank": i + 1, "score": 1.0} for i, t in enumerate(tables)], **extra},
    }


def test_results_are_paired_by_source_and_a_requeued_question_keeps_its_last():
    questions = [_question("1", ["a"]), _question("2", ["b"])]
    results = [_result("2", ["b"]), _result("1", ["x"]), _result("1", ["a"])]  # out of order; "1" re-queued
    cases = batch_cases(questions, results)
    assert [case.selection["SelectionHit"] for case in cases] == [1.0, 1.0]


def test_a_question_without_a_result_or_without_gold_tables_is_not_scored():
    assert batch_cases([_question("1", ["a"]), _question("2", ["b"])], [_result("1", ["a"])]) is None
    assert batch_cases([_question("1", [])], [_result("1", ["a"])]) is None
    assert batch_cases([], []) is None


def test_entries_without_a_source_are_paired_by_position():
    questions = [{"question": "q", "log_fields": {"SOURCE_RELEVANT_TABLE_IDS": ["a"]}}, {"question": "r", "log_fields": {"SOURCE_RELEVANT_TABLE_IDS": ["b"]}}]
    results = [{"result": {"tables": ["a"], "ranking": []}}, {"result": {"tables": ["b"], "ranking": []}}]
    assert [case.selection["SelectionHit"] for case in batch_cases(questions, results)] == [1.0, 1.0]
    assert batch_cases(questions, results[:1]) is None
