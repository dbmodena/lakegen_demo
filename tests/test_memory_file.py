"""The durable keyword memory file: what one write keeps, and what two writers keep."""

from __future__ import annotations

from lakegen.keyword_memory import (
    load_keyword_memory,
    load_question_retrieval_memory,
    persist_keyword_memory,
    persist_question_retrieval_memory,
)
from lakegen.keyword_terms import keyword_terms

EVENT = {"outcome": "zero_results", "terms": ["parks"]}


def test_persisting_keywords_keeps_the_questions_retrieval_memories(tmp_path):
    path = tmp_path / "memory.json"
    persist_question_retrieval_memory(path, "uk", "q1", [EVENT])
    persist_question_retrieval_memory(path, "uk", "q2", [{**EVENT, "terms": ["rivers"]}])

    persist_keyword_memory(path, "uk", [["parks"]], [["parks", "2024"]])

    assert load_question_retrieval_memory(path, "uk", "q1") == [EVENT]
    assert load_question_retrieval_memory(path, "uk", "q2") == [{**EVENT, "terms": ["rivers"]}]
    history, failed = load_keyword_memory(path, "uk")
    assert history == [["parks"]] and failed == [keyword_terms(["parks", "2024"])]


def test_persisting_a_question_memory_keeps_the_keywords(tmp_path):
    path = tmp_path / "memory.json"
    persist_keyword_memory(path, "uk", [["parks"]], [["parks", "2024"]])
    persist_question_retrieval_memory(path, "uk", "q1", [EVENT])
    history, failed = load_keyword_memory(path, "uk")
    assert history == [["parks"]] and failed == [keyword_terms(["parks", "2024"])]
