"""The durable keyword memory file: what one write keeps, and what two writers keep."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from lakegen.tools.shared.terms import keyword_terms
from lakegen.workflow.discovery.memory import (
    load_keyword_memory,
    load_question_retrieval_memory,
    persist_keyword_memory,
    persist_question_retrieval_memory,
)

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


def _write_many(args: tuple[str, str]) -> str:
    path, label = args
    for index in range(25):
        persist_keyword_memory(Path(path), "uk", [[f"{label}{index}"]], [[f"{label}{index}"]])
        persist_question_retrieval_memory(Path(path), "uk", f"{label}{index}", [EVENT])
    return label


def test_two_processes_writing_the_same_memory_lose_nothing(tmp_path):
    path = tmp_path / "memory.json"
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as pool:
        assert sorted(pool.map(_write_many, [(str(path), "alpha"), (str(path), "bravo")])) == ["alpha", "bravo"]
    history, failed = load_keyword_memory(path, "uk")
    assert len(history) == 50 and len(failed) == 50
    assert all(load_question_retrieval_memory(path, "uk", f"{label}{index}") == [EVENT] for label in ("alpha", "bravo") for index in range(25))
    assert (tmp_path / "memory.json.lock").exists()


def test_each_experiment_remembers_on_its_own(monkeypatch, tmp_path):
    from lakegen.config import paths

    monkeypatch.setenv("LAKEGEN_STATE_DIR", str(tmp_path))
    assert paths.keyword_memory_path("qwen-grid") == tmp_path / "runs" / "qwen-grid" / "memory.json"
    assert paths.keyword_memory_path("default") == paths.keyword_memory_path(None) == tmp_path / ".lakegen_keyword_memory.json"
