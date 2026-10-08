"""The starter questions and the reference benchmark the chat offers come from files the repository holds."""

from __future__ import annotations

import json

from lakegen.config.registry import lakes
from lakegen.core import config as core_config
from lakegen.ui import benchmark_sidebar, starters


def test_every_lake_names_question_files_that_exist():
    for lake in lakes():
        assert lake.starters_path is not None and lake.starters_path.is_file(), lake.name
        if lake.reference_questions:
            assert lake.reference_questions_path is not None and lake.reference_questions_path.is_file(), lake.name


def test_a_lake_without_default_starters_takes_them_from_its_benchmark():
    questions = starters.starter_questions_for_core("chicago")
    assert len(questions) == starters.STARTER_LIMIT
    cases = json.loads((core_config.BASE_DIR / "benchmark" / "100q_chicago.json").read_text(encoding="utf-8"))["cases"]
    asked = {" ".join(case["question"].split()) for case in cases}
    assert all(question in asked for _label, question in questions)
    assert all(0 < len(label) <= starters.LABEL_LIMIT for label, _question in questions)


def test_the_uk_sidebar_reads_the_curated_benchmark(monkeypatch):
    monkeypatch.setattr(benchmark_sidebar, "UK_REFERENCE_FILE", core_config.BASE_DIR / "benchmark" / "100q_uk.json")
    benchmark_sidebar.load_uk_reference_questions.cache_clear()
    entries = benchmark_sidebar.load_uk_reference_questions()
    assert len(entries) == 100
    first = entries[0]
    assert first["id"] == "e247c40d" and first["question"].startswith("For each survey date in the 1984 Loch Sween survey")
    assert first["golden_tables"] == [{
        "alias": "Table_0",
        "table_id": "648fb958-f935-4a66-9dc2-1cb7712c865a___3b4e3d7c-9db6-4215-a64b-c909e8ce03d4",
        "description": "",
    }]
    assert first["query_kind"] == "table" and isinstance(first["expected_result"], list)
    assert first["reference_response"].startswith("One row per survey date")


def test_the_sidebar_still_reads_the_generated_queries_shape(tmp_path, monkeypatch):
    payload = {"engine": {"semantic": {"group-1": {
        "_meta": {"tables": {"Table_0": "ds___res"}},
        "q1": {"status": "success", "question": "How many?", "tables": [{"name": "Table_0", "description": "A table"}], "query_result": 3},
        "q2": {"status": "failed", "question": "Ignored", "tables": [{"name": "Table_0"}]},
    }}}}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(benchmark_sidebar, "UK_REFERENCE_FILE", path)
    benchmark_sidebar.load_uk_reference_questions.cache_clear()
    entries = benchmark_sidebar.load_uk_reference_questions()
    assert [entry["question"] for entry in entries] == ["How many?"]
    assert entries[0]["golden_tables"][0]["table_id"] == "ds___res" and entries[0]["expected_result"] == 3
