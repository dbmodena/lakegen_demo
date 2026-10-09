"""The non-API sources are read as stored, and the runs table can carry the raw results."""

from __future__ import annotations

import json
from pathlib import Path

from lakegen.experiments.analysis.runs import COLUMNS, load_runs
from lakegen.experiments.analysis.sources import selector_compare
from lakegen.experiments.analysis.sources.benchmarks import benchmark_cases, gold_is_any_of

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "tests" / "fixtures" / "jobs" / "sample"


def _results_dir(tmp_path: Path) -> Path:
    rows = [
        {"key": "uk|keyword|1", "lake": "uk", "kind": "keyword", "id": "1", "question": "q", "gold": ["a"],
         "context": {"candidates": [{"retrieval_rank": 2, "dataset": "b.parquet"}, {"retrieval_rank": 1, "dataset": "d___a.parquet"}]}},
        {"key": "nyc|keyword|1", "lake": "nyc", "kind": "keyword", "id": "1", "question": "q", "gold": ["z"], "context": {"candidates": []}},
    ]
    (tmp_path / "contexts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "selections.jsonl").write_text(
        json.dumps({"key": "uk|keyword|1|llm", "selected": ["a"]}) + "\n\n"
        + json.dumps({"key": "nyc|keyword|1|llm", "selected": []}) + "\n"
    )
    (tmp_path / "coded").mkdir()
    (tmp_path / "coded" / "uk.jsonl").write_text(
        json.dumps({"key": "uk|1|a", "result": {"status": "completed"}}) + "\n"
        + json.dumps({"key": "uk|1|b", "error": "RateLimitError: slow down", "result": {}}) + "\n"
    )
    return tmp_path


def test_selector_comparison_files_are_read_one_lake_at_a_time(tmp_path):
    results_dir = _results_dir(tmp_path)
    assert list(selector_compare.contexts("uk", results_dir)) == ["uk|keyword|1"]
    assert list(selector_compare.selections("uk", results_dir)) == ["uk|keyword|1|llm"]
    coded = list(selector_compare.coded(results_dir))
    assert [record["key"] for record in coded] == ["uk|1|a", "uk|1|b"]
    assert [selector_compare.provider_failure(record) for record in coded] == [False, True]


def test_benchmark_cases_are_keyed_by_id():
    cases = benchmark_cases("chicago")
    case = next(iter(cases.values()))
    assert set(cases) == {str(case["id"]) for case in cases.values()}
    assert "question" in case and "relevant_table_ids" in case
    assert gold_is_any_of({"pneuma": {"gold_semantics": "any"}}) and not gold_is_any_of({})


def test_load_runs_can_carry_the_stored_result():
    frame = load_runs(job_dirs=[JOBS])
    assert "result" not in frame.columns and list(frame.columns) == list(COLUMNS)
    with_results = load_runs(job_dirs=[JOBS], include_results=True)
    assert list(with_results.columns) == [*COLUMNS, "result"]
    result = with_results.iloc[0].result
    assert isinstance(result, dict) and result["status"] == with_results.iloc[0].status
    assert "code_evaluation" in result and "ranking" in result
