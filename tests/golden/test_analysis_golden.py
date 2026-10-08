"""Goldens of the analysis over stored jobs: the batch report the API logs.

The report of a job must keep coming out the same from its stored questions
and results, whatever moves between the API and ``lakegen.analysis``; the
API's own wrapper must return the report's summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lakegen.analysis.metrics import batch_cases, batch_report, report_summary

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "tests" / "fixtures" / "jobs" / "sample"
JOB_IDS = sorted(path.stem for path in JOBS.glob("*.json") if not path.name.endswith(".questions.json"))


def _job(job_id: str) -> tuple[dict, list[dict], list[dict]]:
    job = json.loads((JOBS / f"{job_id}.json").read_text(encoding="utf-8"))
    questions = json.loads((JOBS / f"{job_id}.questions.json").read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in (JOBS / f"{job_id}.results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return job, questions, results


@pytest.mark.parametrize("job_id", JOB_IDS, ids=[job_id[:8] for job_id in JOB_IDS])
def test_batch_report_of_a_stored_job(job_id: str, golden):
    job, questions, results = _job(job_id)
    cases = batch_cases(questions, results)
    assert cases is not None
    report = batch_report(cases, results, job["settings"]["resolved_config"], created_at="<created>")
    golden(f"analysis/batch_report__{job['settings']['resolved_config']['experiment_id']}", report)


def test_the_batch_wrapper_returns_the_report_summary(monkeypatch):
    from lakegen.experiments import batch

    job, questions, results = _job(JOB_IDS[0])
    logged = []
    monkeypatch.setattr(batch, "append_benchmark_metrics_log", lambda report, *a, **k: logged.append(report))
    summary = batch.append_batch_table_metrics(job["job_id"], questions, results, job["settings"])
    assert len(logged) == 1
    assert summary == report_summary(logged[0])
    assert summary["case_count"] == 4 and "cases" not in summary
