"""The runs table reduces to one row per cell, and to one row per question of a cell."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_runner.analysis.runs import load_runs
from experiment_runner.analysis.summary import cell_questions, cell_summaries

ROOT = Path(__file__).resolve().parents[3]
JOBS = ROOT / "tests" / "golden" / "fixtures" / "jobs" / "sample"


def test_cell_summaries_count_progress_and_rates():
    frame = load_runs(job_dirs=[JOBS])
    frame["cell_id"] = frame.job_id.str[:8]
    summaries = cell_summaries(frame)
    assert [s["experiment_id"] for s in summaries] == [
        "bench-llm-target_bird-hybrid",
        "bench-llm-chicago-pneuma",
    ]
    target = summaries[0]
    assert target["n"] == 4 and target["completed"] + target["failed"] == 4
    assert (
        target["lake"] == "target_bird"
        and target["mode"] == "hybrid"
        and target["selector"] == "llm"
    )
    assert target["effective_models"] == ["openai.gpt-oss-120b", "together.gpt-oss-120b"]
    assert (
        0.0 <= target["solved"] <= 1.0
        and 0.0 <= target["in_context"] <= 1.0
        and 0.0 <= target["recall10"] <= 1.0
    )
    assert target["tokens_total"] == int(frame[frame.cell_id == target["cell_id"]].tokens.sum())
    assert cell_summaries(frame.iloc[0:0]) == []


def test_rates_follow_the_gold_tables():
    frame = pd.DataFrame(
        [
            {
                "cell_id": "c",
                "experiment": "e",
                "experiment_id": "x",
                "job_id": "j",
                "job_status": "completed",
                "lake": "uk",
                "mode": "keyword",
                "selector": "llm",
                "architecture": "divided",
                "model": "m",
                "effective_model": "m",
                "status": "completed",
                "solved": True,
                "tables": ("a",),
                "gold": ("a", "b"),
                "ranking": ("b", "c", "a"),
                "elapsed_seconds": 10.0,
                "tokens": 100,
                "index": 0,
                "question_id": "1",
                "question": "q",
                "error": "",
                "timestamp": None,
                "run_id": None,
            },
            {
                "cell_id": "c",
                "experiment": "e",
                "experiment_id": "x",
                "job_id": "j",
                "job_status": "completed",
                "lake": "uk",
                "mode": "keyword",
                "selector": "llm",
                "architecture": "divided",
                "model": "m",
                "effective_model": "m",
                "status": "failed",
                "solved": False,
                "tables": ("a",),
                "gold": ("a",),
                "ranking": ("a",),
                "elapsed_seconds": 30.0,
                "tokens": 300,
                "index": 1,
                "question_id": "2",
                "question": "q2",
                "error": "boom",
                "timestamp": None,
                "run_id": None,
            },
        ]
    )
    [summary] = cell_summaries(frame)
    assert summary["completed"] == 1 and summary["failed"] == 1 and summary["solved"] == 0.5
    assert (
        summary["exact"] == 0.5
        and summary["hit1"] == 1.0
        and summary["recall10"] == 1.0
        and summary["in_context"] == 1.0
    )
    assert summary["elapsed_mean"] == 20.0 and summary["tokens_mean"] == 200.0

    questions = cell_questions(frame)
    assert [q["question_id"] for q in questions] == ["1", "2"]
    assert (
        questions[0]["gold_ranks"] == {"a": 3, "b": 1}
        and not questions[0]["exact"]
        and questions[0]["solved"]
    )
    assert questions[1]["error"] == "boom" and questions[1]["exact"]
