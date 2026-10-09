"""Stored jobs become one table of runs, with the models each result actually ran on."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from lakegen.experiments.analysis.jobs import iter_jobs, read_job
from lakegen.experiments.analysis.runs import load_runs, runs_of_job
from lakegen.experiments.analysis.switches import ProviderSwitch, model_id, parse_switches, values_at
from lakegen.experiments.store import RunStore

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "tests" / "fixtures" / "jobs" / "sample"
TARGET = JOBS / "f438db7bda7046df8de2df5eda8c146c.json"
CHICAGO = JOBS / "3687ab4f842a4532a0e665fd61c08c5a.json"


def test_a_job_is_read_with_its_questions_results_and_notes():
    job = read_job(TARGET)
    assert job.job_id == TARGET.stem and job.status == "completed" and job.job_dir == JOBS
    assert job.experiment_id == "bench-llm-target_bird-hybrid"
    assert len(job.questions) == len(job.results) == 4
    assert job.config.get("core") == "target_bird" and job.config.get("retrieval.mode") == "hybrid"
    assert [switch.fields for switch in parse_switches(job.provider_switches)] == [
        ("model", "semantic_code_judge_model"), ("retrieval.pneuma_entity_model",),
    ]
    assert job.question(job.results[2])["source_id"] == job.results[2]["source_id"]
    assert [job.job_id[:8] for job in iter_jobs(JOBS)] == ["f438db7b", "3687ab4f"]


def test_switches_are_parsed_from_every_recorded_shape():
    switches = parse_switches([
        {"at": "2026-10-02T06:04:27+00:00", "what": "requeued", "removed_results": 3, "reason": "402"},
        {"at": "2026-10-01T22:59:33+00:00", "what": "retrieval.pneuma_entity_model", "from": "a (OCI)", "to": "b"},
        {"at": "2026-10-01T22:52:43+00:00", "from": "a (OCI)", "to": "b", "results_before_switch": 83, "reason": "key"},
        {"at": "2026-10-01T23:04:27+00:00", "fields": ["model"], "from": "a (OCI)", "to": "b", "results_before_switch": 0, "reason": "x"},
    ])
    assert [switch.fields for switch in switches] == [
        ("model", "semantic_code_judge_model"), ("retrieval.pneuma_entity_model",), ("model",),
    ]
    assert switches[0].from_value == "a" and switches[0].to_value == "b"
    assert model_id("openai.gpt-oss-120b (OCI)") == "openai.gpt-oss-120b" and model_id(None) is None


def test_values_before_two_switches_of_one_field_are_the_earliest():
    stored = {"model": "c", "semantic_code_judge_model": "j", "retrieval.pneuma_entity_model": "e"}
    first = ProviderSwitch(datetime(2026, 10, 1, 10, tzinfo=UTC), ("model",), "a", "b")
    second = ProviderSwitch(datetime(2026, 10, 1, 12, tzinfo=UTC), ("model",), "b", "c")
    at = lambda hour: f"2026-10-01T{hour:02}:00:00Z"  # noqa: E731
    assert values_at(stored, [second, first], at(9))["model"] == "a"
    assert values_at(stored, [second, first], at(11))["model"] == "b"
    assert values_at(stored, [second, first], at(13))["model"] == "c"
    assert values_at(stored, [second, first], None)["model"] == "c"
    assert values_at(stored, [], at(9)) == stored


def test_hand_switches_are_undone_by_the_result_timestamp():
    runs = runs_of_job(read_job(TARGET))
    assert [run.model for run in runs] == ["together.gpt-oss-120b"] * 4
    # Results 0, 1 and 82 of the job ran before the switches; result 83 after both.
    assert [run.effective_model for run in runs] == ["openai.gpt-oss-120b"] * 3 + ["together.gpt-oss-120b"]
    assert [run.judge_model for run in runs] == [run.effective_model for run in runs]
    assert [run.entity_model for run in runs] == [run.effective_model for run in runs]


def test_split_results_carry_their_own_model_and_the_judge_follows():
    runs = runs_of_job(read_job(CHICAGO))
    assert [run.effective_model for run in runs] == ["openai.gpt-oss-120b"] * 2 + ["together.gpt-oss-120b"] * 2
    assert [run.judge_model for run in runs] == [run.effective_model for run in runs]
    assert [run.entity_model for run in runs] == [run.effective_model for run in runs]


def test_load_runs_reads_job_directories():
    frame = load_runs(job_dirs=[JOBS])
    assert len(frame) == 8 and frame.job_id.nunique() == 2
    assert set(frame.lake) == {"target_bird", "chicago"} and set(frame["mode"]) == {"hybrid", "pneuma"}
    assert set(frame.selector) == {"llm"} and set(frame.architecture) == {"divided"}
    first = frame.iloc[0]
    assert first.question_id == "bird_281" and first.gold == ("toxicology_atom", "toxicology_molecule")
    assert first.ranking[:2] == first.gold and first.tables == ("toxicology_molecule",)
    assert first.status == "failed" and not first.solved
    assert first.tokens == first.tokens_p1_p2 + first.tokens_p3 + first.tokens_p4 > 0
    assert first.elapsed_seconds > 0 and first.run_id and first.timestamp
    assert frame.solved.dtype == bool and frame.code_applicable.all()
    assert load_runs(job_dirs=[]).empty and list(load_runs(job_dirs=[]).columns) == list(frame.columns)


def test_load_runs_reads_an_experiment_through_the_store(tmp_path):
    store = RunStore(tmp_path / "runs")
    index_path = tmp_path / "runs" / "bench" / "index.json"
    index_path.parent.mkdir(parents=True)
    index_path.write_text(json.dumps({
        "cell-1": {
            "cell_id": "cell-1", "experiment_id": "bench-llm-target_bird-hybrid", "axes": {},
            "pass_name": None, "questions_file": "q.json", "questions_slice": None,
            "backend": "api", "job_dir": str(JOBS), "job_id": TARGET.stem, "submitted_at": None,
        },
        "cell-2": {
            "cell_id": "cell-2", "experiment_id": "never-submitted", "axes": {},
            "pass_name": None, "questions_file": "q.json", "questions_slice": None,
            "backend": "api", "job_dir": None, "job_id": None, "submitted_at": None,
        },
    }), encoding="utf-8")
    frame = load_runs("bench", store=store)
    assert len(frame) == 4 and set(frame.experiment) == {"bench"} and set(frame.cell_id) == {"cell-1"}
