"""Old jobs are indexed under the spec whose cells they match, or under a legacy experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lakegen.experiments.analysis.jobs import iter_jobs
from lakegen.experiments.analysis.runs import load_runs
from lakegen.config.recorded import load_recorded_config
from lakegen.experiments import cli
from lakegen.experiments.legacy import config_differences, index_legacy, matching_cell
from lakegen.experiments.spec import ExperimentSpec, expand
from lakegen.experiments.store import RunStore

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "tests" / "fixtures" / "jobs" / "sample"
BENCH_SPEC = ROOT / "experiments" / "specs" / "bench_grid.yaml"


def _bench_spec_over(job_dir: Path) -> ExperimentSpec:
    spec = ExperimentSpec.load(BENCH_SPEC)
    return spec.model_copy(update={"execution": spec.execution.model_copy(update={"job_dir": str(job_dir)})})


def test_config_differences_skip_unrecorded_and_ignored_fields():
    stored = {"experiment_id": "x", "core": "uk", "retrieval": {"mode": "keyword"}, "benchmark": {"path": "old.json"}}
    resolved = {"experiment_id": "x", "core": "uk", "retrieval": {"mode": "keyword", "top_k": 20}, "benchmark": {"path": "new.json"}, "model": "m"}
    assert config_differences(resolved, stored) == {}
    assert config_differences({**resolved, "core": "nyc"}, stored) == {"core": ("uk", "nyc")}
    assert config_differences({**resolved, "core": "nyc"}, stored, ignored=("core",)) == {}


def test_the_fixture_jobs_match_the_bench_cells_when_the_spec_names_their_directory():
    cells = {cell.experiment_id: cell for cell in expand(_bench_spec_over(JOBS))}
    jobs = list(iter_jobs(JOBS))
    assert [matching_cell(job, cells).experiment_id for job in jobs] == [job.experiment_id for job in jobs]
    assert matching_cell(jobs[0], {}) is None


def test_index_legacy_records_matched_cells_and_the_rest_under_legacy(tmp_path):
    store = RunStore(tmp_path / "runs")
    indexed = index_legacy(store, [JOBS], [_bench_spec_over(JOBS)])
    assert [(entry.experiment, entry.matched) for entry in indexed] == [("bench-grid", True)] * 2
    index = store.load_index("bench-grid")
    assert {record.experiment_id for record in index.values()} == {
        "bench-llm-target_bird-hybrid", "bench-llm-chicago-pneuma",
    }
    record = next(iter(index.values()))
    assert record.job_dir == str(JOBS) and len(record.job_id) == 32 and record.submitted_at.startswith("2026-10-0")
    assert (store.experiment_dir("bench-grid") / "spec.yaml").is_file()
    assert (store.experiment_dir("bench-grid") / "cells" / record.cell_id / "config.yaml").is_file()
    assert store.submitted("bench-grid") == set(index)

    frame = load_runs("bench-grid", store=store)
    assert len(frame) == 8 and set(frame.cell_id) == set(index)

    # Without a spec naming the directory, the jobs go under legacy-<directory>.
    store = RunStore(tmp_path / "runs2")
    indexed = index_legacy(store, [JOBS], [ExperimentSpec.load(BENCH_SPEC)])
    assert [(entry.experiment, entry.matched) for entry in indexed] == [("legacy-sample", False)] * 2
    legacy = store.load_index("legacy-sample")
    assert set(legacy) == {entry.cell_id for entry in indexed} and all(len(cell_id) == 16 for cell_id in legacy)
    assert len(load_runs("legacy-sample", store=store)) == 8
    cell_dir = store.experiment_dir("legacy-sample") / "cells" / next(iter(legacy))
    assert (cell_dir / "config.yaml").is_file() and (cell_dir / "cell.json").is_file()


def test_index_legacy_is_idempotent_and_keeps_the_latest_job_of_a_cell(tmp_path):
    store = RunStore(tmp_path / "runs")
    spec = _bench_spec_over(JOBS)
    first = index_legacy(store, [JOBS], [spec])
    second = index_legacy(store, [JOBS], [spec])
    assert [entry.cell_id for entry in first] == [entry.cell_id for entry in second]
    assert len(store.load_index("bench-grid")) == 2 and not store.load_index("legacy-sample")


def test_a_varied_field_is_not_compared_and_a_job_moves_out_of_legacy_once_matched(tmp_path):
    spec = _bench_spec_over(JOBS)
    cells = {cell.experiment_id: cell for cell in expand(spec)}
    job = next(iter_jobs(JOBS))
    # The same job as if it had run on OCI throughout, with no switch recorded.
    oci = {**job.resolved_config, "model": "openai.gpt-oss-120b", "semantic_code_judge_model": "openai.gpt-oss-120b"}
    oci_job = replace(job, resolved_config=oci, config=load_recorded_config(oci), provider_switches=[])
    assert matching_cell(oci_job, cells) is None
    assert matching_cell(oci_job, cells, varied=spec.varied) is not None
    assert matching_cell(oci_job, cells, varied=["model"]) is None  # the judge still differs

    store = RunStore(tmp_path / "runs")
    strict = spec.model_copy(update={"varied": []})
    index_legacy(store, [JOBS], [strict])
    assert len(store.load_index("legacy-sample")) == 0  # the fixture jobs carry switches, so they match anyway
    store.record_job("legacy-sample", cell_id="stale", experiment_id=job.experiment_id, config=oci,
                     job_dir=JOBS, job_id=job.job_id, submitted_at=None)
    assert "stale" in store.load_index("legacy-sample")
    index_legacy(store, [JOBS], [spec])
    assert "stale" not in store.load_index("legacy-sample")


def test_the_cli_reports_each_directory(tmp_path, capsys):
    spec_path = tmp_path / "bench.yaml"
    spec_path.write_text(
        BENCH_SPEC.read_text(encoding="utf-8").replace(
            "/data/bilel/lakegen/jobs/lakegen_jobs_bench", str(JOBS)
        ),
        encoding="utf-8",
    )
    assert cli.main(["index-legacy", str(JOBS), "--spec", str(spec_path), "--store", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert f"{JOBS}: 2 jobs, 2 matched bench-grid, 0 under legacy" in out
