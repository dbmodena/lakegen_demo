"""The specs under experiments/specs reproduce the jobs their shell scripts submitted.

Each cell is passed through the API's own resolver, as a submission would
be, and compared with the configuration stored in the job the old script
created. A stored configuration is compared as it is, because it is what the
API ran with. One experiment id can hold several stored configurations (a
smoke job shares the id of its grid cell); a cell matches when it equals any
of them. Fields a job never recorded (it predates them) and the question file
(the old passes used generated subset files where a spec slices the 100q set)
are not compared.

The benchmark-lake jobs are a patchwork: their model, judge and entity model
were switched by hand, job by job, between OCI and Together as credit came
and went, so no single spec reproduces those three fields. The bench spec
records the intended provider, Together, and names the three fields as
`varied`; they are left out of its comparison and the switches are counted.
`lakegen.analysis.load_runs` reports what each result ran on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lakegen.experiments.legacy import config_differences, resolve_as_submission
from lakegen.experiments.spec import ExperimentSpec, expand

ROOT = Path(__file__).resolve().parents[1]
SPECS = sorted((ROOT / "experiments" / "specs").glob("*.yaml"))
STORED = json.loads((ROOT / "tests" / "fixtures" / "configs" / "stored_job_configs.json").read_text(encoding="utf-8"))
# spec name -> (the job directory its script wrote to, ids that are not grid cells)
GRIDS = {
    "qwen-grid": ("lakegen_jobs_qwen", ("qwen-llm-smoke", "qwen-jev-smoke")),
    "rest-run": ("lakegen_jobs_rest", ()),
    "bench-grid": ("lakegen_jobs_bench", ()),
    "portal-grid2": ("lakegen_jobs", ("grid-", "smoke-")),
}


def _stored_jobs(job_dir: str, skipped: tuple[str, ...]) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """The stored configurations of a grid by experiment id, and how many were switched by hand."""
    jobs: dict[str, list[dict[str, Any]]] = {}
    switched = 0
    for entry in STORED.values():
        if not any(job.startswith(job_dir + "/") for job in entry["jobs"]):
            continue
        experiment_id = entry["config"]["experiment_id"]
        if any(experiment_id.startswith(prefix) for prefix in skipped):
            continue
        jobs.setdefault(experiment_id, []).append(entry["config"])
        switched += bool(entry["switches"])
    return jobs, switched


@pytest.mark.parametrize("spec_path", SPECS, ids=[p.stem for p in SPECS])
def test_spec_cells_match_the_submitted_jobs(spec_path: Path):
    spec = ExperimentSpec.load(spec_path)
    job_dir, skipped_ids = GRIDS[spec.name]
    stored, switched = _stored_jobs(job_dir, skipped_ids)
    cells = {cell.experiment_id: cell for cell in expand(spec)}
    assert len(cells) == len(expand(spec)), "experiment ids must be unique"
    skipped_keys = set(spec.varied)  # the fields the spec says were not held fixed

    mismatches: dict[str, dict[str, tuple[Any, Any]]] = {}
    for experiment_id, configs in stored.items():
        assert experiment_id in cells, f"the spec has no cell for the stored job {experiment_id}"
        resolved = resolve_as_submission(cells[experiment_id])
        attempts = [config_differences(resolved, config, ignored=skipped_keys) for config in configs]
        if all(attempts):
            mismatches[experiment_id] = min(attempts, key=len)
    assert not mismatches, json.dumps(mismatches, indent=1, default=str)[:4000]
    never_submitted = sorted(set(cells) - set(stored))
    print(
        f"{spec.name}: {len(stored)} stored ids matched ({switched} configurations switched by hand"
        f"{', ' + ', '.join(spec.varied) + ' not compared' if spec.varied else ''}); "
        f"{len(never_submitted)} planned cells never submitted: {never_submitted[:6]}"
    )
