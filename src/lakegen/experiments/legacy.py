"""Index the jobs the old submit scripts created, so they read like a spec's cells.

Before the specs, each grid was submitted by its own shell script to an API
with its own job directory, and the run store knows nothing of them.
``index_legacy`` walks job directories and records every job in the store:

* under the spec whose cell it matches, with that cell's id, so ``exp run``
  of the spec sees the cell as submitted and ``load_runs(spec)`` reads the
  job. A cell matches a job when the spec names the job's directory, the
  experiment ids are equal, and the cell's configuration, resolved by the
  API as a submission would be, equals the stored one apart from the fields
  the job never recorded, ``benchmark.path`` (the old passes used generated
  subset files) and, for a job whose models were switched by hand, the
  model fields;
* otherwise under ``legacy-<directory name>``, with the job id as its cell.

When several jobs match one cell (a cell re-submitted), the latest is the
cell's; the others go under the legacy experiment, so nothing is lost.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lakegen.config.recorded import load_recorded_config
from lakegen.experiments.analysis.jobs import StoredJob, iter_jobs
from lakegen.experiments.analysis.switches import MODEL_FIELDS
from lakegen.experiments.spec import Cell, ExperimentSpec, expand
from lakegen.experiments.store import RunStore
from lakegen.retrieval import DEFAULT_TOP_K

IGNORED_KEYS = frozenset({"benchmark.path"})


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def resolve_as_submission(cell: Cell) -> dict[str, Any]:
    """The cell's configuration as the API stores it for a submitted batch."""
    from lakegen.experiments.batch import resolve_config

    return resolve_config(
        core=cell.lake,
        model=cell.config.model,
        retrieval_mode=cell.config.retrieval.mode.value,
        top_k=DEFAULT_TOP_K,
        hybrid_alpha=0.5,
        candidate_multiplier=5,
        experiment_id=cell.experiment_id,
        config_data=cell.config.model_dump(mode="json"),
        explicit_fields={"core", "retrieval_mode", "experiment_id"},
    ).model_dump(mode="json")


def config_differences(
    resolved: Mapping[str, Any], stored: Mapping[str, Any], *, ignored: Iterable[str] = (),
) -> dict[str, tuple[Any, Any]]:
    """Dotted key -> (stored, resolved) where they differ; unrecorded fields are not compared."""
    expected, actual = flatten(dict(stored)), flatten(dict(resolved))
    skipped = set(ignored) | IGNORED_KEYS | set(load_recorded_config(stored).not_recorded)
    return {
        key: (expected.get(key, "<absent>"), actual.get(key, "<absent>"))
        for key in sorted(set(expected) | set(actual))
        if key not in skipped and expected.get(key, "<absent>") != actual.get(key, "<absent>")
    }


def matching_cell(job: StoredJob, cells: Mapping[str, Cell], *, varied: Iterable[str] = ()) -> Cell | None:
    """The cell of ``cells`` (by experiment id) whose configuration the job ran, if any.

    ``varied`` names the spec's fields that were not held fixed; a job whose
    models were switched by hand is also compared without its model fields.
    """
    cell = cells.get(job.experiment_id)
    if cell is None:
        return None
    ignored = set(varied) | (set(MODEL_FIELDS) if job.provider_switches else set())
    differences = config_differences(resolve_as_submission(cell), job.resolved_config, ignored=ignored)
    return cell if not differences else None


@dataclass(frozen=True)
class IndexedJob:
    job: StoredJob
    experiment: str
    cell_id: str
    matched: bool


def legacy_experiment(job_dir: Path) -> str:
    return f"legacy-{Path(job_dir).name}"


def index_legacy(
    store: RunStore, job_dirs: Iterable[Path], specs: Iterable[ExperimentSpec],
) -> list[IndexedJob]:
    """Record every job of ``job_dirs`` in the store; see the module docstring."""
    cells_by_dir: dict[Path, tuple[ExperimentSpec, dict[str, Cell]]] = {}
    for spec in specs:
        if spec.execution.job_dir:
            cells = {cell.experiment_id: cell for cell in expand(spec)}
            cells_by_dir[Path(spec.execution.job_dir).resolve()] = (spec, cells)
    indexed: list[IndexedJob] = []
    for job_dir in job_dirs:
        job_dir = Path(job_dir)
        spec, cells = cells_by_dir.get(job_dir.resolve(), (None, {}))
        taken: dict[str, StoredJob] = {}
        for job in iter_jobs(job_dir):  # oldest first: a later job of a cell replaces the earlier
            cell = matching_cell(job, cells, varied=spec.varied) if spec else None
            if cell is None:
                indexed.append(_record_legacy(store, job))
                continue
            if cell.cell_id in taken:
                indexed.append(_record_legacy(store, taken[cell.cell_id]))
            taken[cell.cell_id] = job
            store.write_spec(spec)
            store.record(cell, backend="api", job_dir=job.job_dir, job_id=job.job_id, submitted_at=job.created_at)
            indexed.append(IndexedJob(job=job, experiment=spec.name, cell_id=cell.cell_id, matched=True))
        # A job matched now but recorded under the legacy experiment by an
        # earlier pass (before the spec declared its varied fields) leaves it.
        store.forget(legacy_experiment(job_dir), {job.job_id for job in taken.values()})
    return indexed


def _record_legacy(store: RunStore, job: StoredJob) -> IndexedJob:
    experiment = legacy_experiment(job.job_dir)
    store.record_job(
        experiment,
        cell_id=job.job_id[:16],
        experiment_id=job.experiment_id,
        config=job.resolved_config,
        job_dir=job.job_dir,
        job_id=job.job_id,
        submitted_at=job.created_at,
        questions_file=str((job.resolved_config.get("benchmark") or {}).get("path") or ""),
    )
    return IndexedJob(job=job, experiment=experiment, cell_id=job.job_id[:16], matched=False)
