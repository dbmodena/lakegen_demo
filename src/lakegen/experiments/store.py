"""The run store: what each experiment submitted, under the state root.

    <state root>/runs/<experiment>/
        spec.yaml                 the spec as it was run
        cells/<cell id>/config.yaml   the resolved configuration of the cell
        cells/<cell id>/cell.json     its axes, pass and question slice
        index.json                cell id -> where its results are

Results are never copied: a cell submitted to an API points at that API's job
(``job_dir`` and ``job_id``); a legacy job indexed with ``index-legacy`` points
at its job directory the same way. ``lakegen.analysis`` reads jobs through the
index.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from lakegen.core import paths
from lakegen.experiments.spec import Cell, ExperimentSpec


@dataclass(frozen=True)
class CellRecord:
    cell_id: str
    experiment_id: str
    axes: dict[str, Any]
    pass_name: str | None
    questions_file: str
    questions_slice: str | None
    backend: str
    job_dir: str | None
    job_id: str | None
    submitted_at: str | None


class RunStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or paths.runs_dir()

    def experiment_dir(self, name: str) -> Path:
        return self.root / name

    def write_spec(self, spec: ExperimentSpec) -> Path:
        directory = self.experiment_dir(spec.name)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "spec.yaml"
        target.write_text(yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False), encoding="utf-8")
        return target

    def write_cell(self, cell: Cell) -> Path:
        directory = self.experiment_dir(cell.spec_name) / "cells" / cell.cell_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text(
            yaml.safe_dump(cell.config.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        (directory / "cell.json").write_text(json.dumps({
            "cell_id": cell.cell_id,
            "experiment_id": cell.experiment_id,
            "axes": cell.axes,
            "pass": cell.pass_name,
            "questions_file": str(cell.questions_file),
            "questions_slice": cell.questions_slice,
        }, indent=2), encoding="utf-8")
        return directory

    def _index_path(self, name: str) -> Path:
        return self.experiment_dir(name) / "index.json"

    def load_index(self, name: str) -> dict[str, CellRecord]:
        path = self._index_path(name)
        if not path.is_file():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {cell_id: CellRecord(**record) for cell_id, record in raw.items()}

    def record(
        self,
        cell: Cell,
        *,
        backend: str,
        job_dir: Path | None,
        job_id: str | None,
        submitted_at: str | None = None,
    ) -> CellRecord:
        """Note where a cell's results live; the cell's files are written if missing."""
        self.write_cell(cell)
        entry = CellRecord(
            cell_id=cell.cell_id,
            experiment_id=cell.experiment_id,
            axes=cell.axes,
            pass_name=cell.pass_name,
            questions_file=str(cell.questions_file),
            questions_slice=cell.questions_slice,
            backend=backend,
            job_dir=str(job_dir) if job_dir else None,
            job_id=job_id,
            submitted_at=submitted_at or datetime.now(UTC).isoformat(),
        )
        self._add(cell.spec_name, entry)
        return entry

    def record_job(
        self,
        experiment: str,
        *,
        cell_id: str,
        experiment_id: str,
        config: dict[str, Any],
        job_dir: Path,
        job_id: str,
        submitted_at: str | None,
        questions_file: str = "",
    ) -> CellRecord:
        """Note a job that was not submitted through a spec, under ``experiment``.

        The cell's files are written from the job's stored configuration
        unless a spec's cell already wrote them.
        """
        directory = self.experiment_dir(experiment) / "cells" / cell_id
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / "config.yaml").is_file():
            (directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if not (directory / "cell.json").is_file():
            (directory / "cell.json").write_text(json.dumps({
                "cell_id": cell_id, "experiment_id": experiment_id, "axes": {}, "pass": None,
                "questions_file": questions_file, "questions_slice": None,
            }, indent=2), encoding="utf-8")
        entry = CellRecord(
            cell_id=cell_id,
            experiment_id=experiment_id,
            axes={},
            pass_name=None,
            questions_file=questions_file,
            questions_slice=None,
            backend="api",
            job_dir=str(job_dir),
            job_id=job_id,
            submitted_at=submitted_at,
        )
        self._add(experiment, entry)
        return entry

    def _add(self, experiment: str, entry: CellRecord) -> None:
        index = self.load_index(experiment)
        index[entry.cell_id] = entry
        self._write_index(experiment, index)

    def _write_index(self, name: str, index: dict[str, CellRecord]) -> None:
        path = self._index_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({cell_id: asdict(record) for cell_id, record in index.items()}, indent=2),
            encoding="utf-8",
        )

    def submitted(self, name: str) -> set[str]:
        return {cell_id for cell_id, record in self.load_index(name).items() if record.job_id}
