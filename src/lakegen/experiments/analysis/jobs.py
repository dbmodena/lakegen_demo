"""An API job directory, read as stored.

The API keeps each batch as three files under its job directory: ``<id>.json``
(status, timestamps, the resolved configuration under ``settings``, and the
notes a hand-edit left: ``provider_switches``, ``requeues``),
``<id>.questions.json`` (the questions with their benchmark ``log_fields``)
and ``<id>.results.jsonl`` (one line per finished question, in the order
they finished). Nothing here validates or fills in: the configuration is
read with ``load_recorded_config`` so an old job says only what it recorded.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lakegen.config.recorded import RecordedConfig, load_recorded_config

QUESTIONS_SUFFIX = ".questions.json"
RESULTS_SUFFIX = ".results.jsonl"


def source_key(entry: dict[str, Any]) -> str:
    """The key a question and its result share: the source id, else the source path."""
    source_id = entry.get("source_id")
    return str(source_id) if source_id is not None else str(entry.get("source_path"))


@dataclass(frozen=True)
class StoredJob:
    path: Path
    job_id: str
    status: str
    created_at: str | None
    started_at: str | None
    finished_at: str | None
    resolved_config: dict[str, Any]
    config: RecordedConfig
    questions: list[dict[str, Any]]
    results: list[dict[str, Any]]
    provider_switches: list[dict[str, Any]] = field(default_factory=list)
    requeues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def job_dir(self) -> Path:
        return self.path.parent

    @property
    def experiment_id(self) -> str:
        return str(self.resolved_config.get("experiment_id") or "")

    def question(self, entry: dict[str, Any]) -> dict[str, Any] | None:
        """The job's question record of a result, by source key."""
        key = source_key(entry)
        return next((question for question in self.questions if source_key(question) == key), None)


def is_job_file(path: Path) -> bool:
    return path.suffix == ".json" and not path.name.endswith(QUESTIONS_SUFFIX)


@dataclass(frozen=True)
class JobProgress:
    """What the job file alone says, without reading the results: for listings."""

    job_id: str
    status: str
    question_count: int
    processed: int
    failed: int
    created_at: str | None
    finished_at: str | None
    error: str


def read_progress(path: Path) -> JobProgress | None:
    """The job's progress from its job file; None when the file is missing or unreadable."""
    path = Path(path)
    try:
        job = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return JobProgress(
        job_id=str(job.get("job_id") or path.stem),
        status=str(job.get("status") or ""),
        question_count=int(job.get("question_count") or 0),
        processed=int(job.get("processed") or 0),
        failed=int(job.get("failed") or 0),
        created_at=job.get("created_at"),
        finished_at=job.get("finished_at"),
        error=str(job.get("error") or ""),
    )


def read_job(path: Path) -> StoredJob:
    path = Path(path)
    job = json.loads(path.read_text(encoding="utf-8"))
    questions_path = path.with_name(path.stem + QUESTIONS_SUFFIX)
    results_path = path.with_name(path.stem + RESULTS_SUFFIX)
    questions = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.is_file() else []
    results = [
        json.loads(line)
        for line in (results_path.read_text(encoding="utf-8").splitlines() if results_path.is_file() else [])
        if line.strip()
    ]
    resolved = dict((job.get("settings") or {}).get("resolved_config") or {})
    return StoredJob(
        path=path,
        job_id=str(job.get("job_id") or path.stem),
        status=str(job.get("status") or ""),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        finished_at=job.get("finished_at"),
        resolved_config=resolved,
        config=load_recorded_config(resolved),
        questions=questions,
        results=results,
        provider_switches=list(job.get("provider_switches") or []),
        requeues=list(job.get("requeues") or []),
    )


def iter_jobs(job_dir: Path) -> Iterator[StoredJob]:
    """Every job of a directory, oldest first."""
    paths = [path for path in Path(job_dir).glob("*.json") if is_job_file(path)]
    jobs = [read_job(path) for path in paths]
    yield from sorted(jobs, key=lambda job: (job.created_at or "", job.job_id))
