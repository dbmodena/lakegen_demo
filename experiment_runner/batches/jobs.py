"""The batch jobs of the API: their files on disk and the registry of the running ones.

A job is a JSON file with its settings and progress, a questions file (the
immutable input, used to resume) and a results file with one line per
answered question. ``JOB_DIR`` names the folder (``LAKEGEN_JOB_DIR``), so an
API instance neither shows nor resumes the jobs another instance left
unfinished. ``JOBS`` holds the jobs this process created or recovered;
``snapshot_job`` reads a finished job from disk when it is not there.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lakegen.config import paths

# LAKEGEN_JOB_DIR gives an API instance its own job folder, so a second run
# neither shows nor resumes the jobs another instance left unfinished.
JOB_DIR = paths.jobs_dir()
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


def now() -> str:
    """The current UTC time as an ISO timestamp."""
    return datetime.now(UTC).isoformat()


def job_path(job_id: str) -> Path:
    """The job file, ``<id>.json``."""
    return JOB_DIR / f"{job_id}.json"


def results_path(job_id: str) -> Path:
    """The results file, ``<id>.results.jsonl``."""
    return JOB_DIR / f"{job_id}.results.jsonl"


def questions_path(job_id: str) -> Path:
    """The questions file, ``<id>.questions.json``."""
    return JOB_DIR / f"{job_id}.questions.json"


def valid_job_id(job_id: str) -> bool:
    """Whether the id is a job id: 32 lowercase hex digits, so it is safe in a path."""
    return len(job_id) == 32 and all(character in "0123456789abcdef" for character in job_id)


def persist_job(job: dict[str, Any]) -> None:
    """Write the job file atomically, without the results (they have their own file)."""
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    target = job_path(job["job_id"])
    temporary = target.with_suffix(".tmp")
    metadata = {key: value for key, value in job.items() if key != "results"}
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def persist_questions(job_id: str, questions: list[dict[str, Any]]) -> None:
    """Persist the immutable batch input used to resume interrupted jobs."""
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    target = questions_path(job_id)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def load_questions(job_id: str) -> list[dict[str, Any]] | None:
    """The job's questions; None for an invalid id, a missing file or one that holds no list."""
    if not valid_job_id(job_id):
        return None
    path = questions_path(job_id)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else None


def append_result(job_id: str, result: dict[str, Any]) -> None:
    """Append one line to the job's results file."""
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    with results_path(job_id).open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")


def snapshot_job(job_id: str, *, include_results: bool = True) -> dict[str, Any] | None:
    """A copy of the job: from the registry, else from its files on disk.

    Args:
        job_id: The job's id.
        include_results: Whether the copy carries the results.

    Returns:
        The job's record, or None for an invalid id or an unknown job.
    """
    if not valid_job_id(job_id):
        return None
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            snapshot = (
                job
                if include_results
                else {key: value for key, value in job.items() if key != "results"}
            )
            return json.loads(json.dumps(snapshot))
    path = job_path(job_id)
    if path.is_file():
        job = json.loads(path.read_text(encoding="utf-8"))
        if include_results:
            results_file = results_path(job_id)
            job["results"] = []
            if results_file.is_file():
                for line in results_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        job["results"].append(json.loads(line))
        return job
    return None


def update_job(job_id: str, **changes: Any) -> None:
    """Apply ``changes`` to a registered job, stamp it updated and persist it.

    Args:
        job_id: The job's id; the job must be in ``JOBS``.
        **changes: The fields to set.
    """
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(changes)
        job["updated_at"] = now()
        snapshot = json.loads(json.dumps(job))
    persist_job(snapshot)


def question_key(source: dict[str, Any]) -> str:
    """Return a stable per-question key even when benchmark IDs are absent."""
    source_id = source.get("source_id")
    return str(source_id) if source_id is not None else str(source.get("source_path"))


def new_job(
    job_id: str, questions: list[dict[str, Any]], resolved_config: dict[str, Any]
) -> dict[str, Any]:
    """A queued job's record, as the API creates it."""
    created = now()
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": created,
        "updated_at": created,
        "started_at": None,
        "finished_at": None,
        "question_count": len(questions),
        "processed": 0,
        "failed": 0,
        "metrics_logged": False,
        "settings": {"resolved_config": resolved_config},
        "results": [],
        "error": "",
    }


def register(job: dict[str, Any], questions: list[dict[str, Any]]) -> None:
    """Put the job in the registry and on disk, its questions first."""
    with JOBS_LOCK:
        JOBS[job["job_id"]] = job
    persist_questions(job["job_id"], questions)
    persist_job(job)
