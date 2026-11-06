"""Where a cell runs: a LakeGen API over HTTP, or this process.

``ApiExecutor`` submits a cell the way experiments/portal_grid/run_grid.py
did: the resolved configuration inline, the cases in the body, and the lake,
retrieval mode and experiment id as query parameters, which the API applies as
explicit overrides. ``LocalExecutor`` hands the same payload to the API
module's own batch machinery in this process, so a local run writes the same
job files, with the same pacing, metrics and recovery, without a server.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from experiment_runner.batches import batch, jobs
from experiment_runner.spec import Cell
from lakegen.entrypoints.api.submission import resolve_config
from lakegen.pipeline.service import extract_questions


class ExecutionError(RuntimeError):
    """A submission the executor could not make."""


class Executor(Protocol):
    """Where cells run: submits a cell's cases as a job and reports on the job.

    Attributes:
        backend: The backend's name, recorded with each submitted cell.
    """

    backend: str

    def job_dir(self) -> Path | None:
        """The directory the jobs' files are written to, if known."""
        ...

    def submit(self, cell: Cell, cases: list[dict[str, Any]]) -> str:
        """Submit a cell's cases as one job; the job's id."""
        ...

    def job(self, job_id: str) -> dict[str, Any]:
        """The job's record, without its results."""
        ...


def submission(cell: Cell, cases: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    """The body and the query parameters of a cell's batch submission."""
    body = {"config": cell.config.model_dump(mode="json"), "cases": cases}
    params = {
        "core": cell.lake,
        "retrieval_mode": cell.config.retrieval.mode.value,
        "experiment_id": cell.experiment_id,
    }
    return body, params


class ApiExecutor:
    """Submits cells to a LakeGen API over HTTP.

    Attributes:
        backend: ``"api"``.
        api_url: The API's base URL, without a trailing slash.

    Args:
        api_url: The API's base URL.
        jobs: The API's job directory, recorded with each cell, if known.
        opener: The function opening a request (a test replaces it).
        timeout: The timeout of each request, in seconds.
    """

    backend = "api"

    def __init__(
        self,
        api_url: str,
        *,
        jobs: Path | None = None,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout: float = 120.0,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self._jobs = jobs
        self._opener = opener
        self._timeout = timeout

    def job_dir(self) -> Path | None:
        """The API's job directory, as given."""
        return self._jobs

    def _request(
        self, path: str, *, body: dict[str, Any] | None = None, params: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Call the API: a POST of ``body`` as JSON, or a GET without one.

        Args:
            path: The endpoint's path.
            body: The JSON body; None for a GET.
            params: The query parameters, if any.

        Returns:
            The decoded JSON response.

        Raises:
            ExecutionError: When the API answers with an HTTP error or cannot be reached.
        """
        url = f"{self.api_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with self._opener(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise ExecutionError(f"{self.api_url}{path}: HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ExecutionError(f"{self.api_url}{path}: {exc.reason}") from exc

    def healthy(self) -> bool:
        """Whether the API answers its health check with ``ok``."""
        try:
            return self._request("/health").get("status") == "ok"
        except ExecutionError:
            return False

    def submit(self, cell: Cell, cases: list[dict[str, Any]]) -> str:
        """Post the cell's batch to the API; the job's id."""
        body, params = submission(cell, cases)
        return str(self._request("/v1/batches", body=body, params=params)["job_id"])

    def job(self, job_id: str) -> dict[str, Any]:
        """The job's record as the API returns it, without its results."""
        return self._request(f"/v1/batches/{job_id}", params={"include_results": "false"})


class LocalExecutor:
    """Runs cells through the batch loop in this process, as the API would.

    Attributes:
        backend: ``"local"``.
    """

    backend = "local"

    def job_dir(self) -> Path | None:
        """The job directory of this process's batch machinery."""
        return Path(jobs.JOB_DIR)

    def submit(self, cell: Cell, cases: list[dict[str, Any]]) -> str:
        """Queue the cell's batch on this process's worker pool, as the API would; the job's id."""
        body, params = submission(cell, cases)
        resolved = resolve_config(
            core=params["core"],
            model=cell.config.model,
            retrieval_mode=params["retrieval_mode"],
            top_k=cell.config.retrieval.top_k,
            hybrid_alpha=cell.config.retrieval.alpha,
            candidate_multiplier=cell.config.retrieval.candidate_multiplier,
            experiment_id=params["experiment_id"],
            config_data=body["config"],
            explicit_fields={"core", "retrieval_mode", "experiment_id"},
        )
        records = [
            {
                "question": item.question,
                "source_path": item.path,
                "source_id": item.source_id,
                "log_fields": item.log_fields(),
            }
            for item in extract_questions(body)
        ]
        return str(batch.enqueue(records, resolved)["job_id"])

    def job(self, job_id: str) -> dict[str, Any]:
        """The job's record without its results.

        Args:
            job_id: The job's id.

        Returns:
            The job's snapshot.

        Raises:
            ExecutionError: When no such job exists.
        """
        snapshot = jobs.snapshot_job(job_id, include_results=False)
        if snapshot is None:
            raise ExecutionError(f"unknown job {job_id}")
        return snapshot

    def wait(self, job_id: str, *, poll_seconds: float = 2.0) -> dict[str, Any]:
        """Poll the job until it is completed, failed or parked.

        Args:
            job_id: The job's id.
            poll_seconds: The time between polls.

        Returns:
            The job's final snapshot.
        """
        while True:
            snapshot = self.job(job_id)
            if snapshot.get("status") in {"completed", "failed", "parked"}:
                return snapshot
            time.sleep(poll_seconds)


def make_executor(backend: str, *, api_url: str, job_dir: str | None) -> Executor:
    """The executor of a spec's backend.

    Args:
        backend: ``api`` or ``local``.
        api_url: The API's base URL, for ``api``.
        job_dir: The API's job directory, for ``api``, if known.

    Returns:
        The executor.

    Raises:
        ExecutionError: When the backend is neither.
    """
    if backend == "api":
        return ApiExecutor(api_url, jobs=Path(job_dir) if job_dir else None)
    if backend == "local":
        return LocalExecutor()
    raise ExecutionError(f"unknown backend {backend!r}; use api or local")
