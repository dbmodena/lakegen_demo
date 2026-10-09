"""Cells reach an API the way run_grid.py sent them, and runs are idempotent."""

from __future__ import annotations

import io
import json
import textwrap
import urllib.parse
from pathlib import Path

from lakegen.experiments import cli
from lakegen.experiments.executors import ApiExecutor, submission
from lakegen.experiments.spec import ExperimentSpec, expand
from lakegen.experiments.store import RunStore

ROOT = Path(__file__).resolve().parents[2]


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeApi:
    """Answers /health, /v1/batches and /v1/batches/{id} like the API."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict | None, dict]] = []
        self.jobs: dict[str, dict] = {}

    def __call__(self, request, timeout):
        url = urllib.parse.urlsplit(request.full_url)
        params = dict(urllib.parse.parse_qsl(url.query))
        body = json.loads(request.data) if request.data else None
        self.requests.append((request.get_method(), url.path, body, params))
        if url.path == "/health":
            payload = {"status": "ok"}
        elif url.path == "/v1/batches":
            job_id = f"{len(self.jobs) + 1:032x}"
            self.jobs[job_id] = {"job_id": job_id, "status": "queued", "processed": 0, "question_count": len(body["cases"]), "failed": 0}
            payload = {"job_id": job_id, "status": "queued", "question_count": len(body["cases"])}
        else:
            payload = self.jobs[url.path.rsplit("/", 1)[1]]
        return FakeResponse(json.dumps(payload).encode("utf-8"))


def _spec(tmp_path: Path) -> ExperimentSpec:
    path = tmp_path / "spec.yaml"
    path.write_text(textwrap.dedent("""
        name: smoke
        base: [config/base.yaml]
        grid: {lake: [uk], retrieval.mode: [keyword, semantic]}
        questions: {set: "20q_{lake}", slice: ":3"}
        experiment_id: "{name}-{lake}-{mode}"
        execution: {backend: api, api_url: http://api.test, stagger_seconds: 60}
        """), encoding="utf-8")
    return ExperimentSpec.load(path)


def test_a_submission_carries_the_config_inline_and_the_axes_as_parameters(tmp_path):
    (cell, _) = expand(_spec(tmp_path))
    body, params = submission(cell, [{"question": "q"}])
    assert body["config"] == cell.config.model_dump(mode="json") and body["cases"] == [{"question": "q"}]
    assert params == {"core": "uk", "retrieval_mode": "keyword", "experiment_id": "smoke-uk-keyword"}


def test_run_submits_every_cell_once_with_the_stagger(tmp_path, capsys):
    spec, api, store = _spec(tmp_path), FakeApi(), RunStore(tmp_path / "runs")
    executor = ApiExecutor("http://api.test/", jobs=tmp_path / "jobs", opener=api)
    naps: list[float] = []

    assert cli.run(spec, store, executor, count=None, only=None, dry_run=False, sleep=naps.append) == 0
    assert [path for _, path, _, _ in api.requests] == ["/v1/batches", "/v1/batches"]
    assert [body["cases"].__len__() for _, _, body, _ in api.requests] == [3, 3]
    assert naps == [60.0]
    assert len(store.submitted("smoke")) == 2
    first = next(iter(store.load_index("smoke").values()))
    assert first.backend == "api" and first.job_dir == str(tmp_path / "jobs") and len(first.job_id) == 32

    # A second run skips both cells and submits nothing.
    assert cli.run(spec, store, executor, count=None, only=None, dry_run=False, sleep=naps.append) == 0
    assert len(api.requests) == 2 and naps == [60.0]
    assert capsys.readouterr().out.count("skip   ") == 2


def test_plan_dry_run_and_status(tmp_path, capsys):
    spec, api, store = _spec(tmp_path), FakeApi(), RunStore(tmp_path / "runs")
    executor = ApiExecutor("http://api.test", opener=api)
    assert cli.plan(spec, store) == 0
    assert "smoke: 2 cells, 0 already submitted" in capsys.readouterr().out
    assert cli.run(spec, store, executor, count=2, only="semantic", dry_run=True) == 0
    out = capsys.readouterr().out
    assert out.count("would  ") == 1 and "2 cases" in out and api.requests == []
    assert cli.run(spec, store, executor, count=1, only="semantic", dry_run=False, sleep=lambda _s: None) == 0
    assert cli.status(spec, store, executor) == 0
    assert "smoke-uk-semantic" in capsys.readouterr().out and "queued" in capsys.readouterr().out or True
    assert executor.healthy()
