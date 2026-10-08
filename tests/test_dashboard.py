"""The run dashboard serves the store's experiments, cells and questions, behind an optional token."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lakegen.experiments.legacy import index_legacy
from lakegen.experiments.services import ServiceManager
from lakegen.experiments.spec import ExperimentSpec
from lakegen.experiments.store import RunStore
from lakegen.interfaces.dashboard.app import create_app

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "tests" / "fixtures" / "jobs" / "sample"


def _store(tmp_path: Path) -> RunStore:
    store = RunStore(tmp_path / "runs")
    spec = ExperimentSpec.load(ROOT / "experiments" / "specs" / "bench_grid.yaml")
    spec = spec.model_copy(update={"execution": spec.execution.model_copy(update={"job_dir": str(JOBS)})})
    index_legacy(store, [JOBS], [spec])
    return store


def _down(tmp_path: Path) -> ServiceManager:
    import urllib.error

    def refuse(url, timeout):
        raise urllib.error.URLError("down")

    return ServiceManager(tmp_path / "services", opener=refuse)


def test_experiments_cells_and_questions(tmp_path):
    client = TestClient(create_app(store=_store(tmp_path), services=_down(tmp_path), token=""))
    assert client.get("/health").json() == {"status": "ok"}
    [experiment] = client.get("/api/experiments").json()
    assert experiment["name"] == "bench-grid" and experiment["cells"] == 2 and not experiment["legacy"]
    assert experiment["by_status"] == {"completed": 2} and experiment["processed"] == 8

    detail = client.get("/api/experiments/bench-grid").json()
    assert detail["runs"] == 8 and len(detail["cells"]) == 2
    cell = next(c for c in detail["cells"] if c["experiment_id"] == "bench-llm-target_bird-hybrid")
    assert cell["status"] == "completed" and cell["processed"] == 4 and cell["summary"]["n"] == 4
    assert cell["summary"]["effective_models"] == ["openai.gpt-oss-120b", "together.gpt-oss-120b"]

    questions = client.get(f"/api/experiments/bench-grid/cells/{cell['cell_id']}").json()
    assert questions["cell"]["experiment_id"] == "bench-llm-target_bird-hybrid"
    assert [q["question_id"] for q in questions["questions"]] == ["bird_281", "bird_115", "bird_711", "bird_987"]
    assert questions["questions"][0]["gold_ranks"] == {"toxicology_atom": 1, "toxicology_molecule": 2}

    assert client.get("/api/experiments/nope").status_code == 404
    assert client.get("/api/experiments/bench-grid/cells/nope").status_code == 404
    services = client.get("/api/services").json()
    assert {s["name"] for s in services} >= {"ollama", "pneuma:uk", "runs_dashboard"} and not any(s["up"] for s in services)
    page = client.get("/")
    assert page.status_code == 200 and "LakeGen runs" in page.text


def test_the_token_is_required_once_then_kept_in_a_cookie(tmp_path):
    client = TestClient(create_app(store=_store(tmp_path), services=_down(tmp_path), token="s3cret"))
    assert client.get("/health").status_code == 200
    assert client.get("/api/experiments").status_code == 401
    assert client.get("/").status_code == 401
    assert client.get("/?token=s3cret").status_code == 200
    assert client.get("/api/experiments").status_code == 200  # the cookie carries it now


def test_an_empty_store_lists_nothing(tmp_path):
    client = TestClient(create_app(store=RunStore(tmp_path / "none"), services=_down(tmp_path), token=""))
    assert client.get("/api/experiments").json() == []
