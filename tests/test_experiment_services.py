"""Services are probed, started and stopped; a run refuses while a required one is down."""

from __future__ import annotations

import io
import os
import sys
import time
import urllib.error
from pathlib import Path

import pytest

from lakegen.experiments import cli
from lakegen.experiments.executors import ApiExecutor
from lakegen.experiments.services import (
    Service,
    ServiceManager,
    load_services,
    missing_requirements,
    requirements_of,
    service_for,
)
from lakegen.experiments.spec import ExperimentSpec, expand
from lakegen.experiments.store import RunStore
from test_experiment_executors import FakeApi


class Answering:
    """An opener that answers 200 for the given URLs and refuses the rest."""

    def __init__(self, *urls: str) -> None:
        self.urls = set(urls)
        self.asked: list[str] = []

    def __call__(self, url, timeout):
        self.asked.append(url)
        if url in self.urls:
            response = io.BytesIO(b"{}")
            response.status = 200
            response.__enter__ = lambda s=response: s
            response.__exit__ = lambda s, *exc: False
            return response
        raise urllib.error.URLError("refused")


def _spec(tmp_path: Path, **extra) -> ExperimentSpec:
    path = tmp_path / "spec.yaml"
    body = {
        "name": "needs", "base": ["config/base.yaml"],
        "grid": {"lake": ["uk", "valencia"], "retrieval.mode": ["keyword", "pneuma"]},
        "questions": {"set": "100q_{lake}", "slice": ":2"},
        "experiment_id": "{name}-{lake}-{mode}",
        "execution": {"backend": "api", "api_url": "http://api.test"},
        "requires": ["pneuma:{lake}", "ollama"],
        **extra,
    }
    import yaml

    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return ExperimentSpec.load(path)


def test_services_file_yields_one_pneuma_service_per_lake_on_its_port():
    services = load_services()
    assert services["pneuma:valencia"].health == "http://127.0.0.1:8771/health"
    assert "--port 8771 --out-path /data/bilel/pneuma/valencia" in services["pneuma:valencia"].command
    assert services["pneuma:uk"].base_url == "http://127.0.0.1:8767"  # ours, not the registry's 8766
    assert services["pneuma:nyc"].base_url == "http://127.0.0.1:8774"
    assert services["ollama"].health and services["tunnel"].health is None
    assert services["api"].env == {"PYTHONPATH": ".:src"}
    with pytest.raises(KeyError, match="unknown service 'nope'; known: api"):
        service_for("nope", services)


def test_status_up_and_down_of_a_real_process(tmp_path):
    sleeper = Service("sleeper", f"{sys.executable} -c 'import time; time.sleep(60)'")
    manager = ServiceManager(tmp_path / "services", opener=Answering())
    assert not manager.status(sleeper).up
    status = manager.up(sleeper)
    try:
        assert status.up and status.pid and not status.healthy
        assert manager.status(sleeper).pid == status.pid
        assert (tmp_path / "services" / "sleeper.pid").read_text() == str(status.pid)
        assert manager.up(sleeper).pid == status.pid  # already up: not started again
    finally:
        assert manager.down(sleeper)
    assert not manager.down(sleeper)
    for _ in range(50):  # the process exits on SIGTERM; reap it so it leaves no zombie
        if os.waitpid(status.pid, os.WNOHANG)[0] == status.pid:
            break
        time.sleep(0.05)


def test_a_healthy_service_is_not_started(tmp_path):
    started = []
    manager = ServiceManager(tmp_path, opener=Answering("http://h/health"), popen=lambda *a, **k: started.append(a))
    service = Service("h", "never-run", health="http://h/health")
    assert manager.up(service).healthy and started == []
    assert not manager.status(Service("d", "x", health="http://d/health")).up


def test_requirements_follow_the_cell_and_pneuma_counts_for_pneuma_cells_only(tmp_path):
    spec = _spec(tmp_path)
    cells = expand(spec)
    assert requirements_of(cells[0], spec.requires) == ["pneuma:uk", "ollama"]
    services = load_services()

    everything_down = ServiceManager(tmp_path, opener=Answering())
    missing = missing_requirements(cells, spec.requires, services, everything_down)
    assert [line.split(":")[0] for line in missing] == ["ollama", "pneuma", "pneuma"]
    assert "pneuma:uk: http://localhost:8766 does not answer; start it with: services up pneuma:uk (it listens on http://127.0.0.1:8767" in missing[1]

    pneuma_up = ServiceManager(tmp_path, opener=Answering(
        "http://127.0.0.1:11434/api/version", "http://localhost:8766/health", "http://localhost:8771/health",
    ))
    assert missing_requirements(cells, spec.requires, services, pneuma_up) == []
    keyword_only = [cell for cell in cells if cell.config.retrieval.mode.value == "keyword"]
    assert [line.split(":")[0] for line in missing_requirements(keyword_only, spec.requires, services, everything_down)] == ["ollama"]


def test_run_refuses_while_a_required_service_is_down(tmp_path, capsys):
    spec, api = _spec(tmp_path), FakeApi()
    store, executor = RunStore(tmp_path / "runs"), ApiExecutor("http://api.test", opener=api)
    manager = ServiceManager(tmp_path / "services", opener=Answering())
    assert cli.run(spec, store, executor, count=1, only=None, dry_run=False, services=manager) == 1
    out = capsys.readouterr().out
    assert "not submitting: a required service is down" in out and "ollama: down" in out
    assert api.requests == [] and store.submitted("needs") == set()
    # A dry run reports the cells without asking the services.
    assert cli.run(spec, store, executor, count=1, only=None, dry_run=True, services=manager) == 0


def test_services_status_command_lists_every_service(tmp_path, capsys):
    manager = ServiceManager(tmp_path, opener=Answering("http://127.0.0.1:11434/api/version"))
    assert cli.main(["services", "status"], services=manager) == 0
    out = capsys.readouterr().out
    assert "ollama               up" in out and "pneuma:uk            down" in out
    assert cli.main(["services", "status", "ollama", "tunnel"], services=manager) == 0
    assert capsys.readouterr().out.count("\n") == 2
    assert cli.main(["services", "down", "tunnel"], services=manager) == 0
    assert "not started by services up" in capsys.readouterr().out
