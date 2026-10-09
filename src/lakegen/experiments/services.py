"""The services an experiment needs, and whether they are up.

``config/services.yaml`` names each service with its health URL and the
command that starts it on this machine; one ``pneuma:<lake>`` service per
lake comes from its ``pneuma`` template. ``ServiceManager`` probes, starts
(detached, output and pid under ``<state root>/services``) and stops them.
``missing_requirements`` is what ``exp run`` asks before submitting: a spec's
``requires`` entries, ``{lake}`` filled in per cell, each resolved to a
service and probed; a Pneuma requirement is probed at the URL the cell will
query, which must be its service's.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lakegen.config import paths
from lakegen.config.locations import BASE_DIR
from lakegen.config.registry import CONFIG_DIR, get_lake, lake_names
from lakegen.config.retrieval import RetrievalMode
from lakegen.experiments.spec import Cell

SERVICES_FILE = CONFIG_DIR / "services.yaml"


@dataclass(frozen=True)
class Service:
    name: str
    command: str
    health: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def base_url(self) -> str | None:
        if not self.health:
            return None
        parts = urllib.parse.urlsplit(self.health)
        return f"{parts.scheme}://{parts.netloc}"


@dataclass(frozen=True)
class ServiceStatus:
    service: Service
    healthy: bool
    pid: int | None

    @property
    def up(self) -> bool:
        return self.healthy or (self.service.health is None and self.pid is not None)


def _pneuma_services(template: Mapping[str, Any]) -> dict[str, Service]:
    services = {}
    ports = template.get("ports") or {}
    for lake in lake_names():
        index_path = (template.get("index_paths") or {}).get(lake)
        if not index_path:
            continue
        port = ports.get(lake) or urllib.parse.urlsplit(get_lake(lake).pneuma_route()[0]).port
        values = {"port": port, "index_path": index_path, "lake": lake}
        services[f"pneuma:{lake}"] = Service(
            name=f"pneuma:{lake}",
            command=str(template["command"]).format(**values),
            health=str(template["health"]).format(**values) if template.get("health") else None,
            description=f"the Pneuma service of the {lake} lake ({index_path})",
        )
    return services


def load_services(path: Path = SERVICES_FILE) -> dict[str, Service]:
    """Every service by name: the file's entries, then ``pneuma:<lake>`` for each lake with an index."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    services = {
        name: Service(
            name=name,
            command=str(entry["command"]),
            health=entry.get("health"),
            env={key: str(value) for key, value in (entry.get("env") or {}).items()},
            description=str(entry.get("description") or ""),
        )
        for name, entry in (document.get("services") or {}).items()
    }
    if document.get("pneuma"):
        services.update(_pneuma_services(document["pneuma"]))
    return services


def service_for(requirement: str, services: Mapping[str, Service]) -> Service:
    try:
        return services[requirement]
    except KeyError:
        raise KeyError(f"unknown service {requirement!r}; known: {', '.join(sorted(services))}") from None


class ServiceManager:
    """Probes, starts and stops services; logs and pids live under ``root``."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
        popen: Callable[..., Any] = subprocess.Popen,
        timeout: float = 3.0,
    ) -> None:
        self.root = root or paths.services_dir()
        self._opener = opener
        self._popen = popen
        self._timeout = timeout

    def _pid_file(self, service: Service) -> Path:
        return self.root / f"{service.name.replace(':', '_')}.pid"

    def log_file(self, service: Service) -> Path:
        return self.root / f"{service.name.replace(':', '_')}.log"

    def answers(self, url: str) -> bool:
        try:
            with self._opener(url, timeout=self._timeout) as response:
                return 200 <= getattr(response, "status", 200) < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def healthy(self, service: Service) -> bool:
        return bool(service.health) and self.answers(service.health)

    def pid(self, service: Service) -> int | None:
        """The pid ``up`` recorded, if that process is still alive."""
        path = self._pid_file(service)
        if not path.is_file():
            return None
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError):
            return None
        return pid

    def status(self, service: Service) -> ServiceStatus:
        return ServiceStatus(service, healthy=self.healthy(service), pid=self.pid(service))

    def up(self, service: Service, *, wait_seconds: float = 0.0) -> ServiceStatus:
        """Start the service unless it is already up; optionally wait for its health."""
        status = self.status(service)
        if status.up:
            return status
        self.root.mkdir(parents=True, exist_ok=True)
        with self.log_file(service).open("ab") as log:
            process = self._popen(
                shlex.split(service.command),
                cwd=BASE_DIR,
                env={**os.environ, **service.env},
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self._pid_file(service).write_text(str(process.pid), encoding="utf-8")
        deadline = time.monotonic() + wait_seconds
        while service.health and not self.healthy(service) and time.monotonic() < deadline:
            time.sleep(1.0)
        return self.status(service)

    def down(self, service: Service) -> bool:
        """Stop the process ``up`` started; False when none is running."""
        pid = self.pid(service)
        if pid is None:
            return False
        os.kill(pid, signal.SIGTERM)
        self._pid_file(service).unlink(missing_ok=True)
        return True


def requirements_of(cell: Cell, requires: Iterable[str]) -> list[str]:
    """The spec's ``requires`` for one cell, ``{lake}`` and ``{mode}`` filled in."""
    values = {"lake": cell.lake, "mode": cell.config.retrieval.mode.value}
    return [entry.format(**values) for entry in requires]


def missing_requirements(
    cells: Iterable[Cell], requires: Iterable[str], services: Mapping[str, Service], manager: ServiceManager,
) -> list[str]:
    """One line per required service that is down, for the cells that need it.

    A Pneuma requirement counts only for the cells of a Pneuma mode and is
    probed at the cell's own ``retrieval.pneuma_base_url``, the URL the run
    will query; when that is not its service's URL, the line says so.
    """
    requires = list(requires)
    problems: dict[str, str] = {}
    for cell in cells:
        for requirement in requirements_of(cell, requires):
            if requirement in problems:
                continue
            if requirement.startswith("pneuma:"):
                if not RetrievalMode(cell.config.retrieval.mode).is_pneuma:
                    continue
                url = cell.config.retrieval.pneuma_base_url.rstrip("/")
                service = services.get(requirement)
                if manager.answers(f"{url}/health"):
                    continue
                hint = f"start it with: services up {requirement}" if service else "no such service in config/services.yaml"
                if service and service.base_url and service.base_url.rstrip("/") != url:
                    hint += f" (it listens on {service.base_url}, the cell queries {url})"
                problems[requirement] = f"{requirement}: {url} does not answer; {hint}"
                continue
            service = service_for(requirement, services)
            if not manager.status(service).up:
                problems[requirement] = f"{requirement}: down; start it with: services up {requirement}"
    return [problems[key] for key in sorted(problems)]
