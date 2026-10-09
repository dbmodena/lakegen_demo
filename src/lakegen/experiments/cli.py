"""Plan, run and follow experiment specs (``lakegen exp``, ``lakegen runs``, ``lakegen services``).

    lakegen exp plan   SPEC            # the cells, and which already ran
    lakegen exp run    SPEC [--count N] [--dry-run] [--only TEXT]
    lakegen exp status SPEC
    lakegen runs index-legacy JOB_DIR... [--spec SPEC...]
    lakegen services status|up|down [NAME...] [--wait SECONDS]

``run`` is idempotent: a cell already in the run store's index is skipped, so a
crashed or stopped submission can be restarted. Cells are submitted
``execution.stagger_seconds`` apart, and not at all while a service the spec
``requires`` is down. ``index-legacy`` records the jobs the old submit scripts
created, under the spec whose cells they match (every spec in
experiments/specs by default) or under ``legacy-<directory>``. ``services``
probes, starts and stops the services of config/services.yaml.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from lakegen.config.locations import BASE_DIR
from lakegen.experiments.executors import ExecutionError, Executor, make_executor
from lakegen.experiments.legacy import index_legacy
from lakegen.experiments.questions import select_cases
from lakegen.experiments.services import (
    ServiceManager,
    load_services,
    missing_requirements,
    service_for,
)
from lakegen.experiments.spec import ExperimentSpec, expand
from lakegen.experiments.store import RunStore


def plan(spec: ExperimentSpec, store: RunStore) -> int:
    cells = expand(spec)
    submitted = store.submitted(spec.name)
    print(f"{spec.name}: {len(cells)} cells, {len(submitted)} already submitted")
    for cell in cells:
        marker = "done" if cell.cell_id in submitted else "    "
        print(f"  {marker} {cell.cell_id} {cell.describe()}")
    if spec.requires:
        print("requires:", ", ".join(spec.requires))
    return 0


def run(
    spec: ExperimentSpec,
    store: RunStore,
    executor: Executor,
    *,
    count: int | None,
    only: str | None,
    dry_run: bool,
    sleep=time.sleep,
    services: ServiceManager | None = None,
) -> int:
    cells = [cell for cell in expand(spec) if only is None or only in cell.experiment_id]
    submitted = store.submitted(spec.name)
    pending = [cell for cell in cells if cell.cell_id not in submitted]
    if pending and not dry_run and spec.requires:
        missing = missing_requirements(pending, spec.requires, load_services(), services or ServiceManager())
        if missing:
            print("not submitting: a required service is down")
            for line in missing:
                print(f"  {line}")
            return 1
    store.write_spec(spec)
    first = True
    for cell in cells:
        if cell.cell_id in submitted:
            print(f"skip   {cell.describe()}  (already submitted)")
            continue
        cases = select_cases(cell.questions_file, cell.questions_slice)
        if count is not None:
            cases = cases[:count]
        if dry_run:
            print(f"would  {cell.describe()}  {len(cases)} cases")
            continue
        if not first and spec.execution.stagger_seconds:
            sleep(spec.execution.stagger_seconds)
        first = False
        try:
            job_id = executor.submit(cell, cases)
        except ExecutionError as exc:
            print(f"error  {cell.describe()}: {exc}")
            return 1
        store.record(cell, backend=executor.backend, job_dir=executor.job_dir(), job_id=job_id)
        print(f"sent   {cell.describe()}  {len(cases)} cases -> {job_id}")
    return 0


def status(spec: ExperimentSpec, store: RunStore, executor: Executor) -> int:
    index = store.load_index(spec.name)
    if not index:
        print(f"{spec.name}: nothing submitted")
        return 0
    for record in index.values():
        try:
            job = executor.job(record.job_id) if record.job_id else {}
        except ExecutionError as exc:
            job = {"status": f"unreachable ({exc})"}
        print(
            f"  {record.experiment_id:40} {job.get('status', '?'):10} "
            f"{job.get('processed', '?')}/{job.get('question_count', '?')} failed={job.get('failed', '?')}"
        )
    return 0


def index_legacy_jobs(store: RunStore, job_dirs: Sequence[Path], spec_paths: Sequence[Path]) -> int:
    specs = [ExperimentSpec.load(path) for path in spec_paths]
    indexed = index_legacy(store, job_dirs, specs)
    for job_dir in job_dirs:
        mine = [entry for entry in indexed if entry.job.job_dir == Path(job_dir)]
        matched = [entry for entry in mine if entry.matched]
        specs_hit = sorted({entry.experiment for entry in matched})
        print(
            f"{job_dir}: {len(mine)} jobs, {len(matched)} matched "
            f"{', '.join(specs_hit) or 'no spec'}, {len(mine) - len(matched)} under legacy"
        )
    return 0


def services_command(action: str, names: Sequence[str], manager: ServiceManager, *, wait_seconds: float = 0.0) -> int:
    services = load_services()
    chosen = [service_for(name, services) for name in names] if names else list(services.values())
    failed = 0
    for service in chosen:
        if action == "up":
            status = manager.up(service, wait_seconds=wait_seconds)
            state = "up" if status.up else "started, not healthy yet" if status.pid else "failed to start"
            failed += not status.up
            print(f"{service.name:20} {state}  log: {manager.log_file(service)}")
        elif action == "down":
            print(f"{service.name:20} {'stopped' if manager.down(service) else 'not started by services up'}")
        else:
            status = manager.status(service)
            state = "up" if status.up else "down"
            detail = f"pid {status.pid}" if status.pid else ""
            print(f"{service.name:20} {state:5} {service.health or '(no health url)'}  {detail}")
    return 1 if failed else 0


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lakegen.experiments.cli", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    services = commands.add_parser("services")
    services.add_argument("action", choices=("status", "up", "down"))
    services.add_argument("names", nargs="*", metavar="NAME")
    services.add_argument("--wait", type=float, default=0.0, help="seconds to wait for health after up")
    for name in ("plan", "run", "status"):
        command = commands.add_parser(name)
        command.add_argument("spec", type=Path)
        command.add_argument("--store", type=Path, default=None, help="run store root (default: <state root>/runs)")
        if name == "run":
            command.add_argument("--count", type=int, default=None, help="cases per cell (default: the whole slice)")
            command.add_argument("--only", default=None, help="run only cells whose experiment id contains this")
            command.add_argument("--dry-run", action="store_true")
    legacy = commands.add_parser("index-legacy")
    legacy.add_argument("job_dirs", type=Path, nargs="+", metavar="JOB_DIR")
    legacy.add_argument("--spec", type=Path, action="append", default=None, help="specs to match (default: experiments/specs/*.yaml)")
    legacy.add_argument("--store", type=Path, default=None, help="run store root (default: <state root>/runs)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, services: ServiceManager | None = None) -> int:
    args = _parse(argv)
    if args.command == "services":
        return services_command(args.action, args.names, services or ServiceManager(), wait_seconds=args.wait)
    store = RunStore(args.store)
    if args.command == "index-legacy":
        specs = args.spec or sorted((BASE_DIR / "experiments" / "specs").glob("*.yaml"))
        return index_legacy_jobs(store, args.job_dirs, specs)
    spec = ExperimentSpec.load(args.spec)
    if args.command == "plan":
        return plan(spec, store)
    executor = make_executor(spec.execution.backend, api_url=spec.execution.api_url, job_dir=spec.execution.job_dir)
    if args.command == "run":
        if executor.backend == "api" and not executor.healthy():
            print(f"the API at {spec.execution.api_url} is not reachable")
            return 1
        return run(spec, store, executor, count=args.count, only=args.only, dry_run=args.dry_run, services=services)
    return status(spec, store, executor)


if __name__ == "__main__":
    sys.exit(main())
