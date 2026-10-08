"""Plan, run and follow experiment specs, until the ``lakegen`` command exists.

    python -m lakegen.experiments.cli plan   SPEC            # the cells, and which already ran
    python -m lakegen.experiments.cli run    SPEC [--count N] [--dry-run] [--only TEXT]
    python -m lakegen.experiments.cli status SPEC

``run`` is idempotent: a cell already in the run store's index is skipped, so a
crashed or stopped submission can be restarted. Cells are submitted
``execution.stagger_seconds`` apart.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from lakegen.experiments.executors import ExecutionError, Executor, make_executor
from lakegen.experiments.questions import select_cases
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


def run(spec: ExperimentSpec, store: RunStore, executor: Executor, *, count: int | None, only: str | None, dry_run: bool, sleep=time.sleep) -> int:
    cells = [cell for cell in expand(spec) if only is None or only in cell.experiment_id]
    submitted = store.submitted(spec.name)
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


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m lakegen.experiments.cli", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "status"):
        command = commands.add_parser(name)
        command.add_argument("spec", type=Path)
        command.add_argument("--store", type=Path, default=None, help="run store root (default: <state root>/runs)")
        if name == "run":
            command.add_argument("--count", type=int, default=None, help="cases per cell (default: the whole slice)")
            command.add_argument("--only", default=None, help="run only cells whose experiment id contains this")
            command.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    spec = ExperimentSpec.load(args.spec)
    store = RunStore(args.store)
    if args.command == "plan":
        return plan(spec, store)
    executor = make_executor(spec.execution.backend, api_url=spec.execution.api_url, job_dir=spec.execution.job_dir)
    if args.command == "run":
        if executor.backend == "api" and not executor.healthy():
            print(f"the API at {spec.execution.api_url} is not reachable")
            return 1
        return run(spec, store, executor, count=args.count, only=args.only, dry_run=args.dry_run)
    return status(spec, store, executor)


if __name__ == "__main__":
    sys.exit(main())
