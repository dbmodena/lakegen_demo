#!/usr/bin/env python3
"""Run the thesis configurations on the API in stages, a few batches at a time.

Every configuration runs its benchmark stage by stage (the stages come from
``sample_metadata.stages``, e.g. 100 then 400): all configurations finish
stage 1 before any stage-2 batch is submitted, so a comparable first result
exists early. Each stage is one API batch carrying the configuration inline,
under the configuration's own experiment_id, so the stages of one
configuration share its keyword memory and are merged by
analysis/compare_runs.py.

Submitted job ids are kept in a state file; rerunning the script waits on
jobs still in progress, skips completed ones and resubmits failed ones. The API
itself resumes interrupted jobs after a restart.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any
import urllib.error
import urllib.request

import yaml


POLL_SECONDS = 30


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def api(api_url: str, path: str, payload: Any | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}",
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def load_tasks(
    config_dir: Path, tool_access: str | None, stages: list[int] | None
) -> list[dict[str, Any]]:
    """One task per (configuration, stage), stage-major, Llama first in each stage."""
    tasks = []
    for path in sorted(config_dir.rglob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if tool_access and config.get("tool_access") != tool_access:
            continue
        benchmark_path = Path(config["benchmark"]["path"])
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        cases = benchmark["cases"]
        sizes = benchmark.get("sample_metadata", {}).get("stages") or [len(cases)]
        start = 0
        for number, size in enumerate(sizes, start=1):
            if stages is None or number in stages:
                tasks.append({
                    "key": f"{config['experiment_id']}#stage{number}",
                    "experiment_id": config["experiment_id"],
                    "stage": number,
                    "llama": "llama" in str(config.get("model")),
                    "payload": {
                        "config": config,
                        "sample_metadata": {
                            **benchmark.get("sample_metadata", {}),
                            "benchmark_path": str(benchmark_path),
                            "stage": number,
                            "stage_case_range": [start, start + size],
                        },
                        "cases": cases[start:start + size],
                    },
                })
            start += size
    return sorted(tasks, key=lambda task: (task["stage"], not task["llama"], task["key"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("config/thesis"))
    parser.add_argument("--tool-access", choices=("agentic", "orchestrated_context"),
                        help="Run only configurations with this tool access")
    parser.add_argument("--stages", type=int, nargs="+",
                        help="Run only these stages (1-based); default all")
    parser.add_argument("--parallel", type=int, default=4,
                        help="Batches running at the same time")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--state", type=Path, default=Path("reports/thesis/suite_state.json"))
    parser.add_argument("--dry-run", action="store_true", help="List the tasks and exit")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be positive")

    tasks = load_tasks(args.config_dir, args.tool_access, args.stages)
    if not tasks:
        parser.error("no configuration matches")
    if args.dry_run:
        for task in tasks:
            print(f"{task['key']}: {len(task['payload']['cases'])} domande")
        return 0

    state: dict[str, str] = (
        json.loads(args.state.read_text(encoding="utf-8")) if args.state.is_file() else {}
    )

    def save_state() -> None:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def status(job_id: str) -> dict[str, Any] | None:
        try:
            return api(args.api_url, f"/v1/batches/{job_id}?include_results=false")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    pending = list(tasks)
    running: dict[str, dict[str, Any]] = {}
    done: set[str] = set()
    failed: list[str] = []
    last_summary: tuple[Any, ...] = ()
    log(f"{len(tasks)} batch da eseguire, al massimo {args.parallel} insieme")

    while pending or running:
        # Poll running jobs.
        for key, task in list(running.items()):
            try:
                job = status(state[key])
            except (urllib.error.URLError, OSError) as exc:
                log(f"{key}: API non raggiungibile ({exc}); riprovo")
                continue
            if job is None or job["status"] == "failed":
                log(f"{key}: job {state[key]} fallito o sparito"
                    + (f" ({job.get('error')})" if job else ""))
                failed.append(key)
                del running[key]
            elif job["status"] == "completed":
                log(f"{key}: completato ({job['processed']}/{job['question_count']}, "
                    f"{job.get('failed', 0)} domande con errore)")
                done.add(key)
                del running[key]

        # Start new jobs while there is room. A stage waits for the earlier
        # stage of the same configuration, so they never share memory at once.
        for task in list(pending):
            if len(running) >= args.parallel:
                break
            key = task["key"]
            earlier = [
                other["key"] for other in tasks
                if other["experiment_id"] == task["experiment_id"]
                and other["stage"] < task["stage"]
            ]
            if any(other in failed for other in earlier):
                pending.remove(task)
                failed.append(key)
                log(f"{key}: saltato, lo stage precedente è fallito")
                continue
            if any(other not in done for other in earlier):
                continue
            existing = state.get(key)
            job = status(existing) if existing else None
            if job and job["status"] == "completed":
                log(f"{key}: già completato ({existing})")
                done.add(key)
                pending.remove(task)
                continue
            if job and job["status"] in {"queued", "running"}:
                log(f"{key}: riprendo il job in corso {existing}")
            else:
                try:
                    accepted = api(args.api_url, "/v1/batches", task["payload"])
                except (urllib.error.URLError, OSError) as exc:
                    log(f"{key}: invio non riuscito ({exc}); riprovo")
                    break
                state[key] = accepted["job_id"]
                save_state()
                log(f"{key}: inviato come {accepted['job_id']} "
                    f"({accepted['question_count']} domande)")
            running[key] = task
            pending.remove(task)

        if pending or running:
            summary = (tuple(sorted(running)), len(pending), len(done))
            if summary != last_summary:
                log(f"in corso: {', '.join(summary[0]) or '—'}; "
                    f"in coda: {summary[1]}; completati: {summary[2]}")
                last_summary = summary
            time.sleep(POLL_SECONDS)

    log(f"Fine: {len(done)} completati, {len(failed)} falliti")
    for key in failed:
        log(f"  fallito: {key}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
