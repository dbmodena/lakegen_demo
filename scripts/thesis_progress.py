#!/usr/bin/env python3
"""Print the progress of the thesis suite: one row per submitted batch.

Reads the job ids that scripts/run_thesis_suite.py saved and the per-question
results the API writes, so it works while the suite runs. Use --watch to
refresh on a timer, e.g. in its own tmux window.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path
import time


def rows(state_path: Path, jobs_dir: Path) -> list[dict]:
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    out = []
    for key, job_id in state.items():
        job_path = jobs_dir / f"{job_id}.json"
        if not job_path.is_file():
            continue
        try:
            job = json.loads(job_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        results = []
        results_path = jobs_dir / f"{job_id}.results.jsonl"
        if results_path.is_file():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                try:
                    results.append(json.loads(line)["result"])
                except (ValueError, KeyError):
                    pass
        elapsed = [float(r.get("elapsed_seconds") or 0) for r in results]
        out.append({
            "key": key,
            "status": job.get("status"),
            "done": len(results),
            "total": int(job.get("question_count") or 0),
            "mean": sum(elapsed) / len(elapsed) if elapsed else None,
            "fast": sum(value < 10 for value in elapsed[-10:]),
            "statuses": Counter(r.get("status") for r in results),
        })
    return out


def render(state_path: Path, jobs_dir: Path, parallel: int, planned: int) -> str:
    data = rows(state_path, jobs_dir)
    lines = [f"Suite tesi — {datetime.now():%Y-%m-%d %H:%M:%S}", ""]
    header = f"{'batch':<38} {'stato':<10} {'fatte':>9} {'%':>5} {'s/dom':>6}  esiti"
    lines += [header, "-" * len(header)]
    for row in sorted(data, key=lambda r: (r["status"] != "running", r["key"])):
        pct = 100 * row["done"] / row["total"] if row["total"] else 0
        mean = f"{row['mean']:.0f}" if row["mean"] is not None else "—"
        outcome = ", ".join(f"{k}:{v}" for k, v in row["statuses"].most_common())
        alarm = "  ⚠ risposte in <10s" if row["fast"] >= 6 else ""
        lines.append(
            f"{row['key']:<38} {str(row['status']):<10} {row['done']:>4}/{row['total']:<4} "
            f"{pct:>4.0f}% {mean:>6}  {outcome}{alarm}"
        )
    done = sum(r["done"] for r in data)
    means = [r["mean"] for r in data if r["mean"] is not None]
    completed = sum(r["status"] == "completed" for r in data)
    lines += ["", f"Batch completati: {completed} / {planned}   Domande fatte: {done}"]
    running = [r for r in data if r["status"] in {"running", "queued"}]
    if means and running:
        per_question = sum(means) / len(means)
        remaining = sum(r["total"] - r["done"] for r in running)
        eta = timedelta(seconds=int(remaining * per_question / max(1, len(running))))
        lines.append(
            f"Batch in corso: {len(running)}; tempo medio {per_question:.0f}s/domanda; "
            f"fine dei batch in corso tra circa {eta}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path("reports/thesis/suite_state.json"))
    parser.add_argument("--jobs-dir", type=Path, default=Path(".lakegen_jobs"))
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--planned", type=int, default=32, help="Batch totali della suite")
    parser.add_argument("--watch", type=int, metavar="SECONDI",
                        help="Aggiorna ogni N secondi")
    args = parser.parse_args()
    while True:
        text = render(args.state, args.jobs_dir, args.parallel, args.planned)
        if args.watch:
            print("\033[2J\033[H" + text, flush=True)
            time.sleep(args.watch)
        else:
            print(text)
            break


if __name__ == "__main__":
    main()
