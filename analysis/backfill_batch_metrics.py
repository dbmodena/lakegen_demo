#!/usr/bin/env python3
"""Rebuild table-selection metrics for a completed, historical batch job."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
for module_path in (ROOT_DIR, ROOT_DIR / "src"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from generate_report import generate
from api import _append_batch_table_metrics


def load_results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Historical batch job ID")
    parser.add_argument("--jobs-dir", type=Path, default=Path(".lakegen_jobs"))
    parser.add_argument(
        "--report-output", type=Path, default=None,
        help="Regenerate the HTML/CSV/Markdown report in this directory",
    )
    args = parser.parse_args()

    job_path = args.jobs_dir / f"{args.job_id}.json"
    questions_path = args.jobs_dir / f"{args.job_id}.questions.json"
    results_path = args.jobs_dir / f"{args.job_id}.results.jsonl"
    for path in (job_path, questions_path, results_path):
        if not path.is_file():
            parser.error(f"file richiesto non trovato: {path}")

    job = json.loads(job_path.read_text(encoding="utf-8"))
    if job.get("status") != "completed":
        parser.error("il job deve essere completed")
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    results = load_results(results_path)
    metrics = _append_batch_table_metrics(
        args.job_id, questions, results, job["settings"], append_log=False
    )
    if metrics is None:
        parser.error("il job non contiene metadata gold complete per il calcolo")

    existing = job.get("batch_metrics", {})
    job["batch_metrics"] = {**existing, "table_selection": metrics}
    job["metrics_logged"] = True
    temporary = job_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(job_path)

    if args.report_output:
        generate(job, args.report_output, questions=questions)
        print(f"Report rigenerato: {args.report_output}")
    print(f"Metriche retroattive salvate: {job_path}")


if __name__ == "__main__":
    main()
