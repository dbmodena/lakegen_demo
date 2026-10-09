#!/usr/bin/env python3
"""Split one JSON question collection and submit its chunks concurrently.

The API remains the owner of execution, persistence, retries and results. This
tool only creates independent API batch jobs, so an interrupted client does not
lose already accepted chunks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _split_payload(payload: Any, chunk_size: int) -> list[Any]:
    """Return API-compatible chunks, preserving an enclosing inline config."""
    if isinstance(payload, list):
        container: dict[str, Any] | None = None
        key = None
        questions = payload
    elif isinstance(payload, dict):
        key = next((candidate for candidate in ("questions", "queries", "cases")
                    if isinstance(payload.get(candidate), list)), None)
        if key is None:
            raise ValueError("expected a JSON list or an object with questions, queries, or cases")
        container = payload
        questions = payload[key]
    else:
        raise ValueError("question file must contain a JSON list or object")
    if not questions:
        raise ValueError("question collection is empty")
    chunks: list[Any] = []
    for start in range(0, len(questions), chunk_size):
        part = questions[start:start + chunk_size]
        chunks.append(part if container is None else {**container, key: part})
    return chunks


def _post_batch(api_url: str, payload: Any) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/v1/batches",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc


def _wait_for_jobs(api_url: str, jobs: list[dict[str, Any]]) -> bool:
    pending = {job["job_id"] for job in jobs}
    failed = False
    while pending:
        time.sleep(2)
        for job_id in list(pending):
            try:
                with urllib.request.urlopen(
                    f"{api_url.rstrip('/')}/v1/batches/{job_id}?include_results=false", timeout=30,
                ) as response:
                    job_status = json.loads(response.read().decode("utf-8"))["status"]
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                print(f"{job_id}: status check failed: {exc}", file=sys.stderr)
                continue
            if job_status in {"completed", "failed"}:
                print(f"{job_id}: {job_status}")
                pending.remove(job_id)
                failed |= job_status == "failed"
    return not failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, help="JSON questions/queries/cases file")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--parallelism", type=int, default=2,
                        help="maximum simultaneous submission requests")
    parser.add_argument("--wait", action="store_true", help="wait for every submitted job")
    args = parser.parse_args()
    if args.chunk_size < 1 or args.parallelism < 1:
        parser.error("--chunk-size and --parallelism must be positive")
    try:
        chunks = _split_payload(json.loads(args.questions.read_text(encoding="utf-8")), args.chunk_size)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    jobs: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [executor.submit(_post_batch, args.api_url, chunk) for chunk in chunks]
        for number, future in enumerate(futures, start=1):
            try:
                job = future.result()
            except Exception as exc:
                print(f"chunk {number}/{len(chunks)} was not accepted: {exc}", file=sys.stderr)
                return 1
            jobs.append(job)
            print(f"chunk {number}/{len(chunks)}: {job['job_id']} ({job['question_count']} questions)")
    return 0 if not args.wait or _wait_for_jobs(args.api_url, jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
