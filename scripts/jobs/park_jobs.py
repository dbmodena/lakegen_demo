#!/usr/bin/env python3
r"""Park or un-park API batch jobs so a restarted API does not resume them.

The API resumes every job whose status is ``queued`` or ``running`` when it
starts (``experiment_runner.batches.batch.recover_incomplete_jobs``). While the code base is being
refactored, a job must not continue on code that differs from the code that
produced its earlier results, so unfinished jobs are given the status
``parked``, which the API ignores. The original status is kept in
``parked_from`` so the job can be restored unchanged.

    python scripts/jobs/park_jobs.py park /data/bilel/lakegen/jobs/lakegen_jobs_qwen \
        --backup-dir /data/bilel/lakegen/jobs/_backup_pre_refactor \
        --reason "refactor on tesi, 2026-10-08"
    python scripts/jobs/park_jobs.py unpark /data/bilel/lakegen/jobs/lakegen_jobs_qwen

Every job file is copied to the backup directory before it is changed, under
its job directory's name; an existing backup is never overwritten, so the
first backup is always the pre-park original. Only the ``<job_id>.json``
metadata files are touched: results and question files are left alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

RESUMABLE = {"queued", "running"}
PARKED = "parked"
PARK_KEYS = ("parked_from", "parked_at", "parked_reason")


def job_files(directory: Path) -> list[Path]:
    """The job metadata files of one API job directory."""
    return sorted(
        path
        for path in directory.glob("*.json")
        if not path.name.endswith(".questions.json")
        and len(path.stem) == 32
        and all(character in "0123456789abcdef" for character in path.stem)
    )


def write_job(path: Path, job: dict) -> None:
    """Replace a job file atomically, in the API's own format."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def back_up(path: Path, backup_dir: Path) -> Path:
    """Copy a job file to ``<backup_dir>/<job directory name>/`` unless a backup exists; the backup's path."""
    target = backup_dir / path.parent.name / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    return target


def park(directory: Path, backup_dir: Path, reason: str, dry_run: bool) -> int:
    """Park the directory's queued and running jobs, backing each up first.

    Args:
        directory: The API job directory.
        backup_dir: Where the job files are backed up.
        reason: Why, recorded as ``parked_reason``.
        dry_run: Count the jobs without changing them.

    Returns:
        The number of jobs parked (or that would be).
    """
    parked = 0
    for path in job_files(directory):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("status") not in RESUMABLE:
            continue
        parked += 1
        if dry_run:
            continue
        back_up(path, backup_dir)
        job.update(
            parked_from=job["status"],
            parked_at=datetime.now(UTC).isoformat(),
            parked_reason=reason,
            status=PARKED,
        )
        write_job(path, job)
    return parked


def unpark(directory: Path, dry_run: bool) -> int:
    """Restore the directory's parked jobs to the status they were parked from.

    Args:
        directory: The API job directory.
        dry_run: Count the jobs without changing them.

    Returns:
        The number of jobs restored (or that would be).
    """
    restored = 0
    for path in job_files(directory):
        job = json.loads(path.read_text(encoding="utf-8"))
        if job.get("status") != PARKED:
            continue
        restored += 1
        if dry_run:
            continue
        job["status"] = job.get("parked_from") or "queued"
        for key in PARK_KEYS:
            job.pop(key, None)
        write_job(path, job)
    return restored


def main(argv: list[str] | None = None) -> int:
    """Run the command line; print the count per directory and in total.

    Args:
        argv: The arguments; the process's when None.

    Returns:
        The exit code, 0.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    park_command = commands.add_parser("park", help="park every queued or running job")
    park_command.add_argument("directories", nargs="+", type=Path)
    park_command.add_argument("--backup-dir", type=Path, required=True)
    park_command.add_argument("--reason", required=True)
    park_command.add_argument("--dry-run", action="store_true")
    unpark_command = commands.add_parser("unpark", help="restore every parked job")
    unpark_command.add_argument("directories", nargs="+", type=Path)
    unpark_command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    total = 0
    for directory in args.directories:
        if not directory.is_dir():
            parser.error(f"not a directory: {directory}")
        if args.command == "park":
            count = park(directory, args.backup_dir, args.reason, args.dry_run)
        else:
            count = unpark(directory, args.dry_run)
        total += count
        print(f"{directory}: {count} {args.command}ed")
    verb = "would be" if args.dry_run else "were"
    print(f"{total} jobs {verb} {args.command}ed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
