"""Polars cell scans, run only in child processes.

Scanning a lake runs thousands of small polars queries, and polars can crash
natively under that load: on 2026-09-14 the kernel logged ``polars-3: segfault
... in _polars_runtime.abi3.so`` during a full-lake content scan, and the
Chainlit app died with it. A segfault cannot be caught, so every session in
the process went down. The crash is intermittent -- twice in roughly a hundred
multi-threaded full-lake scans -- and no polars release after 1.40.1 lists a
matching fix.

The defence is a process boundary. Every polars call in this module runs in a
worker started as ``python -m lakegen.retrieval.pneuma.cell_scan``, and the process that calls
:func:`run` never runs a polars query (this module never imports polars there;
the application may still import it through dependencies such as ``valentine``,
which is harmless -- the crash needs queries, not an import). A worker that dies
costs the batch it was holding, which is retried, instead of costing the
application.

Workers are plain subprocesses rather than :mod:`multiprocessing` ones.
``spawn`` re-executes the parent's ``__main__`` in each worker, which would load
Chainlit or pytest into every scanner and fails outright for a script read from
stdin. And each worker holds exactly one batch at a time, so a crash is
attributable to that batch; a process pool fails every pending task at once.
"""

from __future__ import annotations

import importlib
import logging
import os

# The frames carry tasks and results between this module and the workers it
# starts itself (``_serve``), over their private pipes: nothing else writes them.
import pickle  # nosec B403
import struct
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

_HEADER = struct.Struct(">Q")
# Tasks per round trip: small enough that a crash loses little work and slow
# files spread across workers, large enough that pickling is not the cost.
_BATCH_SIZE = 8
# A task that takes its worker down this many times on its own is skipped.
_MAX_CRASHES_PER_TASK = 2

# Casting these to String raises rather than yielding text, so they are skipped.
# Struct is deliberately absent: it renders as ``{1,"ferry"}`` and is searchable.
_UNCASTABLE = ("List", "Array", "Binary", "Object", "Null")

# Rows per read of one column. A worker holds one such chunk at a time, which is
# what bounds its memory when a scan reads every row of a multi-GB table.
_SCAN_CHUNK_ROWS = 1_000_000


class ScanWorkersFailed(RuntimeError):
    """Workers kept dying, so the scan stopped instead of retrying without end."""


def _write_frame(stream: BinaryIO, payload: bytes) -> None:
    """Write ``payload`` with its length in front, and flush."""
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_frame(stream: BinaryIO) -> bytes | None:
    """The next length-prefixed payload, or None when the stream ends before it does."""
    header = stream.read(_HEADER.size)
    if len(header) < _HEADER.size:
        return None
    (size,) = _HEADER.unpack(header)
    payload = stream.read(size)
    return payload if len(payload) == size else None


# ------------------------------------------------------------------ child side
#
# Polars is imported inside each function, so importing this module -- as the
# parent does to call ``run`` -- never loads it.


def _searchable(path: str) -> tuple[list[str], int] | None:
    """Column names that can be cast to text, and how many cannot."""
    import polars as pl

    try:
        schema = pl.scan_parquet(path).collect_schema()
    except Exception:
        logger.debug("cannot read the schema of %s; not scanned", path, exc_info=True)
        return None
    uncastable = tuple(getattr(pl, name) for name in _UNCASTABLE)
    names = [name for name in schema.names() if schema[name].base_type() not in uncastable]
    return names, len(schema.names()) - len(names)


def scan_columns(
    path: str,
    patterns: Sequence[tuple[str, str]],
    max_columns: int | None,
    max_rows: int | None,
) -> tuple[dict[str, dict[str, int]], tuple[str, ...], int]:
    """Count matches per term per column, one column in flight at a time.

    ``patterns`` pairs each term with its regex, escaped by the caller, so
    nothing an agent supplies reaches polars as syntax.

    A ``max_columns`` or ``max_rows`` of ``None`` reads every column or every
    row. Rows are read ``_SCAN_CHUNK_ROWS`` at a time either way, so however
    long the table, a worker holds one chunk of one column.

    Args:
        path: The Parquet file.
        patterns: Each term with its regex.
        max_columns: How many castable columns to scan, or None for all.
        max_rows: How many rows to read, or None for all.

    Returns:
        ``(term -> column -> matching cells, scanned columns, uncastable count)``;
        empty for a file whose schema cannot be read.
    """
    import polars as pl

    counts: dict[str, dict[str, int]] = {}
    found = _searchable(path)
    if found is None:
        return counts, (), 0
    searchable, uncastable = found
    # Uncastable columns are counted before the cap, so the figure describes the
    # table rather than the budget.
    searchable = searchable[:max_columns]
    for name in searchable:
        expressions = [
            pl.col(name)
            .cast(pl.String, strict=False)
            .str.contains(pattern, literal=False)
            .fill_null(False)
            .sum()
            .alias(f"__t{index}")
            for index, (_term, pattern) in enumerate(patterns)
        ]
        expressions.append(pl.len().alias("__rows"))
        hits = [0] * len(patterns)
        offset = 0
        try:
            while max_rows is None or offset < max_rows:
                length = (
                    _SCAN_CHUNK_ROWS
                    if max_rows is None
                    else min(_SCAN_CHUNK_ROWS, max_rows - offset)
                )
                matches = (
                    pl.scan_parquet(path)
                    .select(pl.col(name))
                    .slice(offset, length)
                    .select(expressions)
                    .collect()
                )
                for index in range(len(patterns)):
                    hits[index] += int(matches[f"__t{index}"].item() or 0)
                rows = int(matches["__rows"].item())
                offset += rows
                if rows < length:
                    break
        except Exception:
            # A single unreadable or uncastable column must not lose the file,
            # and a column that fails part-way counts for nothing rather than
            # for whichever prefix happened to be read.
            logger.debug("cannot scan column %r of %s; skipped", name, path, exc_info=True)
            continue
        for index, (term, _pattern) in enumerate(patterns):
            if hits[index]:
                counts.setdefault(term, {})[name] = hits[index]
    return counts, tuple(searchable), uncastable


def _text_columns(path: str) -> tuple[list[str], int] | None:
    """String column names, and how many columns are not String."""
    import polars as pl

    try:
        schema = pl.scan_parquet(path).collect_schema()
    except Exception:
        logger.debug("cannot read the schema of %s; not scanned", path, exc_info=True)
        return None
    names = [name for name in schema.names() if schema[name] == pl.String]
    return names, len(schema.names()) - len(names)


def _occurrences(
    path: str,
    columns: Sequence[str],
    patterns: Sequence[tuple[str, str]],
    max_rows: int | None,
) -> list[int]:
    """Occurrences per (column, term), in that order, over the row budget.

    Args:
        path: The Parquet file.
        columns: The String columns to count in, all in one query per row chunk.
        patterns: Each term with its regex.
        max_rows: How many rows to read, or None for all.

    Returns:
        One total per column and term, the terms of a column together.
    """
    import polars as pl

    expressions = [
        pl.col(name)
        .str.count_matches(pattern)
        .fill_null(0)
        .sum()
        .alias(f"__{column_index}_{term_index}")
        for column_index, name in enumerate(columns)
        for term_index, (_term, pattern) in enumerate(patterns)
    ]
    expressions.append(pl.len().alias("__rows"))
    totals = [0] * (len(columns) * len(patterns))
    offset = 0
    while max_rows is None or offset < max_rows:
        length = _SCAN_CHUNK_ROWS if max_rows is None else min(_SCAN_CHUNK_ROWS, max_rows - offset)
        row = (
            pl.scan_parquet(path)
            .select([pl.col(name) for name in columns])
            .slice(offset, length)
            .select(expressions)
            .collect()
            .row(0)
        )
        for index, value in enumerate(row[:-1]):
            totals[index] += int(value or 0)
        rows = int(row[-1])
        offset += rows
        if rows < length:
            break
    return totals


def scan_text(
    path: str,
    patterns: Sequence[tuple[str, str]],
    max_columns: int | None,
    max_rows: int | None,
) -> tuple[dict[str, dict[str, int]], tuple[str, ...], int]:
    """Count regex occurrences per term per String column, one query per file.

    Pneuma-Seeker's reference content search: only VARCHAR/TEXT columns are
    read, and every occurrence counts (``SUM(ARRAY_LENGTH(REGEXP_EXTRACT_ALL(..)))``),
    so a cell naming a term twice counts twice.

    All text columns go into one query per row chunk. Querying per column
    re-parses the footer for every column, which on a 16,185-column UK table
    cost minutes; one query per file scanned the whole UK lake in seconds. A
    file whose combined query fails is retried one column at a time, so one bad
    column does not lose the others.

    Args:
        path: The Parquet file.
        patterns: Each term with its regex, escaped by the caller.
        max_columns: How many String columns to scan, or None for all.
        max_rows: How many rows to read, or None for all.

    Returns:
        ``(term -> column -> occurrences, scanned columns, columns that are not
        text)``; empty for a file whose schema cannot be read.
    """
    counts: dict[str, dict[str, int]] = {}
    found = _text_columns(path)
    if found is None:
        return counts, (), 0
    names, other = found
    names = names[:max_columns]
    if not names:
        return counts, (), other
    try:
        per_column = _occurrences(path, names, patterns, max_rows)
        totals = {
            name: per_column[index * len(patterns) : (index + 1) * len(patterns)]
            for index, name in enumerate(names)
        }
    except Exception:
        logger.debug("combined text scan of %s failed; scanning per column", path, exc_info=True)
        totals = _occurrences_per_column(path, names, patterns, max_rows)
    for name, values in totals.items():
        for (term, _pattern), value in zip(patterns, values, strict=True):
            if value:
                counts.setdefault(term, {})[name] = value
    return counts, tuple(names), other


def _occurrences_per_column(
    path: str,
    names: Sequence[str],
    patterns: Sequence[tuple[str, str]],
    max_rows: int | None,
) -> dict[str, list[int]]:
    """The fallback of ``scan_text``: one query per column, skipping the ones that fail.

    Args:
        path: The Parquet file.
        names: The String columns.
        patterns: Each term with its regex.
        max_rows: How many rows to read, or None for all.

    Returns:
        The occurrences per term of each column that could be scanned.
    """
    totals: dict[str, list[int]] = {}
    for name in names:
        try:
            totals[name] = _occurrences(path, [name], patterns, max_rows)
        except Exception:
            # Like the per-column scan: a column that fails part-way counts
            # for nothing rather than for whichever prefix was read.
            logger.debug("cannot scan column %r of %s; skipped", name, path, exc_info=True)
            continue
    return totals


def probe_count(
    path: str,
    patterns: Sequence[tuple[str, str]],
    max_columns: int | None,
    max_rows: int,
) -> int:
    """Count matching rows in a short prefix, as a cheap stand-in for a scan.

    Args:
        path: The Parquet file.
        patterns: Each term with its regex.
        max_columns: How many castable columns to test, or None for all.
        max_rows: The length of the prefix.

    Returns:
        The rows of the prefix where any column matches any pattern; 0 when the
        file cannot be read.
    """
    import polars as pl

    found = _searchable(path)
    if found is None:
        return 0
    columns = found[0][:max_columns]
    if not columns:
        return 0
    predicate = pl.any_horizontal(
        [
            pl.col(name)
            .cast(pl.String, strict=False)
            .str.contains(pattern, literal=False)
            .fill_null(False)
            for name in columns
            for _term, pattern in patterns
        ]
    )
    try:
        # The row cap keeps this bounded however wide the table is.
        return int(
            pl.scan_parquet(path)
            .select([pl.col(name) for name in columns])
            .head(max_rows)
            .select(predicate.sum().alias("__hits"))
            .collect()["__hits"]
            .item()
            or 0
        )
    except Exception:
        logger.debug("cannot probe %s; counted as no match", path, exc_info=True)
        return 0


def sample_rows(
    path: str,
    columns: Sequence[str],
    patterns: Sequence[tuple[str, str]],
    max_rows: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Read a few matching rows from columns already known to match.

    Values are reduced to plain Python here, so nothing polars-shaped crosses
    back into the calling process.

    Args:
        path: The Parquet file.
        columns: The columns to read and match in.
        patterns: Each term with its regex.
        max_rows: The length of the prefix searched.
        limit: How many matching rows to return.

    Returns:
        The matching rows as plain dicts; none when the file cannot be read.
    """
    import polars as pl

    predicate = pl.any_horizontal(
        [
            pl.col(name)
            .cast(pl.String, strict=False)
            .str.contains(pattern, literal=False)
            .fill_null(False)
            for name in columns
            for _term, pattern in patterns
        ]
    )
    try:
        frame = (
            pl.scan_parquet(path)
            .select([pl.col(name) for name in columns])
            .head(max_rows)
            .filter(predicate)
            .head(limit)
            .collect()
        )
    except Exception:
        logger.debug("cannot sample %s; no rows", path, exc_info=True)
        return []
    return [
        {
            name: value
            if value is None or isinstance(value, (str, int, float, bool))
            else str(value)
            for name, value in row.items()
        }
        for row in frame.to_dicts()
    ]


_OPERATIONS: dict[str, Callable[..., Any]] = {
    "scan": scan_columns,
    "scan_text": scan_text,
    "probe": probe_count,
    "sample": sample_rows,
}


def _resolve(operation: str) -> Callable[..., Any]:
    """A built-in operation, or any importable ``module:function``."""
    if operation in _OPERATIONS:
        return _OPERATIONS[operation]
    module, _, name = operation.partition(":")
    return getattr(importlib.import_module(module), name)


def _serve() -> None:
    """Worker loop: one pickled batch in, one pickled list of results out."""
    stdin = sys.stdin.buffer
    # Results get a private copy of the pipe, and fd 1 is pointed at stderr, so
    # nothing a library prints -- from Python or native code -- can corrupt the
    # result stream.
    results = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    while (frame := _read_frame(stdin)) is not None:
        # Written by ``run`` in the process that started this worker.
        operation, tasks = pickle.loads(frame)  # nosec B301
        function = _resolve(operation)
        _write_frame(
            results,
            pickle.dumps([function(*task) for task in tasks], pickle.HIGHEST_PROTOCOL),
        )


# ----------------------------------------------------------------- parent side


def _start_worker(workers: int = 1) -> subprocess.Popen[bytes]:
    """Start one worker process, sharing the cores among ``workers`` of them."""
    root = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ)
    # Make ``lakegen`` importable in the child however the parent found it.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    # Left alone, every worker starts one polars thread per core, so 16 workers
    # on 32 cores ran 512 threads. The cores are shared instead: 16 workers x 2
    # threads was the fastest setting measured on the UK and NYC lakes.
    env.setdefault("POLARS_MAX_THREADS", str(max(1, (os.cpu_count() or 1) // max(1, workers))))
    return subprocess.Popen(
        # faulthandler turns a native crash into a traceback on stderr, which is
        # the only diagnostic a segfault leaves behind.
        [sys.executable, "-X", "faulthandler", "-m", "lakegen.retrieval.pneuma.cell_scan"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )


def _stop_worker(process: subprocess.Popen[bytes]) -> None:
    """End the worker's input and wait for it, killing it if that fails or takes too long."""
    try:
        if process.stdin is not None:
            process.stdin.close()  # end of input ends the worker loop
        process.wait(timeout=10)
    except Exception:
        logger.debug("cell scan worker %s did not stop; killing it", process.pid, exc_info=True)
        process.kill()
        process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()


def _requeue(batch: list[int], crashes: Sequence[int], pending: deque[list[int]]) -> str:
    """Put the tasks of a batch whose worker died back on the queue.

    A batch of several tasks comes back one task per batch; a lone task comes
    back until it has crashed its worker ``_MAX_CRASHES_PER_TASK`` times.

    Args:
        batch: The task indices of the batch.
        crashes: How many times each task crashed its worker alone.
        pending: The queue of batches.

    Returns:
        What was done, for the log.
    """
    if len(batch) > 1:
        pending.extend([index] for index in batch)
        return "retrying its tasks one at a time"
    if crashes[batch[0]] < _MAX_CRASHES_PER_TASK:
        pending.append(batch)
        return "retrying"
    return "skipping the task, it crashed its worker twice on its own"


def _work(
    operation: str,
    tasks: Sequence[tuple[Any, ...]],
    workers: int,
    next_batch: Callable[[], list[int] | None],
    settle: Callable[[list[int], list[Any] | None, int | None], None],
) -> None:
    """One thread of ``run``: send batches to a worker process of its own until none are left.

    The worker is started on the first batch and again after one dies.

    Args:
        operation: The operation the workers run.
        tasks: All the tasks of the run.
        workers: How many threads share the cores.
        next_batch: Takes the next batch, or None when the run is over.
        settle: Records a batch's results, or its worker's exit code when it died.

    Raises:
        RuntimeError: When a worker was started without its pipes.
    """
    process: subprocess.Popen[bytes] | None = None
    try:
        while (batch := next_batch()) is not None:
            if process is None:
                process = _start_worker(workers)
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("cell scan worker started without its stdin and stdout pipes")
            frame: bytes | None
            try:
                _write_frame(
                    process.stdin,
                    pickle.dumps(
                        (operation, [tasks[index] for index in batch]),
                        pickle.HIGHEST_PROTOCOL,
                    ),
                )
                frame = _read_frame(process.stdout)
            except OSError:
                frame = None
            if frame is not None:
                # Written by ``_serve`` in the worker this thread started.
                settle(batch, pickle.loads(frame), None)  # nosec B301
                continue
            exit_code = process.wait()
            _stop_worker(process)
            process = None
            settle(batch, None, exit_code)
    finally:
        if process is not None:
            _stop_worker(process)


def _run_threads(count: int, target: Callable[..., None], *args: Any) -> None:
    """Run ``target(*args)`` on ``count`` daemon threads and wait for them all."""
    threads = [
        threading.Thread(target=target, args=args, name=f"cell-scan-{index}", daemon=True)
        for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def run(operation: str, tasks: Sequence[tuple[Any, ...]], *, workers: int) -> list[Any]:
    """Run ``operation`` over ``tasks`` in worker processes, keeping task order.

    A worker that dies has its batch re-queued one task per batch, so a file
    that really does break the scanner is isolated and stops costing its
    neighbours, while an intermittent crash simply succeeds on the retry. A task
    that crashes its worker ``_MAX_CRASHES_PER_TASK`` times alone gets ``None``
    as its result. If crashes pile up past a small budget the environment is
    broken rather than unlucky, and :class:`ScanWorkersFailed` is raised -- an
    ordinary exception the application can report and survive.

    Args:
        operation: A built-in operation (``scan``, ``scan_text``, ``probe``,
            ``sample``) or an importable ``module:function``.
        tasks: The argument tuples, one call each.
        workers: How many worker processes to run at most.

    Returns:
        The result of each task, in task order; None for a skipped task.

    Raises:
        ScanWorkersFailed: When the workers crash more often than the budget allows.
    """
    if not tasks:
        return []
    results: list[Any] = [None] * len(tasks)
    crashes = [0] * len(tasks)
    pending: deque[list[int]] = deque(
        list(range(start, min(start + _BATCH_SIZE, len(tasks))))
        for start in range(0, len(tasks), _BATCH_SIZE)
    )
    condition = threading.Condition()
    budget = max(8, workers)
    state: dict[str, Any] = {"in_flight": 0, "crashes": 0, "error": None}

    def next_batch() -> list[int] | None:
        with condition:
            # Wait rather than leave while other workers still hold batches: a
            # crash can put work back on the queue after it looked empty.
            while not pending and state["in_flight"] and state["error"] is None:
                condition.wait()
            if state["error"] is not None or not pending:
                return None
            state["in_flight"] += 1
            return pending.popleft()

    def settle(batch: list[int], outcome: list[Any] | None, exit_code: int | None) -> None:
        with condition:
            state["in_flight"] -= 1
            if outcome is not None:
                # The worker answers a batch with one result per task.
                for index, value in zip(batch, outcome, strict=True):
                    results[index] = value
            else:
                state["crashes"] += 1
                if len(batch) == 1:
                    crashes[batch[0]] += 1
                if state["crashes"] > budget:
                    state["error"] = ScanWorkersFailed(
                        f"cell scan workers crashed {state['crashes']} times in one "
                        f"'{operation}' run; stopping"
                    )
                    action = "stopping"
                else:
                    action = _requeue(batch, crashes, pending)
                logger.warning(
                    "[cell_scan] worker exited with %s during a batch of %d task(s); %s",
                    exit_code,
                    len(batch),
                    action,
                )
            condition.notify_all()

    _run_threads(
        max(1, min(workers, len(pending))), _work, operation, tasks, workers, next_batch, settle
    )
    if state["error"] is not None:
        raise state["error"]
    return results


if __name__ == "__main__":
    _serve()
