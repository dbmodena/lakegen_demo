"""Polars cell scans, run only in child processes.

Scanning a lake runs thousands of small polars queries, and polars can crash
natively under that load: on 2026-09-14 the kernel logged ``polars-3: segfault
... in _polars_runtime.abi3.so`` during a full-lake content scan, and the
Chainlit app died with it. A segfault cannot be caught, so every session in
the process went down. The crash is intermittent -- twice in roughly a hundred
multi-threaded full-lake scans -- and no polars release after 1.40.1 lists a
matching fix.

The defence is a process boundary. Every polars call in this module runs in a
worker started as ``python -m lakegen.cell_scan``, and the process that calls
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

from collections import deque
import importlib
import os
from pathlib import Path
import pickle
import struct
import subprocess
import sys
import threading
from typing import Any, BinaryIO, Callable, Sequence

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
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _read_frame(stream: BinaryIO) -> bytes | None:
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
        return None
    uncastable = tuple(getattr(pl, name) for name in _UNCASTABLE)
    names = [
        name for name in schema.names() if schema[name].base_type() not in uncastable
    ]
    return names, len(schema.names()) - len(names)


def scan_columns(
    path: str,
    patterns: Sequence[tuple[str, str]],
    max_columns: int | None,
    max_rows: int | None,
) -> tuple[dict[str, dict[str, int]], tuple[str, ...], int]:
    """Count matches per term per column, one column in flight at a time.

    ``patterns`` pairs each term with its regex, escaped by the caller, so
    nothing an agent supplies reaches polars as syntax. Returns
    ``(term -> column -> matching cells, scanned columns, uncastable count)``.

    A ``max_columns`` or ``max_rows`` of ``None`` reads every column or every
    row. Rows are read ``_SCAN_CHUNK_ROWS`` at a time either way, so however
    long the table, a worker holds one chunk of one column.
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
            continue
        for index, (term, _pattern) in enumerate(patterns):
            if hits[index]:
                counts.setdefault(term, {})[name] = hits[index]
    return counts, tuple(searchable), uncastable


def probe_count(
    path: str,
    patterns: Sequence[tuple[str, str]],
    max_columns: int | None,
    max_rows: int,
) -> int:
    """Count matching rows in a short prefix, as a cheap stand-in for a scan."""
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
        operation, tasks = pickle.loads(frame)
        function = _resolve(operation)
        _write_frame(
            results,
            pickle.dumps([function(*task) for task in tasks], pickle.HIGHEST_PROTOCOL),
        )


# ----------------------------------------------------------------- parent side


def _start_worker() -> subprocess.Popen[bytes]:
    root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    # Make ``lakegen`` importable in the child however the parent found it.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (root, env.get("PYTHONPATH"))))
    return subprocess.Popen(
        # faulthandler turns a native crash into a traceback on stderr, which is
        # the only diagnostic a segfault leaves behind.
        [sys.executable, "-X", "faulthandler", "-m", "lakegen.cell_scan"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=env,
    )


def _stop_worker(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.stdin is not None:
            process.stdin.close()  # end of input ends the worker loop
        process.wait(timeout=10)
    except Exception:
        process.kill()
        process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()


def run(operation: str, tasks: Sequence[tuple[Any, ...]], *, workers: int) -> list[Any]:
    """Run ``operation`` over ``tasks`` in worker processes, keeping task order.

    A worker that dies has its batch re-queued one task per batch, so a file
    that really does break the scanner is isolated and stops costing its
    neighbours, while an intermittent crash simply succeeds on the retry. A task
    that crashes its worker ``_MAX_CRASHES_PER_TASK`` times alone gets ``None``
    as its result. If crashes pile up past a small budget the environment is
    broken rather than unlucky, and :class:`ScanWorkersFailed` is raised -- an
    ordinary exception the application can report and survive.
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

    def settle(
        batch: list[int], outcome: list[Any] | None, exit_code: int | None
    ) -> None:
        with condition:
            state["in_flight"] -= 1
            if outcome is not None:
                for index, value in zip(batch, outcome):
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
                elif len(batch) > 1:
                    pending.extend([index] for index in batch)
                    action = "retrying its tasks one at a time"
                elif crashes[batch[0]] < _MAX_CRASHES_PER_TASK:
                    pending.append(batch)
                    action = "retrying"
                else:
                    action = "skipping the task, it crashed its worker twice on its own"
                print(
                    f"[cell_scan] worker exited with {exit_code} during a batch of "
                    f"{len(batch)} task(s); {action}",
                    file=sys.stderr,
                    flush=True,
                )
            condition.notify_all()

    def work() -> None:
        process: subprocess.Popen[bytes] | None = None
        try:
            while (batch := next_batch()) is not None:
                if process is None:
                    process = _start_worker()
                frame: bytes | None
                try:
                    assert process.stdin is not None and process.stdout is not None
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
                    settle(batch, pickle.loads(frame), None)
                    continue
                exit_code = process.wait()
                _stop_worker(process)
                process = None
                settle(batch, None, exit_code)
        finally:
            if process is not None:
                _stop_worker(process)

    threads = [
        threading.Thread(target=work, name=f"cell-scan-{index}", daemon=True)
        for index in range(max(1, min(workers, len(pending))))
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if state["error"] is not None:
        raise state["error"]
    return results


if __name__ == "__main__":
    _serve()
