"""Worker-process scanning: a native crash must cost a batch, not the caller.

The crashes here are real SIGSEGVs in real worker processes, raised by helper
functions the workers import by name, so the retry and isolation paths are
exercised end to end rather than simulated.
"""

import textwrap

import pandas as pd
import pytest

from lakegen.retrieval import cell_scan

_HELPERS = textwrap.dedent("""
    import os
    import signal
    from pathlib import Path

    def crash_first_time(marker_dir, value):
        marker = Path(marker_dir) / f"{value}.seen"
        if value == 5 and not marker.exists():
            marker.write_text("seen")
            os.kill(os.getpid(), signal.SIGSEGV)
        return value * 2

    def poison(value):
        if value == 3:
            os.kill(os.getpid(), signal.SIGSEGV)
        return value * 2

    def always(value):
        os.kill(os.getpid(), signal.SIGSEGV)
""")


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    (tmp_path / "crash_helpers.py").write_text(_HELPERS)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return tmp_path


def test_an_intermittent_worker_crash_is_retried(helpers):
    """The crash seen in production: rare, and gone on the next attempt."""
    markers = helpers / "markers"
    markers.mkdir()

    results = cell_scan.run(
        "crash_helpers:crash_first_time",
        [(str(markers), value) for value in range(20)],
        workers=4,
    )

    assert results == [value * 2 for value in range(20)]


def test_a_task_that_always_crashes_is_isolated_and_skipped(helpers):
    """One poison file must not cost its batch-mates, or the whole search."""
    tasks = [(value,) for value in range(12)]
    results = cell_scan.run("crash_helpers:poison", tasks, workers=3)

    assert results[3] is None
    assert all(results[v] == v * 2 for v in range(12) if v != 3)


def test_a_broken_environment_raises_instead_of_retrying_forever(helpers):
    """Past a small crash budget the caller gets an ordinary, catchable error."""
    with pytest.raises(cell_scan.ScanWorkersFailed):
        cell_scan.run("crash_helpers:always", [(v,) for v in range(40)], workers=4)


def test_results_keep_task_order_across_workers(tmp_path):
    for index in range(20):
        pd.DataFrame({"note": ["ferry"] * (index + 1)}).to_parquet(
            tmp_path / f"t{index:02d}.parquet"
        )
    tasks = [
        (str(tmp_path / f"t{index:02d}.parquet"), [("ferry", "(?i)ferry")], 40, 100_000)
        for index in range(20)
    ]

    results = cell_scan.run("scan", tasks, workers=4)

    assert [counts["ferry"]["note"] for counts, _cols, _unc in results] == list(
        range(1, 21)
    )


def test_chunk_boundaries_neither_lose_nor_double_count_rows(tmp_path, monkeypatch):
    """An uncapped scan reads in chunks; the counts must not show where they split."""
    path = tmp_path / "rows.parquet"
    pd.DataFrame(
        {"note": ["ferry" if index % 2 else "bus" for index in range(12)]}
    ).to_parquet(path)
    # Run in this process, so the patched chunk size applies: 12 rows in 3s ends
    # on an exact boundary, and a cap of 7 ends inside a chunk.
    monkeypatch.setattr(cell_scan, "_SCAN_CHUNK_ROWS", 3)
    patterns = [("ferry", "(?i)ferry")]

    full, _columns, _uncastable = cell_scan.scan_columns(str(path), patterns, None, None)
    capped, _columns, _uncastable = cell_scan.scan_columns(str(path), patterns, None, 7)

    assert full == {"ferry": {"note": 6}}
    assert capped == {"ferry": {"note": 3}}


def test_no_tasks_starts_no_workers():
    assert cell_scan.run("scan", [], workers=8) == []


def test_text_scan_counts_occurrences_in_string_columns_only(tmp_path):
    """The reference counts every occurrence, and only in VARCHAR/TEXT columns."""
    path = tmp_path / "mixed.parquet"
    pd.DataFrame(
        {"note": ["ferry to the ferry", "bus", None], "year": [2019, 2019, 2020]}
    ).to_parquet(path)
    patterns = [("ferry", "(?i)ferry"), ("2019", "2019")]

    counts, scanned, other = cell_scan.scan_text(str(path), patterns, None, None)

    assert counts == {"ferry": {"note": 2}}
    assert scanned == ("note",) and other == 1


def test_text_scan_chunk_boundaries_neither_lose_nor_double_count(tmp_path, monkeypatch):
    path = tmp_path / "rows.parquet"
    pd.DataFrame(
        {
            "a": ["ferry" if index % 2 else "bus" for index in range(12)],
            "b": ["ferry ferry"] * 12,
        }
    ).to_parquet(path)
    monkeypatch.setattr(cell_scan, "_SCAN_CHUNK_ROWS", 3)
    patterns = [("ferry", "(?i)ferry")]

    full, _scanned, _other = cell_scan.scan_text(str(path), patterns, None, None)
    capped, _scanned, _other = cell_scan.scan_text(str(path), patterns, None, 7)

    assert full == {"ferry": {"a": 6, "b": 24}}
    assert capped == {"ferry": {"a": 3, "b": 14}}


def test_text_scan_runs_in_workers(tmp_path):
    path = tmp_path / "t.parquet"
    pd.DataFrame({"note": ["ferry, ferry"], "n": [1]}).to_parquet(path)

    [result] = cell_scan.run(
        "scan_text", [(str(path), [("ferry", "(?i)ferry")], None, None)], workers=1
    )

    assert result == ({"ferry": {"note": 2}}, ("note",), 1)
