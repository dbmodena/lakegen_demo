"""`read_table_sample` bounds the cost of measuring pairs of very large tables (NYC has tables of
up to 19 million rows) without reading them whole."""
import pandas as pd
import pytest

from lakegen.core.table_io import read_table_sample


def _write(tmp_path, rows, row_group_size=None, name="t.parquet"):
    path = tmp_path / name
    pd.DataFrame({"id": range(rows), "v": [f"x{i}" for i in range(rows)]}).to_parquet(
        path, row_group_size=row_group_size
    )
    return path


def test_a_table_that_fits_is_returned_whole_with_its_exact_row_count(tmp_path):
    frame, total = read_table_sample(_write(tmp_path, 500), 1000)
    assert len(frame) == 500 == total


def test_a_large_multi_row_group_table_is_sampled_across_the_whole_file(tmp_path):
    """Rows come from row groups spaced through the file, not just its first rows -- sorted data
    (dates, ids) would otherwise be sampled from one end only."""
    frame, total = read_table_sample(_write(tmp_path, 20_000, row_group_size=1_000), 3_000)
    assert total == 20_000 and 0 < len(frame) <= 3_000
    assert frame["id"].min() < 3_000 and frame["id"].max() > 17_000


def test_a_large_parquet_table_is_never_read_whole(tmp_path, monkeypatch):
    path = _write(tmp_path, 20_000, row_group_size=1_000)

    def whole(*_args, **_kwargs):
        raise AssertionError("read the whole table")

    monkeypatch.setattr(pd, "read_parquet", whole)
    frame, _ = read_table_sample(path, 3_000)
    assert len(frame) <= 3_000


def test_a_single_row_group_parquet_can_only_be_read_from_the_top(tmp_path):
    frame, total = read_table_sample(_write(tmp_path, 5_000, row_group_size=5_000), 1_000)
    assert total == 5_000 and len(frame) == 1_000 and frame["id"].max() == 999


def test_a_large_csv_returns_its_first_rows_and_no_total(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"id": range(5_000), "v": range(5_000)}).to_csv(path, index=False)
    frame, total = read_table_sample(path, 1_000)
    assert total is None and len(frame) == 1_000 and frame["id"].max() == 999


def test_a_small_csv_is_returned_whole(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"id": range(50)}).to_csv(path, index=False)
    frame, _ = read_table_sample(path, 1_000)
    assert len(frame) == 50


def test_columns_are_projected(tmp_path):
    path = _write(tmp_path, 20_000, row_group_size=1_000)
    frame, _ = read_table_sample(path, 3_000, columns=["id"])
    assert list(frame.columns) == ["id"]
    small, _ = read_table_sample(_write(tmp_path, 10, name="s.parquet"), 1_000, columns=["v"])
    assert list(small.columns) == ["v"]


def test_the_row_cap_is_at_least_one(tmp_path):
    frame, _ = read_table_sample(_write(tmp_path, 100, row_group_size=10), 0)
    assert len(frame) >= 1


def test_a_budget_smaller_than_one_row_group_samples_from_the_middle_of_the_file(tmp_path):
    frame, total = read_table_sample(_write(tmp_path, 20_000, row_group_size=1_000), 500)
    assert total == 20_000 and len(frame) == 500
    assert frame["id"].min() >= 10_000 and frame["id"].max() < 11_000
