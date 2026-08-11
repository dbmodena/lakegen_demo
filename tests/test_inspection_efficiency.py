import pandas as pd

from lakegen.agent_tools import tools_p2


def test_temporal_profile_is_bounded_and_keeps_exact_parquet_row_count(monkeypatch):
    chunks_requested = []

    def fake_chunks(_path, *, columns, chunk_rows):
        for index in range(10):
            chunks_requested.append(index)
            yield pd.DataFrame({"event_date": ["2024-01-01"] * chunk_rows})

    monkeypatch.setattr(tools_p2, "table_row_count", lambda _path: 2_000_000)
    monkeypatch.setattr(tools_p2, "iter_table_chunks", fake_chunks)

    row_label, coverage = tools_p2._temporal_profile(
        tools_p2.Path("large.parquet"), ["event_date"]
    )

    assert row_label == "2,000,000"
    assert len(chunks_requested) == 5
    assert coverage[0] == "- sampled first 500,000 rows (bounded profile)"


def test_temporal_profile_uses_metadata_without_scanning_non_temporal_parquet(
    monkeypatch,
):
    monkeypatch.setattr(tools_p2, "table_row_count", lambda _path: 7_000_000)
    monkeypatch.setattr(
        tools_p2,
        "iter_table_chunks",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("table data should not be scanned")
        ),
    )

    row_label, coverage = tools_p2._temporal_profile(
        tools_p2.Path("large.parquet"), ["name", "value"]
    )

    assert row_label == "7,000,000"
    assert coverage == []
