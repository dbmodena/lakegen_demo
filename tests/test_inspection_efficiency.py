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


def test_temporal_column_names_may_use_any_separator():
    matches = tools_p2._TEMPORAL_COLUMN_PATTERN.search
    for name in (
        "Invoice Payment Date", "Payment-Date", "invoice_date", "Date", "event_date",
        "Financial Year", "Year",
    ):
        assert matches(name), name
    for name in ("Candidate", "Update", "Timestamped", "Response Times", "Amount"):
        assert not matches(name), name


def _profile(tmp_path, frame):
    path = tmp_path / "t.parquet"
    frame.to_parquet(path)
    return tools_p2._temporal_profile(path, list(frame.columns))


def test_day_first_dates_are_read_day_first(tmp_path):
    # A UK January-March table. Read month-first, 02/01/2025 is 1 February and
    # 12/03/2025 is 3 December, so the table would seem to span the whole year.
    frame = pd.DataFrame({"Invoice Payment Date": [
        "02/01/2025", "12/02/2025", "12/03/2025", "31/03/2025",
    ]})

    _, coverage = _profile(tmp_path, frame)

    assert coverage == [
        "- Invoice Payment Date: 2025-01-02 to 2025-03-31 (missing/unparseable 0.0%)"
    ]


def test_month_first_dates_stay_month_first(tmp_path):
    frame = pd.DataFrame({"Payment Date": ["01/31/2025", "02/03/2025"]})

    _, coverage = _profile(tmp_path, frame)

    assert coverage == [
        "- Payment Date: 2025-01-31 to 2025-02-03 (missing/unparseable 0.0%)"
    ]


def test_a_duration_named_time_is_not_profiled_as_a_date(tmp_path):
    # Read as timestamps, minutes land in 1970 and would fail every year check.
    _, coverage = _profile(tmp_path, pd.DataFrame({"Response Time": [5, 10, 12]}))

    assert coverage == []


def test_the_periods_a_question_names_treat_a_fiscal_year_as_two_calendar_years():
    periods = tools_p2._requested_periods(
        "spend in the first quarter of 2024/25, the 4th quarter of 2024/2025, "
        "the 4th quarter of 2020/21, the 1st quarter of 2021/22 and in 2019"
    )

    assert periods == [(2019,), (2020, 2021), (2021, 2022), (2024, 2025)]
    # A range or a full date is not a fiscal year: each year stands alone.
    assert tools_p2._requested_periods("from 2015-2020") == [(2015,), (2020,)]
    assert tools_p2._requested_periods("since 2020-01-05") == [(2020,)]


def _coverage(*years):
    return "Temporal coverage:\n" + "\n".join(
        f"- Date: {start} to {end} (missing/unparseable 0.0%)" for start, end in years
    )


def test_year_check_accepts_the_calendar_year_a_fiscal_quarter_falls_in():
    # 4th quarter of 2020/21 is January-March 2021: no date in 2020.
    issue = tools_p2._temporal_coverage_issue(
        "spend in the 4th quarter of 2020/21", ["a"], {"a": _coverage((2021, 2021))}
    )

    assert issue is None


def test_year_check_still_blocks_a_proven_mismatch():
    fiscal = tools_p2._temporal_coverage_issue(
        "spend in 2020/21", ["a"], {"a": _coverage((2019, 2019))}
    )
    bare = tools_p2._temporal_coverage_issue(
        "spend in 2020", ["a"], {"a": _coverage((2021, 2021))}
    )

    assert "outside the inspected temporal coverage" in fiscal and "2020/21" in fiscal
    assert "outside the inspected temporal coverage" in bare and "['2020']" in bare
