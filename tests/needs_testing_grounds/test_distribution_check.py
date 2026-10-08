import pandas as pd

from lakegen.distribution_check import (
    format_distribution_report, profile_column_distribution, profile_table_distribution,
)


def test_profile_numeric_column_basics():
    series = pd.Series([1.0, 2.0, 3.0, None])
    dist = profile_column_distribution(series, "amount")
    assert dist.name == "amount"
    assert dist.cardinality == 3
    assert dist.null_ratio == 0.25
    assert dist.numeric_min == 1.0
    assert dist.numeric_max == 3.0


def test_encoded_fiscal_year_span_detected():
    # 20 consecutive fiscal-year-span codes: "200001".."201920"
    codes = [int(f"{2000+y}{(y+1) % 100:02d}") for y in range(20)]
    series = pd.Series(codes)
    dist = profile_column_distribution(series, "Financial Year")
    assert dist.encoded_span is not None
    assert dist.encoded_span["kind"] == "fiscal_year_span"


def test_plain_years_not_flagged_as_encoded_span():
    series = pd.Series([2015, 2016, 2017, 2018] * 5)
    dist = profile_column_distribution(series, "Year")
    assert dist.encoded_span is None


def test_pinned_extreme_detected_for_top_coded_column():
    series = pd.Series([90] * 15 + list(range(1, 15)))
    dist = profile_column_distribution(series, "score")
    assert dist.pinned_extreme is not None
    assert "max=90" in dist.pinned_extreme


def test_categorical_top_values_populated():
    series = pd.Series(["a", "a", "a", "b", "b", "c"])
    dist = profile_column_distribution(series, "category")
    assert dist.top_values[0] == ("a", 3)


def test_profile_table_distribution_limits_to_requested_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    dists = profile_table_distribution(df, columns=["a", "c"])
    assert [d.name for d in dists] == ["a", "c"]


def test_format_distribution_report_includes_encoded_span_warning():
    codes = [int(f"{2000+y}{(y+1) % 100:02d}") for y in range(20)]
    series = pd.Series(codes)
    dist = profile_column_distribution(series, "Financial Year")
    report = format_distribution_report([dist])
    assert "ENCODED VALUE WARNING" in report
    assert "201617" in report or dist.encoded_span["example"] in report


def test_format_distribution_report_omits_warning_when_absent():
    dist = profile_column_distribution(pd.Series([1, 2, 3]), "plain")
    report = format_distribution_report([dist])
    assert "ENCODED VALUE WARNING" not in report
