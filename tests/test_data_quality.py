import pandas as pd

from lakegen.tools.coder.quality.data_quality import profile_column_quality, profile_table_quality


def test_clean_column_returns_none():
    series = pd.Series(["Bologna", "Milano", "Roma"])
    assert profile_column_quality(series) is None


def test_blank_values_are_reported():
    series = pd.Series(["A", "", None, "B"])
    evidence = profile_column_quality(series)
    assert evidence["blank_or_null_count"] == 2
    assert evidence["blank_or_null_pct"] == 50.0


def test_sentinel_tokens_are_reported_case_insensitively():
    series = pd.Series(["N/A", "n/a", "Unknown", "42"])
    evidence = profile_column_quality(series)
    tokens = evidence["sentinel_tokens_observed"]
    assert tokens["n/a"] == 2
    assert tokens["unknown"] == 1


def test_numeric_formatting_noise_is_reported():
    series = pd.Series(["1,200", "3,400", "clean text"])
    evidence = profile_column_quality(series)
    assert set(evidence["numeric_format_examples"]) == {"1,200", "3,400"}


def test_profile_table_quality_skips_clean_columns():
    df = pd.DataFrame({
        "city": ["Bologna", "Milano"],
        "value": ["N/A", "100"],
    })
    findings = profile_table_quality(df)
    assert "city" not in findings
    assert "value" in findings
