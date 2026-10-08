"""Value-grounding check: direct port of the scratchpad checks validated
against real production data (see coder-robustness plan) -- these fixtures
reproduce the exact bug shapes found live, not synthetic what-ifs."""

import pandas as pd
import pytest

from lakegen.value_grounding import (
    check_case_sensitivity_gaps, check_date_parse_sanity, check_value_grounding, extract_literal_comparisons,
)


@pytest.fixture
def fy_span_df():
    # Fiscal-year-span-coded column, exactly like the motivating bug's table.
    return pd.DataFrame({
        "Financial Year": [200001, 200102, 200203, 201516, 201617],
        "Amount written-off": [10.0, 20.0, 30.0, 40.0, 5.86],
        "Postcode": ["LS1 1AA", "LS2 2BB", "LS3 3CC", "LS4 4DD", "LS8 5NA"],
    })


def test_extract_literal_comparisons_direct_eq():
    comps = extract_literal_comparisons("mask = df['Financial Year'] == 2016")
    assert len(comps) == 1
    assert comps[0].column == "Financial Year"
    assert comps[0].operator == "Eq"
    assert comps[0].literal == 2016


def test_extract_literal_comparisons_isin():
    comps = extract_literal_comparisons("mask = df['col'].isin(['a', 'b'])")
    assert {c.literal for c in comps} == {"a", "b"}
    assert all(c.operator == "In" for c in comps)


def test_extract_literal_comparisons_eq_method_through_chain():
    """The st_53-class bug: .eq() on a chain of type/case transforms, not a
    bare == Compare node."""
    code = "mask = df['DEFECT'].astype(str).str.strip().str.lower().eq('open')"
    comps = extract_literal_comparisons(code)
    assert len(comps) == 1
    assert comps[0].column == "DEFECT"
    assert comps[0].operator == "Eq"
    assert comps[0].literal == "open"


def test_extract_literal_comparisons_ignores_variables_and_other_columns():
    code = "mask = df['a'] == df['b']\nmask2 = df['c'] == some_variable"
    assert extract_literal_comparisons(code) == []


def test_extract_literal_comparisons_multiple_conditions_independent():
    code = "mask = (df['a'] == 1) & (df['b'] == 2)"
    comps = extract_literal_comparisons(code)
    assert {(c.column, c.literal) for c in comps} == {("a", 1), ("b", 2)}


def test_catches_the_fy_span_bug(fy_span_df):
    comps = extract_literal_comparisons("mask = df['Financial Year'] == 2016")
    violations = check_value_grounding(comps, fy_span_df)
    assert len(violations) == 1
    assert violations[0].confidence == "HIGH"
    assert "fiscal-year-span" in violations[0].message


def test_no_false_positive_on_corrected_fy_span_code(fy_span_df):
    comps = extract_literal_comparisons("mask = df['Financial Year'] == 201617")
    assert check_value_grounding(comps, fy_span_df) == []


def test_no_false_positive_on_legitimate_numeric_range_filter(fy_span_df):
    comps = extract_literal_comparisons("mask = df['Amount written-off'] > 5")
    assert check_value_grounding(comps, fy_span_df) == []


def test_soft_violation_on_out_of_range_numeric_filter(fy_span_df):
    comps = extract_literal_comparisons("mask = df['Amount written-off'] > 999999")
    violations = check_value_grounding(comps, fy_span_df)
    assert violations and violations[0].confidence == "SOFT"


def test_hard_violation_on_string_literal_never_observed(fy_span_df):
    comps = extract_literal_comparisons("mask = df['Postcode'] == 'ZZ9 9ZZ'")
    violations = check_value_grounding(comps, fy_span_df)
    assert violations and violations[0].confidence == "HIGH"


def test_date_parse_sanity_catches_wrong_column():
    df = pd.DataFrame({
        "_duplicated_2": ["Dragonfly House Rent and SC", "Contracted Services"],
    })
    code = "x = pd.to_datetime(df['_duplicated_2'], format='%d/%m/%Y', errors='coerce')"
    violations = check_date_parse_sanity(code, df)
    assert len(violations) == 1
    assert violations[0].confidence == "HIGH"
    assert "100%" in violations[0].message


def test_date_parse_sanity_no_false_positive_on_real_dates():
    df = pd.DataFrame({"date_col": ["04/12/2018", "14/12/2018", "22/12/2018"]})
    code = "x = pd.to_datetime(df['date_col'], format='%d/%m/%Y', errors='coerce')"
    assert check_date_parse_sanity(code, df) == []


@pytest.fixture
def scarborough_df():
    # Direct reproduction of the live sediment-habitat bug's data shape: a
    # LocationName column with inconsistent capitalization for the same
    # real-world place name ("Scarborough" vs "scarborough").
    return pd.DataFrame({
        "LocationName": [f"Station {i}. (Scarborough)" for i in range(1, 9)]
                         + [f"Station {i}. (scarborough)" for i in range(9, 17)],
        "HabitatCode": ["SS.SSa"] * 16,
    })


def test_catches_the_scarborough_case_sensitivity_bug(scarborough_df):
    code = "mask = df['LocationName'].str.contains('Scarborough', na=False)"
    violations = check_case_sensitivity_gaps(code, scarborough_df)
    assert len(violations) == 1
    v = violations[0]
    assert v.column == "LocationName"
    assert v.confidence == "SOFT"
    assert "8" in v.message  # the 8 lowercase-labelled rows silently excluded


def test_no_false_positive_when_case_false_is_already_used(scarborough_df):
    code = "mask = df['LocationName'].str.contains('Scarborough', case=False, na=False)"
    assert check_case_sensitivity_gaps(code, scarborough_df) == []


def test_no_false_positive_when_chain_is_already_case_normalized(scarborough_df):
    code = "mask = df['LocationName'].str.lower().str.contains('scarborough', na=False)"
    assert check_case_sensitivity_gaps(code, scarborough_df) == []


def test_no_false_positive_when_capitalization_is_actually_consistent():
    df = pd.DataFrame({"LocationName": ["Scarborough Station 1", "Scarborough Station 2"]})
    code = "mask = df['LocationName'].str.contains('Scarborough', na=False)"
    assert check_case_sensitivity_gaps(code, df) == []


def test_no_false_positive_on_contains_call_for_missing_column(scarborough_df):
    code = "mask = df['NotAColumn'].str.contains('Scarborough', na=False)"
    assert check_case_sensitivity_gaps(code, scarborough_df) == []
