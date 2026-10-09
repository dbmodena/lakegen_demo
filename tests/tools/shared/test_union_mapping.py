"""Tiered union evidence. Regression tests are named after what they pin: the false positives of the
old average-score rule (19-21% of random pairs on OrQa's UK and NYC data), the organogram unions a
value-domain veto wrongly rejected, and the placeholder-header tables."""
import numpy as np
import pandas as pd

from lakegen.tools.shared.join_union.union_mapping import (
    NO_UNION,
    PARTIAL_UNION,
    SUBSET_UNION,
    UNION,
    _pair_warning,
    format_union_section,
    real_columns,
    union_by_names,
)


def _coma(cols_l, cols_r=None, score=1.0):
    cols_r = cols_r or cols_l
    return [(a, b, score) for a, b in zip(cols_l, cols_r)]


def _warn(a, b):
    return _pair_warning(pd.Series(a), pd.Series(b))


# ---- warnings on a mapped pair: only what breaks a concat ------------------------------------

def test_time_slices_with_different_constant_years_are_the_same_domain():
    """Two yearly snapshots: Year is 2019 in one, 2020 in the other -- same attribute."""
    assert _warn([2019] * 50, [2020] * 50) == ""


def test_amounts_and_small_counts_warn_about_magnitude():
    amounts = np.random.default_rng(0).uniform(500, 90_000, 200)
    counts = np.random.default_rng(1).integers(0, 12, 200)
    assert "magnitude" in _warn(amounts, counts)


def test_a_magnitude_warning_shows_the_medians_not_a_meaningless_ratio():
    """A column of zeros against real amounts once printed 'medians differ 1,000,000,000x'."""
    zeros = np.zeros(100)
    amounts = np.random.default_rng(1).uniform(500, 90_000, 100)
    warning = _warn(zeros, amounts)
    assert "magnitude" in warning and "medians 0 vs " in warning
    assert "1,000,000,000" not in warning


def test_counts_from_two_periods_do_not_warn():
    a = np.random.default_rng(0).integers(0, 40, 200)
    b = np.random.default_rng(1).integers(0, 55, 200)
    assert _warn(a, b) == ""


def test_numbers_stored_as_text_with_separators_are_numbers():
    assert _warn(["2,012,938", "2,093,291", "2,048,821"], [2103161.0, 2005136.0, 2096672.0]) == ""


def test_text_ids_against_integer_ids_are_a_type_clash():
    """`Post Unique Reference` is 'DWP-001' in one department and 961 in another."""
    warning = _warn([f"DWP-{i:03d}" for i in range(20)], list(range(900, 920)))
    assert "type clash" in warning and "text" in warning and "number" in warning


def test_a_date_against_text_is_a_type_clash():
    assert "date" in _warn(pd.to_datetime(["2020-01-01", "2020-02-01"]), ["a", "b"])


def test_empty_columns_warn():
    assert "empty" in _warn([np.nan, np.nan], [1, 2])
    assert "empty" in _warn([None, None], ["a", "b"])            # an all-null TEXT column too


def test_free_text_differences_are_silent():
    """12.7% of mapped pairs in TRUE unions differ in free-text vocabulary or format (department
    names, job titles, comments); warning on them would bury the informative warnings."""
    assert _warn([f"DWP-{i:03d}" for i in range(20)], [f"CO{i:03d}" for i in range(20)]) == ""
    assert _warn(["Department for Work and Pensions"] * 5, ["Cabinet Office"] * 5) == ""
    assert _warn(["Brooklyn", "Queens"], ["K", "Q"]) == ""


# ---- placeholder headers ---------------------------------------------------------------------

def test_placeholder_headers_are_not_attributes():
    """Spreadsheets parsed with unnamed columns: '' and _duplicated_N match each other by name."""
    df = pd.DataFrame({"Real": [1], "": [2], "_duplicated_0": [3], "Unnamed: 4": [4], "Column1": [5]})
    assert real_columns(df) == ["Real"]


def test_placeholder_columns_do_not_create_a_union():
    """Real failure: FCDO and GPC sheets 'matched' on ''/_duplicated_N columns holding unrelated data."""
    a = pd.DataFrame({"FCDO Business Appointments": [1], "": ["x"], "_duplicated_0": ["y"], "_duplicated_1": ["z"]})
    b = pd.DataFrame({"August GPC Spend": [1], "": ["p"], "_duplicated_0": ["q"], "_duplicated_1": ["r"]})
    coma = [(c, c, 1.0) for c in a.columns if c != "FCDO Business Appointments"]
    coma += [("FCDO Business Appointments", "August GPC Spend", 0.3)]
    mapping = union_by_names(a, b, coma)
    assert mapping.verdict == NO_UNION and mapping.placeholders == 6


# ---- tiers -----------------------------------------------------------------------------------

def test_same_schema_snapshots_are_a_union():
    a = pd.DataFrame({"Name": ["Ann Lee", "Bob Ray"], "Grade": ["G6", "G7"], "Year": [2019, 2019]})
    b = pd.DataFrame({"Name": ["Cy Dunn", "Di Fox"], "Grade": ["G7", "G6"], "Year": [2020, 2020]})
    mapping = union_by_names(a, b, _coma(list(a.columns)))
    assert mapping.verdict == UNION and len(mapping.aligned) == 3 and mapping.cov_large == 1.0


def test_domain_differences_warn_but_never_veto_the_mapping():
    """Organogram tables from two departments: `Organisation` values never overlap, yet it is the
    same attribute. A value-domain veto rejected these genuine unions (22-42% of true unions)."""
    a = pd.DataFrame({"Organisation": ["Department for Work and Pensions"] * 5, "Grade": ["G6"] * 5})
    b = pd.DataFrame({"Organisation": ["Cabinet Office"] * 5, "Grade": ["G7"] * 5})
    mapping = union_by_names(a, b, _coma(["Organisation", "Grade"]))
    assert mapping.verdict == UNION and len(mapping.aligned) == 2 and not mapping.warnings


def test_a_type_clash_is_reported_on_the_mapped_pair_and_the_union_stands():
    a = pd.DataFrame({"Post Unique Reference": [f"DWP-{i:03d}" for i in range(20)], "Name": ["Ann Lee"] * 20})
    b = pd.DataFrame({"Post Unique Reference": list(range(900, 920)), "Name": ["Bob Ray"] * 20})
    mapping = union_by_names(a, b, _coma(["Post Unique Reference", "Name"]))
    assert mapping.verdict == UNION
    assert [m.left for m in mapping.warnings] == ["Post Unique Reference"]


def test_name_similar_columns_without_coverage_are_not_a_union():
    """The old rule's false-positive class: two or three name-similar columns among many."""
    a = pd.DataFrame({f"a{i}": [1, 2] for i in range(10)} | {"Date": ["x", "y"], "Name": ["p", "q"]})
    b = pd.DataFrame({f"b{i}": [1, 2] for i in range(10)} | {"Date": ["x", "y"], "Name": ["p", "q"]})
    assert union_by_names(a, b, _coma(["Date", "Name"])).verdict == NO_UNION


def _frames(n_left, n_right, n_mapped):
    a = pd.DataFrame({f"c{i}": [1, 2] for i in range(n_mapped)} | {f"l{i}": [1, 2] for i in range(n_left - n_mapped)})
    b = pd.DataFrame({f"c{i}": [1, 2] for i in range(n_mapped)} | {f"r{i}": [1, 2] for i in range(n_right - n_mapped)})
    return a, b, _coma([f"c{i}" for i in range(n_mapped)])


def test_partial_union_needs_half_of_both_tables_and_three_columns():
    assert union_by_names(*_frames(6, 5, 3)).verdict == PARTIAL_UNION      # 3 of 6 and 3 of 5
    assert union_by_names(*_frames(20, 5, 3)).verdict == NO_UNION          # 15% of the larger table
    assert union_by_names(*_frames(4, 4, 2)).verdict == NO_UNION           # 2 columns is not enough


def test_a_smaller_table_fully_contained_in_a_larger_one_is_a_subset_union_never_a_union():
    """NYC: a 45-column table whose columns all appear in a 69-column table. 42 of OrQa's labelled
    NYC unions look like this; requiring both tables to map lost them, and calling them UNION
    would claim the same arity."""
    small = pd.DataFrame({"id": ["a"], "score": [1], "n": [2]})
    big = pd.DataFrame({"id": ["b"], "score": [3], "n": [4], "p": [1], "q": [2], "r": [3]})
    mapping = union_by_names(small, big, _coma(["id", "score", "n"]))
    assert mapping.cov_small == 1.0 and mapping.cov_large == 0.5
    assert mapping.verdict == SUBSET_UNION


def test_a_single_matching_column_is_never_a_union():
    a, b = pd.DataFrame({"id": ["a1", "a2"]}), pd.DataFrame({"id": ["b1", "b2"]})
    assert union_by_names(a, b, _coma(["id"])).verdict == NO_UNION
    a2, b2 = pd.DataFrame({"id": ["a1"], "n": [1]}), pd.DataFrame({"id": ["b1"], "n": [2]})
    assert union_by_names(a2, b2, _coma(["id", "n"])).verdict == UNION     # two columns is enough


def test_the_mapping_is_one_to_one():
    """Two candidate targets for one source column: only the better-named one is mapped, and a
    target column is never used twice (paper: 1-1 mapping h)."""
    a = pd.DataFrame({"Total": [10, 20, 30], "Amount": [1, 2, 3], "Note": ["a", "b", "c"]})
    b = pd.DataFrame({"Total": [12, 22, 32], "Total (adj)": [11, 21, 31], "Note": ["d", "e", "f"]})
    coma = [("Total", "Total", 1.0), ("Total", "Total (adj)", 0.7), ("Amount", "Total", 0.6), ("Note", "Note", 1.0)]
    mapping = union_by_names(a, b, coma)
    pairs = [(m.left, m.right) for m in mapping.aligned]
    assert ("Total", "Total") in pairs and ("Total", "Total (adj)") not in pairs and ("Amount", "Total") not in pairs
    assert len({m.left for m in mapping.aligned}) == len(mapping.aligned) == len({m.right for m in mapping.aligned})


def test_low_name_scores_are_not_candidates():
    """Two columns per table, all name-scored below the 0.5 threshold: without the threshold this
    would be a full UNION on two mapped columns."""
    a = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
    b = pd.DataFrame({"y1": [1, 2], "y2": [3, 4]})
    mapping = union_by_names(a, b, [("x1", "y1", 0.3), ("x2", "y2", 0.3)])
    assert mapping.verdict == NO_UNION and mapping.aligned == []


def test_a_renamed_column_still_maps_when_its_name_score_passes():
    a = pd.DataFrame({"LSOA": ["E01000001", "E01000002"], "Attendance (%)": [92.5, 88.1]})
    b = pd.DataFrame({"LSOA11": ["E01000003", "E01000004"], "Attendance (%)": [90.2, 85.0]})
    mapping = union_by_names(a, b, [("LSOA", "LSOA11", 0.667), ("Attendance (%)", "Attendance (%)", 1.0)])
    assert mapping.verdict == UNION and len(mapping.aligned) == 2


# ---- report ----------------------------------------------------------------------------------

def test_report_for_a_union_names_coverage_and_the_mapped_pairs():
    a = pd.DataFrame({"Name": ["Ann"], "Grade": ["G6"]})
    b = pd.DataFrame({"Name": ["Bob"], "Grade": ["G7"]})
    text = format_union_section("a.parquet", "b.parquet", union_by_names(a, b, _coma(["Name", "Grade"])))
    assert text.startswith("UNION: 'a.parquet' unions with 'b.parquet'; 2 columns map 1-1")
    assert "100% of the left table's columns" in text and "Name -> Name" in text and "Grade -> Grade" in text


def test_report_for_a_subset_union_says_which_table_has_the_extra_columns():
    small = pd.DataFrame({"id": ["a"], "score": [1], "n": [2]})
    big = pd.DataFrame({"id": ["b"], "score": [3], "n": [4], "p": [1], "q": [2], "r": [3]})
    text = format_union_section("small.parquet", "big.parquet", union_by_names(small, big, _coma(["id", "score", "n"])))
    assert text.startswith("SUBSET UNION")
    assert "smaller table 'small.parquet'" in text and "'big.parquet' has 3 more" in text


def test_report_for_no_union_states_the_rule_it_applied():
    text = format_union_section("A", "B", union_by_names(*_frames(20, 5, 3)))
    assert text.startswith("NO UNION: only 3 of 20 / 5 columns") and ">= 80%" in text
    assert "RELATIONSHIP" not in text


def test_report_lists_warnings_and_placeholder_note():
    a = pd.DataFrame({"Post Unique Reference": [f"DWP-{i:03d}" for i in range(20)], "Name": ["Ann Lee"] * 20, "": [1] * 20})
    b = pd.DataFrame({"Post Unique Reference": list(range(900, 920)), "Name": ["Bob Ray"] * 20, "": [2] * 20})
    text = format_union_section("A", "B", union_by_names(a, b, _coma(["Post Unique Reference", "Name", ""])))
    assert "warning: Post Unique Reference -> Post Unique Reference: type clash" in text
    assert "ignored 2 placeholder header column(s)" in text


def test_report_truncates_a_long_mapping():
    cols = [f"col{i}" for i in range(12)]
    a = pd.DataFrame({c: [1] for c in cols})
    b = pd.DataFrame({c: [2] for c in cols})
    text = format_union_section("A", "B", union_by_names(a, b, _coma(cols)))
    assert text.count(" -> ") == 8 and "... and 4 more" in text
