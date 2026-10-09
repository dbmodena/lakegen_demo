"""Measured join-key evidence. Every regression test is named after the real failure it pins
(OrQa UK/NYC benchmark matches and the 80-pair name-only probe that motivated the module)."""
import numpy as np
import pandas as pd
import pytest

from lakegen.tools.shared.join_union import join_keys as jp
from lakegen.tools.shared.join_union.join_keys import (
    CONSTANT_KEY,
    JOIN,
    NO_OVERLAP,
    SURROGATE_ID,
    WEAK_OVERLAP,
    find_join_keys,
    format_join_section,
    measure_key,
)


def _m(q, r, lc, rc, score=1.0):
    return measure_key(q, r, lc if isinstance(lc, list) else [lc],
                       rc if isinstance(rc, list) else [rc], score)


def test_clean_one_to_many_join():
    q = pd.DataFrame({"id": ["a", "b", "c"], "name": ["x", "y", "z"]})
    r = pd.DataFrame({"id": ["a", "a", "b", "c", "c"], "amt": [1, 2, 3, 4, 5]})
    ev = _m(q, r, "id", "id")
    assert ev.verdict == JOIN and ev.cardinality == "1:N"
    assert ev.inner_rows == 5 and ev.left_match_rate == 1.0 and ev.right_match_rate == 1.0


def test_zero_overlap_is_rejected_despite_identical_names():
    """The dominant real failure: `Date=Date` / `Name=NAME` score 1.000 in COMA."""
    q = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"]})
    r = pd.DataFrame({"Date": ["2021-05-05", "2021-05-06"]})
    assert _m(q, r, "Date", "Date").verdict == NO_OVERLAP


def test_single_valued_key_is_constant():
    """`Parent Department` (one value per file) -- a join would be a cross product."""
    q = pd.DataFrame({"Parent Department": ["Cabinet Office"] * 5, "v": range(5)})
    r = pd.DataFrame({"Parent Department": ["Cabinet Office"] * 4, "w": range(4)})
    ev = _m(q, r, "Parent Department", "Parent Department")
    assert ev.verdict == CONSTANT_KEY and ev.shared_distinct == 1


def test_case_and_whitespace_are_normalised():
    q = pd.DataFrame({"k": [" Leeds ", "YORK"]})
    r = pd.DataFrame({"k": ["leeds", "york", "hull"]})
    ev = _m(q, r, "k", "k")
    assert ev.shared_distinct == 2 and ev.verdict == JOIN


def test_int_and_integral_float_match():
    q = pd.DataFrame({"k": [1, 2, 3]})
    r = pd.DataFrame({"k": [1.0, 2.0, 4.0, np.nan]})
    ev = _m(q, r, "k", "k")
    assert ev.shared_distinct == 2


def test_numeric_looking_strings_match_numbers_and_flag_no_dtype_issue():
    q = pd.DataFrame({"k": [10, 20, 30]})
    r = pd.DataFrame({"k": ["10", "20", "99"]})
    ev = _m(q, r, "k", "k")
    assert ev.shared_distinct == 2
    assert not any("dtype" in n for n in ev.notes)


def test_incompatible_dtypes_are_noted_and_rejected():
    q = pd.DataFrame({"k": [1, 2, 3]})
    r = pd.DataFrame({"k": ["alpha", "beta", "gamma"]})
    ev = _m(q, r, "k", "k")
    assert ev.verdict == NO_OVERLAP and any("dtype" in n for n in ev.notes)


def test_row_number_columns_are_not_evidence():
    """`OBJECTID=ObjectId` on two unrelated tables: both are 1..N."""
    q = pd.DataFrame({"OBJECTID": range(1, 74)})
    r = pd.DataFrame({"ObjectId": range(1, 74)})
    ev = _m(q, r, "OBJECTID", "ObjectId")
    assert ev.verdict == SURROGATE_ID and not ev.supported


def test_sparse_real_ids_are_not_mistaken_for_row_numbers():
    q = pd.DataFrame({"id": [1003, 1007, 1011, 2050]})
    r = pd.DataFrame({"id": [1003, 1007, 1011, 2050, 3000]})
    assert _m(q, r, "id", "id").verdict == JOIN


def test_many_to_many_is_flagged_not_rejected():
    """The coder prompt already says to aggregate first; N:N must stay usable."""
    q = pd.DataFrame({"k": ["a", "a", "b", "b"], "x": range(4)})
    r = pd.DataFrame({"k": ["a", "a", "a", "b"], "y": range(4)})
    ev = _m(q, r, "k", "k")
    assert ev.verdict == JOIN and ev.cardinality == "N:N"
    assert ev.inner_rows == 2 * 3 + 2 * 1
    assert any("aggregate" in n for n in ev.notes)


def test_thin_overlap_backed_by_many_values_is_still_a_join_but_is_flagged():
    """Gold merges include how="left" joins matching ~13-20% of rows; rejecting on
    match rate alone would throw away a correct join."""
    q = pd.DataFrame({"k": [f"q{i}" for i in range(100)] + [f"s{i}" for i in range(6)]})
    r = pd.DataFrame({"k": [f"r{i}" for i in range(100)] + [f"s{i}" for i in range(6)]})
    ev = _m(q, r, "k", "k")
    assert ev.verdict == JOIN and ev.thin and ev.shared_distinct == 6
    assert any("thin overlap" in n for n in ev.notes)


def test_thin_overlap_backed_by_a_couple_of_values_is_coincidence():
    """`Body name=NAME` on unrelated tables: 1-2 shared values, ~0% of rows match."""
    q = pd.DataFrame({"k": [f"q{i}" for i in range(100)] + ["shared"]})
    r = pd.DataFrame({"k": [f"r{i}" for i in range(100)] + ["shared"]})
    ev = _m(q, r, "k", "k")
    assert ev.verdict == WEAK_OVERLAP and not ev.supported
    assert any("coincidence" in n for n in ev.notes)


def test_two_shared_values_covering_most_rows_is_not_coincidence():
    """A tiny lookup table fully contained in the other is a genuine join."""
    q = pd.DataFrame({"k": ["a", "b"]})
    r = pd.DataFrame({"k": ["a", "b", "c", "d"]})
    assert _m(q, r, "k", "k").verdict == JOIN


def test_small_table_fully_contained_in_large_one_is_a_join():
    q = pd.DataFrame({"k": ["a", "b"]})
    r = pd.DataFrame({"k": [f"{c}" for c in "abcdefghij"]})
    ev = _m(q, r, "k", "k")
    assert ev.verdict == JOIN and ev.left_match_rate == 1.0 and ev.right_match_rate == 0.2


def test_null_keys_do_not_match_each_other():
    q = pd.DataFrame({"k": [None, None, "a"]})
    r = pd.DataFrame({"k": [None, "a", "b"]})
    ev = _m(q, r, "k", "k")
    assert ev.shared_distinct == 1 and ev.left_rows == 1


def test_composite_key_found_when_no_single_column_is_unique():
    """Direct port of the benchmark shape: Lough + Point + Date is the key."""
    q = pd.DataFrame({
        "Lough": ["A", "A", "B", "B"], "Point": ["p1", "p2", "p1", "p2"],
        "Date": ["d1"] * 4, "phyto": range(4),
    })
    r = pd.DataFrame({
        "Lough": ["A", "A", "B", "B"], "Point": ["p1", "p2", "p1", "p2"],
        "Date": ["d1"] * 4, "bio": range(4),
    })
    coma = [("Lough", "Lough", 1.0), ("Point", "Point", 1.0), ("Date", "Date", 1.0)]
    ranked = find_join_keys(q, r, coma)
    best = ranked[0]
    assert best.verdict == JOIN and best.cardinality == "1:1"
    assert set(best.left_cols) >= {"Lough", "Point"}


def test_ranking_prefers_a_unique_key_over_a_higher_name_score_but_worse_key():
    q = pd.DataFrame({"Name": ["x", "x", "y", "y"], "ref": ["r1", "r2", "r3", "r4"]})
    r = pd.DataFrame({"Name": ["x", "y", "y", "x"], "Reference": ["r1", "r2", "r3", "r4"]})
    coma = [("Name", "Name", 1.0), ("ref", "Reference", 0.6)]
    ranked = find_join_keys(q, r, coma)
    assert ranked[0].left_cols == ("ref",) and ranked[0].cardinality == "1:1"


def test_no_name_similar_pairs_means_no_candidates():
    q = pd.DataFrame({"a": [1]})
    r = pd.DataFrame({"b": [1]})
    assert find_join_keys(q, r, [("a", "b", 0.2)]) == []


def _name_grade_frames(shared_names):
    names_q = [f"person{i}" for i in range(100)]
    names_r = [f"person{i}" for i in range(100 - shared_names, 200 - shared_names)]
    grades = ["G1", "G2", "G3"]
    q = pd.DataFrame({"Name": names_q, "Grade": [grades[i % 3] for i in range(100)]})
    r = pd.DataFrame({"Name": names_r, "Grade": [grades[i % 3] for i in range(100)]})
    return q, r, [("Grade", "Grade", 1.0), ("Name", "Name", 1.0)]


def test_thin_unique_key_outranks_a_fully_overlapping_categorical_key():
    """Gold shape 8fdda43c: `Name` (unique, thin overlap) is the real key;
    `Grade` (categorical, 100% overlap, N:N) must not win on overlap alone --
    provided the overlap is backed by more than a coincidence's worth of values."""
    q, r, coma = _name_grade_frames(shared_names=6)
    ranked = find_join_keys(q, r, coma)
    assert ranked[0].left_cols == ("Name",)
    assert ranked[0].verdict == JOIN and ranked[0].thin


def test_two_shared_names_are_coincidence_so_the_categorical_key_wins():
    """The trade-off, stated: at <=2 shared values a unique thin key is no longer
    distinguishable from coincidence (the benchmark's own Name=Name gold join,
    which shares exactly 1 name, is classified this way)."""
    q, r, coma = _name_grade_frames(shared_names=2)
    ranked = find_join_keys(q, r, coma)
    assert ranked[0].left_cols == ("Grade",)
    name = next(e for e in ranked if e.left_cols == ("Name",))
    assert name.verdict == WEAK_OVERLAP


def test_thin_composite_unique_key_outranks_categorical_key():
    """Gold shape 399a6a23: a 2-column composite key that is unique but only
    partly overlapping beats a fully-overlapping categorical N:N column."""
    rows = [(a, b) for a in "ABCDE" for b in "wxyz"]          # 20 unique combos
    q = pd.DataFrame(rows, columns=["Lough", "Point"])
    q["Status"] = ["ok"] * len(q)
    q["Status"] = ["ok", "bad"] * (len(q) // 2)
    r = q.iloc[:6].copy()                                       # only 6 of 20 combos overlap
    r["Status"] = ["ok", "bad", "ok", "bad", "ok", "bad"]
    coma = [("Status", "Status", 1.0), ("Lough", "Lough", 1.0), ("Point", "Point", 1.0)]
    best = find_join_keys(q, r, coma)[0]
    assert set(best.left_cols) == {"Lough", "Point"} and best.unique_side


def test_fractional_float_key_is_flagged_and_ranked_last():
    """Gold shape f7b9a1d7: `Amount=Amount` overlaps by coincidence and is nearly
    unique, but joining on a payment amount is never the intent."""
    q = pd.DataFrame({"Supplier": ["a", "a", "b", "b", "c"], "Amount": [10.5, 20.25, 30.75, 40.5, 50.1]})
    r = pd.DataFrame({"Supplier": ["a", "b", "b", "c", "c"], "Amount": [10.5, 99.9, 30.75, 41.5, 50.1]})
    coma = [("Amount", "Amount", 1.0), ("Supplier", "Supplier", 1.0)]
    ranked = find_join_keys(q, r, coma)
    amount = next(e for e in ranked if e.left_cols == ("Amount",))
    assert amount.measure_like and any("measurement" in n for n in amount.notes)
    assert ranked[0].left_cols == ("Supplier",)


def test_integral_float_ids_are_not_measure_like():
    q = pd.DataFrame({"id": [1.0, 2.0, 3.0]})
    r = pd.DataFrame({"id": [2.0, 3.0, 4.0]})
    assert not _m(q, r, "id", "id").measure_like


def test_key_with_one_coincidental_shared_value_ranks_below_real_overlap():
    q = pd.DataFrame({"Ref": [f"q{i}" for i in range(20)] + ["x"], "Area": ["N", "S", "E"] * 7})
    r = pd.DataFrame({"Ref": [f"r{i}" for i in range(20)] + ["x"], "Area": ["N", "S", "E"] * 7})
    coma = [("Ref", "Ref", 1.0), ("Area", "Area", 1.0)]
    ranked = find_join_keys(q, r, coma)
    assert ranked[0].left_cols == ("Area",)          # 3 shared values beats 1 shared value
    assert [e.left_cols for e in ranked] == [("Area",), ("Ref",)]   # no composite proposed either


def test_key_ness_orders_non_unique_candidates():
    """Gold shape a2094364: Supplier Name (many distinct) over Service Area (few)."""
    sup = [f"s{i % 40}" for i in range(200)]
    area = [f"a{i % 3}" for i in range(200)]
    q = pd.DataFrame({"Supplier Name": sup, "Service Area": area})
    r = pd.DataFrame({"Supplier Name": sup, "Service Area": area})
    coma = [("Service Area", "Service Area", 1.0), ("Supplier Name", "Supplier Name", 1.0)]
    assert find_join_keys(q, r, coma)[0].left_cols == ("Supplier Name",)


def test_composite_search_is_skipped_when_a_single_column_is_already_a_key(monkeypatch):
    monkeypatch.setattr(jp, "_composite_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran")))
    q = pd.DataFrame({"id": ["a", "b", "c"], "n": [1, 2, 3]})
    r = pd.DataFrame({"id": ["a", "b", "c"], "n": [1, 2, 3]})
    assert jp.find_join_keys(q, r, [("id", "id", 1.0), ("n", "n", 1.0)])[0].left_cols == ("id",)


def test_noise_column_does_not_hijack_the_composite_key():
    """Real failure on 399a6a23: a free-text `Comments` column looks unique but
    shares a single value; growing the key greedily on uniqueness picked it."""
    combos = [(a, b, d) for a in "AB" for b in "xy" for d in ("d1", "d2", "d3")]   # 12 unique
    q = pd.DataFrame(combos, columns=["Lough", "Point", "Date"])
    q["Comments"] = [f"note-q-{i}" for i in range(len(q))]
    r = q.iloc[:9].copy()                                                          # 9 combos overlap
    r["Comments"] = [f"note-r-{i}" for i in range(len(r))]
    r.loc[r.index[0], "Comments"] = q.loc[q.index[0], "Comments"]                  # 1 coincidental match
    coma = [("Lough", "Lough", 1.0), ("Point", "Point", 1.0), ("Date", "Date", 1.0),
            ("Comments", "Comments", 1.0)]
    best = find_join_keys(q, r, coma)[0]
    assert set(best.left_cols) == {"Lough", "Point", "Date"}
    assert best.unique_side and best.shared_distinct == 9


@pytest.mark.xfail(strict=True, reason="needs testing-grounds: composite-key pruning")
def test_composite_search_prunes_supersets_of_zero_overlap_subsets():
    """Once (a, b) shares nothing, any key containing both is skipped."""
    calls = []
    real = jp.measure_key

    def spy(q, r, lc, rc, score):
        calls.append(tuple(lc))
        return real(q, r, lc, rc, score)

    # c and d overlap but are not unique, so no single column is already a key
    q = pd.DataFrame({"a": ["1", "2", "3", "4"], "b": ["x", "y", "z", "w"],
                      "c": ["p", "p", "q", "q"], "d": ["m", "n", "m", "n"]})
    r = pd.DataFrame({"a": ["9", "8", "7", "6"], "b": ["s", "t", "u", "v"],
                      "c": ["p", "p", "q", "q"], "d": ["m", "n", "m", "n"]})
    coma = [("a", "a", 1.0), ("b", "b", 1.0), ("c", "c", 1.0), ("d", "d", 1.0)]
    jp.measure_key = spy
    try:
        jp.find_join_keys(q, r, coma)
    finally:
        jp.measure_key = real
    composites = [c for c in calls if len(c) > 1]
    assert ("a", "b") in composites
    assert not any({"a", "b"} <= set(c) for c in composites if len(c) > 2)


def test_a_more_unique_but_still_non_unique_composite_is_not_proposed():
    """Crossing categorical columns raises the distinct ratio without being a key."""
    sup = [f"s{i % 40}" for i in range(200)]
    area = [f"a{i % 3}" for i in range(200)]
    q = pd.DataFrame({"Supplier Name": sup, "Service Area": area})
    ranked = find_join_keys(q, q.copy(), [("Service Area", "Service Area", 1.0),
                                           ("Supplier Name", "Supplier Name", 1.0)])
    assert all(len(e.left_cols) == 1 for e in ranked)


def test_candidates_beyond_the_old_top_ten_are_still_measured_on_small_tables():
    """Regression: a wide table pair with dozens of identically-named columns
    (all scoring 1.0) lost `BBL` to an arbitrary top-10 cut."""
    n = 30
    cols = {f"c{i:02d}": ["x"] * 5 for i in range(n)}
    q = pd.DataFrame({**cols, "BBL": [f"b{i}" for i in range(5)]})
    r = pd.DataFrame({**cols, "BBL": [f"b{i}" for i in range(5)]})
    coma = [(f"c{i:02d}", f"c{i:02d}", 1.0) for i in range(n)] + [("BBL", "BBL", 1.0)]
    ranked = find_join_keys(q, r, coma)
    assert any(e.left_cols == ("BBL",) for e in ranked)
    assert ranked[0].left_cols == ("BBL",)


def test_key_unique_on_both_sides_outranks_one_unique_on_a_single_side():
    """Gold merges use the 1:1 composite; a 3-column key that is unique on only ONE
    side fans rows out on the other."""
    q = pd.DataFrame({
        "a": ["A", "A", "B", "B"], "b": ["x", "y", "x", "y"], "c": ["1", "1", "1", "1"],
        "d": ["p", "q", "r", "s"],
    })
    r = pd.DataFrame({
        "a": ["A", "A", "A", "B", "B"], "b": ["x", "y", "y", "x", "y"], "c": ["1"] * 5,
        "d": ["p", "q", "z", "r", "s"],
    })
    coma = [("a", "a", 1.0), ("b", "b", 1.0), ("d", "d", 1.0)]
    ranked = find_join_keys(q, r, coma)
    best = ranked[0]
    assert best.cardinality != "N:N"
    assert best.left_unique_ratio >= 0.99


def test_a_key_backed_by_two_shared_values_outranks_one_backed_by_a_single_value():
    """Even when the single-value key has the better name score and is not thin
    (a two-row table can match 50% of its rows on one value)."""
    q = pd.DataFrame({"x": ["a", "b"], "y": ["m", "n"]})
    r = pd.DataFrame({"x": ["a", "c"], "y": ["m", "n"]})
    coma = [("x", "x", 1.0), ("y", "y", 0.9)]
    ranked = find_join_keys(q, r, coma)
    assert {e.left_cols[0]: e.shared_distinct for e in ranked} == {"x": 1, "y": 2}
    assert ranked[0].left_cols == ("y",)


def test_one_to_one_key_outranks_a_one_to_many_key_with_more_shared_values():
    """Unique on BOTH sides is the stronger key, even at equal name score and
    fewer shared values (a key unique on one side fans rows out on the other)."""
    q = pd.DataFrame({"A": ["a1", "a2", "a3", "a4", "a5"], "B": ["b1", "b2", "b3", "b4", "zz"]})
    r = pd.DataFrame({"A": ["a1", "a1", "a2", "a3", "a4", "a5"], "B": ["b1", "b2", "b3", "b4", "y1", "y2"]})
    coma = [("A", "A", 1.0), ("B", "B", 1.0)]
    ranked = find_join_keys(q, r, coma)
    by = {e.left_cols[0]: e for e in ranked}
    assert by["A"].cardinality == "1:N" and by["A"].shared_distinct == 5
    assert by["B"].cardinality == "1:1" and by["B"].shared_distinct == 4
    assert ranked[0].left_cols == ("B",)


def test_a_97_percent_unique_id_beats_a_person_name_column():
    """OrQa organogram pairs: `Post Unique Reference` repeats for a few vacant posts
    (97% unique) and must not lose to `Name` just because Name crosses a hard 99% cliff."""
    ids_q = [f"p{i}" for i in range(97)] + ["p0", "p1", "p2"]
    ids_r = [f"p{i}" for i in range(5, 102)] + ["p5", "p6", "p7"]
    q = pd.DataFrame({"Post Unique Reference": ids_q, "Name": [f"n{i}" for i in range(100)]})
    r = pd.DataFrame({"Post Unique Reference": ids_r, "Name": [f"n{i % 74 + 30}" for i in range(100)]})
    coma = [("Name", "Name", 1.0), ("Post Unique Reference", "Post Unique Reference", 1.0)]
    ranked = find_join_keys(q, r, coma)
    by = {e.left_cols[0]: e for e in ranked}
    assert by["Post Unique Reference"].left_unique_ratio < 0.99         # the cliff case
    assert by["Name"].left_unique_ratio == 1.0 and by["Name"].shared_distinct >= 10
    assert ranked[0].left_cols == ("Post Unique Reference",)


def test_value_overlap_outranks_name_similarity_between_equally_key_like_columns():
    """OrQa CSP tables: `CSP23CD=CSP22CD` (name score 0.667, 100% overlap) is the join;
    `BNG_E=BNG_E` (name score 1.0) only overlaps 73%."""
    codes = [f"E{i:05d}" for i in range(100)]
    q = pd.DataFrame({"CSP23CD": codes, "BNG_E": [400000 + i * 7 for i in range(100)]})
    r = pd.DataFrame({"CSP22CD": codes, "BNG_E": [400000 + i * 7 for i in range(27, 127)]})
    coma = [("BNG_E", "BNG_E", 1.0), ("CSP23CD", "CSP22CD", 0.667)]
    ranked = find_join_keys(q, r, coma)
    by = {e.left_cols[0]: e for e in ranked}
    assert by["BNG_E"].cardinality == by["CSP23CD"].cardinality == "1:1"
    assert ranked[0].left_cols == ("CSP23CD",)


# ---- defects found by evaluating on OrQa's NYC/UK matches -------------------------------------

def test_comma_formatted_text_numbers_match_float_numbers():
    """NYC: BIN is '2,012,938' (text) in one table and 2012938.0 (float) in another --
    the same identifier; reporting no overlap would be a false 'no relationship'."""
    q = pd.DataFrame({"BIN": ["2,012,938", "2,093,291", "2,048,821"]})
    r = pd.DataFrame({"BIN": [2012938.0, 2093291.0, 2103161.0]})
    ev = _m(q, r, "BIN", "BIN")
    assert ev.shared_distinct == 2 and ev.verdict == JOIN
    assert not any("dtype" in n for n in ev.notes)


def test_comma_formatted_text_matches_plain_text_numbers():
    q = pd.DataFrame({"BBL": ["2,023,157,501", "2,023,647,501", "2,023,697,502"]})
    r = pd.DataFrame({"BBL": ["2023157501", "2023647501", "2099999999"]})
    ev = _m(q, r, "BBL", "BBL")
    assert ev.shared_distinct == 2 and ev.verdict == JOIN


def test_non_numeric_text_with_commas_is_not_coerced():
    q = pd.DataFrame({"place": ["Leeds, UK", "York, UK"]})
    r = pd.DataFrame({"place": ["leeds, uk", "hull, uk"]})
    ev = _m(q, r, "place", "place")
    assert ev.shared_distinct == 1


def test_a_key_single_valued_on_only_one_side_is_a_filter_join_not_a_cross_product():
    """NYC `Year=year`: one table holds only 2022, the other 11 years.  The join is
    not R1 x R2 (paper: R1 join R2 != R1 x R2) -- it selects the 2022 rows."""
    q = pd.DataFrame({"Year": [2019, 2020, 2021, 2022, 2022, 2023]})
    r = pd.DataFrame({"year": [2022] * 5})
    ev = _m(q, r, "Year", "year")
    assert ev.verdict == JOIN and ev.shared_distinct == 1
    assert any("single-valued" in n and "filter" in n for n in ev.notes)


def test_a_key_single_valued_on_both_sides_is_still_a_cross_product():
    q = pd.DataFrame({"d": ["Cabinet Office"] * 4})
    r = pd.DataFrame({"d": ["Cabinet Office"] * 3})
    assert _m(q, r, "d", "d").verdict == CONSTANT_KEY


def test_single_valued_one_side_key_ranks_below_a_real_key():
    q = pd.DataFrame({"Year": [2019, 2020, 2021, 2022, 2022, 2023], "id": list("abcdef")})
    r = pd.DataFrame({"year": [2022] * 5 + [2022], "id": list("abcdxy")})
    coma = [("Year", "year", 1.0), ("id", "id", 0.9)]
    assert find_join_keys(q, r, coma)[0].left_cols == ("id",)


# ---- report -----------------------------------------------------------------------------------

def _report(q, r, coma, left="A", right="B"):
    return format_join_section(left, right, find_join_keys(q, r, coma), q, r)


def test_report_states_measured_facts_for_a_supported_join():
    q = pd.DataFrame({"id": ["a", "b"]})
    r = pd.DataFrame({"id": ["a", "a", "b"]})
    text = _report(q, r, [("id", "id", 1.0)], "schools.parquet", "attendance.parquet")
    assert text.startswith("JOIN: 'schools.parquet' joins 'attendance.parquet'")
    assert "1. id (schools.parquet) = id (attendance.parquet)" in text
    assert "1:N" in text and "100% of the left table's rows" in text and "inner join = 3 rows" in text


def test_report_lists_at_most_three_ranked_candidates():
    cols = {f"k{i}": [f"v{i}-{j}" for j in range(5)] for i in range(6)}
    q, r = pd.DataFrame(cols), pd.DataFrame(cols)
    text = _report(q, r, [(c, c, 1.0) for c in cols])
    assert all(f"  {n}. " in text for n in (1, 2, 3))
    assert "  4. " not in text


def test_report_shows_notes_such_as_the_n_to_n_warning():
    q = pd.DataFrame({"k": ["a", "a", "b", "b"]})
    r = pd.DataFrame({"k": ["a", "a", "a", "b"]})
    assert "aggregate one side to the key first" in _report(q, r, [("k", "k", 1.0)])


def test_no_join_report_shows_sample_values_from_both_sides():
    """NYC: `Borough` is 'Brooklyn' in one table and the code 'K' in the other -- one glance at
    the values tells the coder why the join is empty."""
    q = pd.DataFrame({"Borough": ["Brooklyn", "Queens", "Manhattan"], "n": [1, 2, 3]})
    r = pd.DataFrame({"Borough": ["K", "Q", "M"], "n": [9, 8, 7]})
    text = _report(q, r, [("Borough", "Borough", 1.0)], "left.parquet", "right.parquet")
    assert text.startswith("NO JOIN:")
    assert "'Brooklyn'" in text and "'K'" in text
    assert "left.parquet" in text.split("values:")[1] and "right.parquet" in text.split("values:")[1]


def test_no_join_report_says_an_outer_join_needs_no_overlap_and_names_the_limit():
    """An executed, judge-approved `how='outer'` merge had no overlap, so 'no shared values' must
    not read as 'these tables cannot be combined'; and only name-similar columns are tested."""
    q = pd.DataFrame({"Date": ["2020-01-01"]})
    r = pd.DataFrame({"Date": ["2021-01-01"]})
    text = _report(q, r, [("Date", "Date", 1.0)])
    assert "outer join" in text and "name-similar" in text
    assert "RELATIONSHIP" not in text


def test_report_with_no_name_similar_pair_says_no_values_were_compared():
    q, r = pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]})
    text = format_join_section("A", "B", find_join_keys(q, r, [("a", "b", 0.2)]), q, r)
    assert text.startswith("NO JOIN:") and "no values were compared" in text


def test_sample_values_are_truncated():
    q = pd.DataFrame({"k": ["x" * 80]})
    r = pd.DataFrame({"k": ["y" * 80]})
    text = _report(q, r, [("k", "k", 1.0)])
    assert "x" * 30 not in text and "..." in text


def test_blank_cells_are_not_a_shared_key_value():
    """Two tables that both have empty cells do not 'share' the empty string."""
    q = pd.DataFrame({"k": ["a", "", "  "]})
    r = pd.DataFrame({"k": ["b", "", " "]})
    ev = _m(q, r, "k", "k")
    assert ev.shared_distinct == 0 and ev.verdict == NO_OVERLAP


def test_boolean_columns_match_their_text_form():
    q = pd.DataFrame({"flag": [True, False, True]})
    r = pd.DataFrame({"flag": ["True", "False", "false"]})
    ev = _m(q, r, "flag", "flag")
    assert ev.shared_distinct == 2 and not any("dtype" in n for n in ev.notes)


def test_a_column_of_nested_values_is_a_noted_non_result_not_a_crash():
    """Parquet allows list-valued columns; they are unhashable and cannot be compared. One such
    column must not abort the whole report -- it becomes a candidate with a note."""
    q = pd.DataFrame({"k": [[1], [2], [3]], "id": ["a", "b", "c"]})
    r = pd.DataFrame({"k": [[1], [2], [3]], "id": ["a", "b", "c"]})
    ranked = find_join_keys(q, r, [("k", "k", 1.0), ("id", "id", 1.0)])
    by = {e.left_cols[0]: e for e in ranked}
    assert by["id"].verdict == JOIN                       # the comparable column is unaffected
    assert by["k"].verdict == NO_OVERLAP and any("nested" in n for n in by["k"].notes)


def test_a_join_whose_candidates_are_all_many_to_many_says_so_in_the_header():
    """Same-schema monthly files share categorical values (payment type, vendor) on every row: a
    bare 'JOIN:' headline would invite a 21-billion-row merge on them."""
    q = pd.DataFrame({"vendor": ["a", "b"] * 20, "payment": ["x", "y", "z", "x"] * 10})
    r = pd.DataFrame({"vendor": ["a", "b"] * 20, "payment": ["x", "y", "z", "x"] * 10})
    text = _report(q, r, [("vendor", "vendor", 1.0), ("payment", "payment", 1.0)])
    assert text.startswith("JOIN (many-to-many only):")
    assert "aggregat" in text.split("\n")[0]


def test_a_join_with_a_unique_side_keeps_the_plain_header():
    q = pd.DataFrame({"id": ["a", "b", "c"]})
    r = pd.DataFrame({"id": ["a", "b", "c", "c"]})
    assert _report(q, r, [("id", "id", 1.0)]).startswith("JOIN: 'A' joins 'B'")


def test_report_names_each_file_only_where_the_caller_must_copy_it():
    """Real UK file names are ~75 characters; repeating them in every fact line would blow the tool
    budget. The key line carries both names (the architect copies them into requirements.joins);
    the facts say left/right."""
    name_l = "05af53dc-891a-4598-8d8b-7f9c70fa6df3___9591a8f3-e83f-4fc9-8c0c-1b2e92a5f5bc.parquet"
    name_r = "ee3cd9c1-ae38-4b2f-81e6-2332c4799f3f___dfce03d1-98fc-4f5c-b355-08c2fdddeee2.parquet"
    q = pd.DataFrame({"id": ["a", "b", "c"]})
    r = pd.DataFrame({"id": ["a", "b", "c"]})
    text = _report(q, r, [("id", "id", 1.0)], name_l, name_r)
    key_line = next(line for line in text.splitlines() if line.startswith("  1. "))
    assert name_l in key_line and name_r in key_line
    assert text.count(name_l) == 2 and text.count(name_r) == 2     # head + key line, nowhere else
    assert "of the left table's rows" in text and "of the right table's rows" in text


def test_a_key_single_valued_on_one_side_ranks_below_a_real_overlap():
    """Real report (DWP organograms): `Salary Cost of Reports (£)` shared ONE value because one file
    held a single value, yet outranked categorical columns with genuine overlap -- a filter-style
    join is not a link between entities and must sink below keys that both sides vary on."""
    q = pd.DataFrame({"cat": ["a", "b", "c", "d"] * 10, "amount": range(100, 140)})
    r = pd.DataFrame({"cat": ["a", "b", "c", "d"] * 10, "amount": [107] * 40})
    ranked = find_join_keys(q, r, [("amount", "amount", 1.0), ("cat", "cat", 1.0)])
    by = {e.left_cols[0]: e for e in ranked}
    assert by["amount"].verdict == JOIN and by["amount"].shared_distinct == 1
    assert any("single-valued on one side" in n for n in by["amount"].notes)
    assert ranked[0].left_cols == ("cat",)


def test_a_single_valued_side_key_still_leads_when_it_is_the_only_candidate():
    """Year = year (one table holds only 2022) is a legitimate filter join; demoting is a ranking
    matter, never a rejection."""
    q = pd.DataFrame({"Year": [2019, 2020, 2021, 2022, 2022, 2023]})
    r = pd.DataFrame({"year": [2022] * 5})
    ranked = find_join_keys(q, r, [("Year", "year", 1.0)])
    assert ranked[0].verdict == JOIN


# ---- columns matched on their values --------------------------------------------------------------

def _by_values(left, right, score=0.9):
    from lakegen.tools.shared.join_union.schema_matching import ColumnMatch

    return ColumnMatch(left, right, score, 0.0, score / 0.9)


def _ids(n, prefix="Z"):
    return [f"{prefix}{i:03d}" for i in range(n)]


def test_a_renamed_identifier_is_kept_when_its_values_really_overlap_and_says_so():
    q = pd.DataFrame({"zona": _ids(60)})
    r = pd.DataFrame({"col_0": _ids(60)})
    ranked = find_join_keys(q, r, [_by_values("zona", "col_0")])
    assert ranked and ranked[0].verdict == JOIN
    assert any("named differently" in note for note in ranked[0].notes)


def test_a_pair_matched_on_values_with_too_few_shared_values_is_dropped():
    q = pd.DataFrame({"a": _ids(60)})
    r = pd.DataFrame({"b": _ids(5) + _ids(55, "Y")})
    assert find_join_keys(q, r, [_by_values("a", "b")]) == []


def test_a_pair_matched_on_values_where_most_rows_find_no_partner_is_dropped():
    q = pd.DataFrame({"a": _ids(200)})
    r = pd.DataFrame({"b": _ids(30) + _ids(170, "Y")})
    assert find_join_keys(q, r, [_by_values("a", "b")]) == []


def test_counts_that_overlap_in_value_are_not_a_key_when_neither_side_is_unique():
    q = pd.DataFrame({"immigrati": [i % 50 for i in range(500)]})
    r = pd.DataFrame({"emigrati": [(i * 3) % 50 for i in range(500)]})
    assert find_join_keys(q, r, [_by_values("immigrati", "emigrati")]) == []
    # the same pair matched on its NAMES is still measured and reported
    assert find_join_keys(q, r, [("immigrati", "emigrati", 0.74)])


def test_row_numbers_matched_on_values_are_not_a_key():
    q = pd.DataFrame({"OBJECTID": range(1, 101)})
    r = pd.DataFrame({"ref": range(1, 101)})
    assert find_join_keys(q, r, [_by_values("OBJECTID", "ref")]) == []
