"""How big the merge of two tables on a key is, and whether it is empty, over every row of the tables.

The real failures behind these tests (OrQa lakes, executed reference merges): a many-to-many key of two
19,763 x 90,164-row tables was counted as 197 million rows from a 20,000-row sample when the merge has 808
million; a plain pandas merge of two NYC tables on `Grade` needed 21 GB; a date column against a year column
could not be merged at all; missing keys met each other and turned 482 rows into 47,134.
"""
import pandas as pd
import pytest

from lakegen.domain.merge_limits import EMPTY, EXPLODES, FANOUT, SAFE, classify_merge
from lakegen.tools.shared.join_union import full_scan
from lakegen.tools.shared.join_union.full_scan import TableSide, refine_on_whole_tables
from lakegen.tools.shared.join_union.join_keys import (
    NO_OVERLAP,
    add_raw_merge_facts,
    measure_key,
    raw_merge_rows,
)
from lakegen.tools.shared.join_union.join_report import format_join_section
from lakegen.tools.shared.join_union.key_search import find_join_keys
from lakegen.tools.shared.join_union.layout import ESSENTIAL, EXTRA, HEADLINE, fit_lines


def _measure(q, r, lc="k", rc="k", **kw):
    return measure_key(q, r, [lc], [rc], 1.0, **kw)


# ---- the size classes -------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("rows", "left", "right", "expected"),
    [
        (0, 10, 10, EMPTY),
        (10, 10, 10, SAFE),
        (11, 10, 10, FANOUT),
        (49_999, 100, 100, FANOUT),                 # 500x, but under the floor the sandbox ignores
        (50_001, 100, 100, EXPLODES),
        (1_000_000, 40_000, 30_000, FANOUT),        # exactly 25x the larger table is let through
        (1_000_001, 40_000, 30_000, EXPLODES),      # one row more is stopped
        (1_000_000, 39_000, 38_000, EXPLODES),      # 25.6x
        (1_000_000, 50_000, 40_000, FANOUT),        # 20x
        (1_000_000, 90_000, 5_000, FANOUT),         # 11x the larger table
    ],
)
def test_the_size_class_follows_the_rule_the_sandbox_applies(rows, left, right, expected):
    assert classify_merge(rows, left, right) == expected


# ---- sizes over the whole tables --------------------------------------------------------------------

def test_a_one_to_many_join_has_the_rows_of_the_many_side_and_is_safe():
    q = pd.DataFrame({"k": ["a", "b", "c"]})
    r = pd.DataFrame({"k": ["a", "a", "b", "c", "c"]})
    ev = _measure(q, r)
    assert (ev.inner_total, ev.left_join_total, ev.outer_join_total) == (5, 5, 5)
    assert ev.size_class == SAFE and ev.exact


def test_left_and_outer_joins_keep_the_unmatched_rows_and_the_missing_keys():
    q = pd.DataFrame({"k": ["a", "b", "x", None]})
    r = pd.DataFrame({"k": ["a", "a", "y", "z"]})
    ev = _measure(q, r)
    assert ev.inner_total == 2                   # a x a, a x a
    assert ev.left_join_total == 2 + 3           # b, x and the missing key keep a row each
    assert ev.outer_join_total == 5 + 2          # y and z keep a row each


def test_a_join_that_shares_nothing_is_empty():
    ev = _measure(pd.DataFrame({"k": ["a", "b"]}), pd.DataFrame({"k": ["x", "y"]}))
    assert ev.verdict == NO_OVERLAP and ev.size_class == EMPTY and ev.inner_total == 0


def test_a_many_to_many_key_on_two_thousand_rows_explodes_and_says_what_the_aggregated_join_has():
    q = pd.DataFrame({"k": ["a", "b"] * 500})
    r = pd.DataFrame({"k": ["a", "b"] * 500})
    ev = _measure(q, r)
    assert ev.inner_total == 500_000 and ev.size_class == EXPLODES and ev.grouped_rows == 2
    assert ev.fanout == 500
    note = next(n for n in ev.notes if n.startswith("EXPLODES"))
    assert "500x the larger table" in note and "that join has 2 rows" in note


def test_sizes_of_a_sample_are_scaled_up_to_the_whole_tables_and_marked_as_estimates():
    """One row per key in each sample; the tables repeat each key 10 and 5 times: 100 x 10 x 5 rows."""
    q = pd.DataFrame({"k": [f"k{i}" for i in range(100)]})
    r = pd.DataFrame({"k": [f"k{i}" for i in range(100)]})
    ev = _measure(q, r, left_total=1_000, right_total=500)
    assert ev.inner_rows == 100                  # what the sample holds
    assert ev.inner_total == 5_000 and not ev.exact
    text = format_join_section("A", "B", [ev], q, r)
    assert "inner join = ~5,000 rows" in text


def test_counting_the_whole_tables_does_not_overflow_on_a_key_shared_by_millions_of_rows():
    q = pd.DataFrame({"k": ["a"] * 3_000_000})
    ev = _measure(q, q.copy())
    assert ev.inner_total == 9_000_000_000_000 and ev.size_class == EXPLODES


# ---- ranking and the report -------------------------------------------------------------------------

def test_a_key_the_sandbox_would_stop_ranks_below_one_it_lets_through():
    q = pd.DataFrame({"id": range(200), "year": [2020 + i % 2 for i in range(200)]})
    r = pd.DataFrame({"id": range(200), "year": [2020 + i % 2 for i in range(200)]})
    r.loc[:, "id"] = list(range(150)) + [149] * 50          # not unique, but nowhere near a blow-up
    ranked = find_join_keys(q, r, [("year", "year", 1.0), ("id", "id", 0.9)])
    assert ranked[0].left_cols == ("id",)


def test_the_report_says_which_keys_the_sandbox_would_stop_and_how_big_the_aggregated_join_is():
    q = pd.DataFrame({"year": [2020, 2021] * 500, "id": range(1000)})
    r = pd.DataFrame({"year": [2020, 2021] * 500, "id": range(1000)})
    text = format_join_section("A", "B", find_join_keys(q, r, [("year", "year", 1.0), ("id", "id", 1.0)]), q, r)
    year_block = text.split("year (A)")[1].split("2. ")[0] if "2. " in text else text.split("year (A)")[1]
    assert "EXPLODES" in year_block and "that join has 2 rows" in year_block


def test_the_left_join_is_shown_only_when_it_differs_from_the_inner_join():
    q = pd.DataFrame({"k": ["a", "b", "c", "d"]})
    r = pd.DataFrame({"k": ["a", "b"]})
    assert "(left join 4)" in format_join_section("A", "B", find_join_keys(q, r, [("k", "k", 1.0)]), q, r)
    same = pd.DataFrame({"k": ["a", "b", "c", "d"]})
    assert "left join" not in format_join_section("A", "B", find_join_keys(q, same, [("k", "k", 1.0)]), q, same)


def test_the_other_shared_columns_are_listed_smallest_domain_first_without_junk():
    n = 400
    q = pd.DataFrame({
        "id": [f"i{i}" for i in range(n)], "u": [f"u{i}" for i in range(n)], "w": [f"w{i}" for i in range(n)],
        "year": [2000 + i % 5 for i in range(n)], "district": [f"d{i % 40}" for i in range(n)],
        "flag": ["x"] * n, "amount": [i / 7 for i in range(n)],
    })
    r = q.copy()
    coma = [(c, c, 1.0) for c in q.columns]
    text = format_join_section("A", "B", find_join_keys(q, r, coma), q, r)
    line = next(line for line in text.splitlines() if line.strip().startswith("also shared"))
    assert line.index("year") < line.index("district")
    assert "flag" not in line and "amount" not in line     # one value on each side; a measurement


def test_the_shared_repeated_columns_are_also_measured_together_as_the_grain():
    q = pd.DataFrame({"year": [2020, 2021] * 250, "sex": ["f", "m", "f", "m", "f"] * 100})
    r = pd.DataFrame({"year": [2020, 2021] * 250, "sex": ["f", "m", "f", "m", "f"] * 100})
    ranked = find_join_keys(q, r, [("year", "year", 1.0), ("sex", "sex", 1.0)])
    grain = next(ev for ev in ranked if ev.grain)
    assert grain.left_cols == ("sex", "year") or grain.left_cols == ("year", "sex")
    assert ranked[-1] is grain                                   # never ranked among the keys
    text = format_join_section("A", "B", ranked, q, r)
    assert "all the shared repeated columns together" in text


# ---- the merge as stored ----------------------------------------------------------------------------

def test_text_is_compared_exactly_by_a_plain_merge_and_normalised_by_the_check():
    q = pd.DataFrame({"k": ["Brooklyn", "Queens"]})
    r = pd.DataFrame({"k": ["brooklyn ", "QUEENS"]})
    assert raw_merge_rows(q, r, ["k"], ["k"]) == 0
    ev = _measure(q, r)
    add_raw_merge_facts(ev, q, r)
    assert ev.inner_total == 2 and ev.raw_inner_total == 0
    assert "a plain merge returns 0 rows" in format_join_section("A", "B", [ev], q, r)


def test_pandas_refuses_to_merge_text_with_numbers_and_the_report_says_how_to_fix_it():
    q = pd.DataFrame({"annee": ["2018", "2019"], "x": [1, 2]})
    r = pd.DataFrame({"annee": [2018, 2019], "y": [3, 4]})
    assert raw_merge_rows(q, r, ["annee"], ["annee"]) is None
    ev = _measure(q, r, "annee", "annee")
    add_raw_merge_facts(ev, q, r)
    text = format_join_section("A", "B", [ev], q, r)
    assert "cannot be merged" in text and "pd.to_numeric" in text


def test_missing_keys_match_each_other_in_pandas_and_the_report_warns_of_the_extra_rows():
    q = pd.DataFrame({"k": ["a", "b", "c"] + [None] * 300})
    r = pd.DataFrame({"k": ["a", "b", "c"] + [None] * 300})
    assert raw_merge_rows(q, r, ["k"], ["k"]) == 3 + 300 * 300
    ev = _measure(q, r)
    add_raw_merge_facts(ev, q, r)
    assert ev.inner_total == 3
    text = format_join_section("A", "B", [ev], q, r)
    assert "a plain merge returns 90,003 rows, not 3 (the sandbox stops it)" in text
    assert "dropna" in text


def test_numbers_that_are_stored_as_numbers_need_no_warning():
    q = pd.DataFrame({"k": [1, 2, 3]})
    r = pd.DataFrame({"k": [1.0, 2.0, 3.0]})
    ev = _measure(q, r)
    add_raw_merge_facts(ev, q, r)
    assert ev.raw_inner_total == 3
    assert "as stored" not in format_join_section("A", "B", [ev], q, r)


# ---- counting the whole tables ----------------------------------------------------------------------

def _write(path, frame, row_group_size=1_000):
    frame.to_parquet(path, row_group_size=row_group_size)
    return path


def _side(path, name, rows=1_000):
    from lakegen.integrations.table_io import read_table_sample

    sample, total = read_table_sample(path, rows)
    return TableSide(name, path, sample, total)


def test_the_best_keys_are_counted_over_every_row_not_over_the_sample(tmp_path):
    n = 20_000
    left = pd.DataFrame({"k": [f"k{i % 4}" for i in range(n)]})
    right = pd.DataFrame({"k": [f"k{i % 4}" for i in range(n)]})
    a, b = _side(_write(tmp_path / "a.parquet", left), "a"), _side(_write(tmp_path / "b.parquet", right), "b")
    assert a.sampled and b.sampled
    keys = find_join_keys(a.sample, b.sample, [("k", "k", 1.0)], None, None, a.total_rows, b.total_rows)
    assert not keys[0].exact                           # scaled up from the sample

    refined = refine_on_whole_tables(keys, a, b)

    expected = len(left.merge(right, on="k"))
    assert refined[0].exact and refined[0].inner_total == expected == 4 * 5_000 * 5_000
    assert refined[0].size_class == EXPLODES


def test_a_rare_shared_value_the_sample_missed_is_found_in_the_whole_table(tmp_path):
    left = pd.DataFrame({"k": [f"l{i}" for i in range(20_000)]})
    right = pd.DataFrame({"k": [f"r{i}" for i in range(20_000)]})
    left.loc[0, "k"] = "shared"                        # in the first row group, outside the middle one
    right.loc[0, "k"] = "shared"
    a, b = _side(_write(tmp_path / "a.parquet", left), "a"), _side(_write(tmp_path / "b.parquet", right), "b")
    keys = find_join_keys(a.sample, b.sample, [("k", "k", 1.0)], None, None, a.total_rows, b.total_rows)
    assert keys[0].verdict == NO_OVERLAP               # not in the rows read

    refined = refine_on_whole_tables(keys, a, b)

    assert refined[0].shared_distinct == 1 and refined[0].inner_total == 1


def test_tables_that_fit_in_the_sample_are_not_read_again(tmp_path):
    left = pd.DataFrame({"k": ["a", "b"]})
    a, b = _side(_write(tmp_path / "a.parquet", left), "a"), _side(_write(tmp_path / "b.parquet", left), "b")
    keys = find_join_keys(a.sample, b.sample, [("k", "k", 1.0)])
    assert refine_on_whole_tables(keys, a, b) is keys


def test_a_table_beyond_the_scan_limit_is_read_as_a_larger_sample_and_scaled(tmp_path, monkeypatch):
    monkeypatch.setattr(full_scan, "FULL_SCAN_MAX_CELLS", 5_000)
    monkeypatch.setattr(full_scan, "SCALED_SAMPLE_CELLS", 4_000)
    left = pd.DataFrame({"k": [f"k{i % 10}" for i in range(10_000)]})
    a, b = _side(_write(tmp_path / "a.parquet", left), "a"), _side(_write(tmp_path / "b.parquet", left), "b")
    keys = find_join_keys(a.sample, b.sample, [("k", "k", 1.0)], None, None, a.total_rows, b.total_rows)

    refined = refine_on_whole_tables(keys, a, b)

    assert not refined[0].exact
    assert abs(refined[0].inner_total - 10 * 1_000 * 1_000) / (10 * 1_000 * 1_000) < 0.2


def test_the_scan_stops_when_its_time_is_spent_and_the_rest_keep_their_scaled_sizes(tmp_path):
    n = 20_000
    frame = pd.DataFrame({"a": [f"a{i % 4}" for i in range(n)], "b": [f"b{i % 5}" for i in range(n)]})
    a, b = _side(_write(tmp_path / "a.parquet", frame), "a"), _side(_write(tmp_path / "b.parquet", frame), "b")
    keys = find_join_keys(a.sample, b.sample, [("a", "a", 1.0), ("b", "b", 1.0)], None, None, a.total_rows, b.total_rows)
    ticks = iter([0.0, 1.0, 99.0, 99.0, 99.0, 99.0])

    refined = refine_on_whole_tables(keys, a, b, seconds=10, clock=lambda: next(ticks))

    assert sum(1 for ev in refined if ev.exact) == 1


def test_a_key_that_could_not_finish_in_the_time_left_is_not_started(tmp_path, monkeypatch):
    n = 20_000
    frame = pd.DataFrame({"a": [f"a{i % 4}" for i in range(n)]})
    a, b = _side(_write(tmp_path / "a.parquet", frame), "a"), _side(_write(tmp_path / "b.parquet", frame), "b")
    keys = find_join_keys(a.sample, b.sample, [("a", "a", 1.0)], None, None, a.total_rows, b.total_rows)
    monkeypatch.setattr(full_scan, "SECONDS_PER_CELL", 1.0)          # 40,000 cells would take 11 hours

    assert refine_on_whole_tables(keys, a, b, seconds=60)[0].exact is False


def test_an_unreadable_key_column_leaves_the_sampled_evidence(tmp_path):
    frame = pd.DataFrame({"k": [f"k{i % 4}" for i in range(20_000)]})
    a, b = _side(_write(tmp_path / "a.parquet", frame), "a"), _side(_write(tmp_path / "b.parquet", frame), "b")
    keys = find_join_keys(a.sample, b.sample, [("k", "k", 1.0)], None, None, a.total_rows, b.total_rows)
    (tmp_path / "b.parquet").unlink()

    assert refine_on_whole_tables(keys, a, b)[0].exact is False


# ---- the room of a section --------------------------------------------------------------------------

def test_the_least_important_lines_go_first_when_a_section_is_too_long():
    lines = [(HEADLINE, "head"), (ESSENTIAL, "key one"), (EXTRA, "x" * 100), (ESSENTIAL, "key two")]
    assert fit_lines(lines, 10_000) == "head\nkey one\n" + "x" * 100 + "\nkey two"
    assert fit_lines(lines, 30) == "head\nkey one\nkey two"
    assert fit_lines(lines, 1) == "head\nkey one"    # the headline and the best key always stay


def test_a_headline_is_never_dropped():
    assert fit_lines([(HEADLINE, "a" * 50), (EXTRA, "b")], 10) == "a" * 50


def test_a_join_section_with_very_long_names_still_fits_its_budget():
    long = "Number of residents who received financial counseling " * 3
    q = pd.DataFrame({long + str(i): [f"v{j}" for j in range(30)] for i in range(10)})
    r = q.copy()
    ranked = find_join_keys(q, r, [(c, c, 1.0) for c in q.columns])
    text = format_join_section("A", "B", ranked, q, r, limit=1_500)
    assert len(text) <= 1_500 or text.count("\n") <= 3
    assert text.startswith("JOIN")
