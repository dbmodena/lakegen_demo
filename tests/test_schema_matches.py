import pandas as pd

from lakegen.agent_tools import schema_matching, tools_p2


# ---- match_columns: COMA only proposes candidate pairs -----------------------------------------

def test_match_columns_gives_valentine_a_bounded_sample(monkeypatch):
    """Schema-only COMA never reads values, so it is given at most COMA_SAMPLE_ROWS rows: identical
    output on 1000 rows as on a 19M-row table (62 s vs 0.0 s measured)."""
    calls = []

    def fake_valentine_match(dfs, matcher, **kwargs):
        calls.append(([len(df) for df in dfs], kwargs))
        return {}

    monkeypatch.setattr(schema_matching, "valentine_match", fake_valentine_match)

    big = pd.DataFrame({"a": range(5000)})
    schema_matching.match_columns(big, big.rename(columns={"a": "b"}))

    assert calls == [([1000, 1000], {"instance_sample_size": 1000})]


def test_verify_pair_schema_ranks_matches_best_first(monkeypatch):
    monkeypatch.setattr(
        schema_matching,
        "match_columns",
        lambda *_args: {("a", "x"): 0.4, ("b", "y"): 0.9, ("c", "z"): 0.6},
    )

    evidence = schema_matching.verify_pair_schema(pd.DataFrame({"a": [1]}), pd.DataFrame({"x": [1]}))

    assert evidence["matches"] == [("b", "y", 0.9), ("c", "z", 0.6), ("a", "x", 0.4)]
    assert evidence["sm_micro_avg"] == 0.9


# ---- check_join_union: measured facts, not a name-score verdict --------------------------------

def test_check_join_union_states_that_same_kind_tables_union(tmp_path):
    rows = range(50)
    pd.DataFrame(
        {
            "borough": ["Bronx"] * 50,
            "school_id": [f"S{i}" for i in rows],
            "year": [2020] * 50,
            "enrollment": list(rows),
        }
    ).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame(
        {
            "borough": ["Queens"] * 50,
            "school_id": [f"T{i}" for i in rows],
            "year": [2021] * 50,
            "enrollment": list(rows),
        }
    ).to_parquet(tmp_path / "right.parquet")

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert result.startswith("Join/union check between 'left.parquet' and 'right.parquet'")
    assert "UNION: 'left.parquet' unions with 'right.parquet'; 4 columns map 1-1" in result
    assert "borough -> borough" in result
    # every same-named pair is a name match, but none of them shares a usable key value
    assert "NO JOIN:" in result
    assert "schema" not in result.casefold()


def test_check_join_union_states_the_join_key_with_measured_facts(tmp_path):
    rows = range(50)
    pd.DataFrame(
        {"school_id": [f"S{i}" for i in rows], "enrollment": list(rows)}
    ).to_parquet(tmp_path / "schools.parquet")
    pd.DataFrame(
        {"School ID": [f"S{i}" for i in rows], "attendance_rate": [0.9] * 50}
    ).to_parquet(tmp_path / "attendance.parquet")

    result = tools_p2._check_join_union(tmp_path, "schools.parquet", "attendance.parquet")

    assert "JOIN: 'schools.parquet' joins 'attendance.parquet'" in result
    assert "school_id (schools.parquet) = School ID (attendance.parquet)" in result
    assert "1:1, 50 shared key values" in result and "inner join = 50 rows" in result


def test_check_join_union_states_join_without_union(monkeypatch, tmp_path):
    ids = list(range(10, 60))
    pd.DataFrame({"id": ids, "b": ["p"] * 50, "c": ["q"] * 50}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"key": ids, "y": ["r"] * 50, "z": ["s"] * 50}).to_parquet(tmp_path / "right.parquet")
    monkeypatch.setattr(
        schema_matching,
        "match_columns",
        lambda *_args: {("id", "key"): 0.9, ("b", "y"): 0.1, ("c", "z"): 0.1},
    )

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert "JOIN: 'left.parquet' joins 'right.parquet'" in result
    assert "id (left.parquet) = key (right.parquet)" in result
    assert "NO UNION: only 1 of 3 / 3 columns" in result


def test_check_join_union_no_longer_trusts_a_name_match_without_shared_values(tmp_path):
    """The old rule called two tables with a same-named `Date` column a JOIN (48% of unrelated
    pairs). Same name, no shared value: the report says so and shows the values."""
    pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "n": [1, 2]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"Date": ["2021-05-05", "2021-05-06"], "m": [3, 4]}).to_parquet(tmp_path / "right.parquet")

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert "NO JOIN:" in result
    assert "'2020-01-01'" in result and "'2021-05-05'" in result
    assert "NO UNION:" in result


def test_check_join_union_states_no_join_and_no_union_without_name_similar_columns(monkeypatch, tmp_path):
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "right.parquet")
    monkeypatch.setattr(schema_matching, "match_columns", lambda *_args: {("a", "x"): 0.2})

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert "NO JOIN:" in result and "no values were compared" in result
    assert "NO UNION:" in result
    assert "RELATIONSHIP" not in result


def test_check_join_union_runs_valentine_for_the_same_file(monkeypatch, tmp_path):
    pd.DataFrame({"identifier": [1, 2], "name": ["one", "two"]}).to_parquet(
        tmp_path / "table.parquet"
    )
    calls = []

    def fake_match(q, r):
        calls.append((list(q.columns), list(r.columns)))
        return {("identifier", "identifier"): 1.0, ("name", "name"): 1.0}

    monkeypatch.setattr(schema_matching, "match_columns", fake_match)

    result = tools_p2._check_join_union(tmp_path, "table.parquet", "table.parquet")

    assert calls == [(["identifier", "name"], ["identifier", "name"])]
    assert "identifier (table.parquet) = identifier (table.parquet)" in result
    assert "UNION: 'table.parquet' unions with 'table.parquet'" in result


def test_check_join_union_measures_every_column_of_a_wide_table(monkeypatch, tmp_path):
    wide = pd.DataFrame({f"c{i}": range(300) for i in range(25)})
    wide.to_parquet(tmp_path / "wide.parquet")
    shapes = []

    def fake_match(q, r):
        shapes.append((q.shape, r.shape))
        return {("c0", "c0"): 1.0}

    monkeypatch.setattr(schema_matching, "match_columns", fake_match)

    result = tools_p2._check_join_union(tmp_path, "wide.parquet", "wide.parquet")

    assert shapes == [((300, 25), (300, 25))]
    assert "'wide.parquet' 300 rows x 25 columns" in result
    assert "was sampled" not in result


def test_check_join_union_says_when_it_measured_a_sample_of_a_big_table(monkeypatch, tmp_path):
    monkeypatch.setattr(tools_p2, "PAIR_SAMPLE_ROWS", 100)
    big = pd.DataFrame({"id": range(1000), "v": [f"x{i}" for i in range(1000)]})
    big.to_parquet(tmp_path / "big.parquet", row_group_size=50)
    big.to_parquet(tmp_path / "big2.parquet", row_group_size=50)

    result = tools_p2._check_join_union(tmp_path, "big.parquet", "big2.parquet")

    assert "'big.parquet' was sampled (" in result and " of 1,000 rows)" in result
    assert "lower bounds" in result and "upper bound" in result


def test_check_join_union_output_stays_within_the_tool_budget(tmp_path):
    cols = {f"column_{i}": [f"value {i}-{j}" for j in range(40)] for i in range(40)}
    pd.DataFrame(cols).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame(cols).to_parquet(tmp_path / "right.parquet")

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert len(result) <= tools_p2.MAX_TOOL_OUTPUT_CHARS


def test_check_join_union_survives_a_column_of_nested_values(tmp_path):
    """One list-valued column must not cost the agent the whole report."""
    pd.DataFrame({"k": [[1], [2], [3]], "id": ["a", "b", "c"]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"k": [[1], [2], [3]], "id": ["a", "b", "c"]}).to_parquet(tmp_path / "right.parquet")

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert not result.startswith("Error")
    assert "id (left.parquet) = id (right.parquet)" in result
    assert "UNION: 'left.parquet' unions with 'right.parquet'" in result


def test_check_join_union_reports_an_unexpected_measurement_failure_as_a_tool_error(monkeypatch, tmp_path):
    pd.DataFrame({"k": [1, 2, 3]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"k": [1, 2, 3]}).to_parquet(tmp_path / "right.parquet")

    def boom(*_args, **_kwargs):
        raise RuntimeError("measurement exploded")

    monkeypatch.setattr(tools_p2, "find_join_keys", boom)

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert result.startswith("Error checking join/union between 'left.parquet'")
    assert "measurement exploded" in result


def test_check_join_union_reports_unreadable_tables(tmp_path):
    result = tools_p2._check_join_union(tmp_path, "missing.parquet", "other.parquet")

    assert result.startswith("Error checking join/union between 'missing.parquet'")


def test_check_join_union_worst_case_report_fits_the_budget_with_real_length_file_names(tmp_path):
    """The caps (3 join keys, 8 mapped pairs, 4 warnings) bound the report: even with ~80-character
    UK file names, three many-to-many keys with notes, four type-clash warnings and placeholder
    columns it stays within the tool budget and still reaches every section."""
    left = "05af53dc-891a-4598-8d8b-7f9c70fa6df3___9591a8f3-e83f-4fc9-8c0c-1b2e92a5f5bc.parquet"
    right = "ee3cd9c1-ae38-4b2f-81e6-2332c4799f3f___dfce03d1-98fc-4f5c-b355-08c2fdddeee2.parquet"
    rows = 60
    a, b = {}, {}
    for i in range(3):                     # categorical columns: many-to-many candidates with notes
        a[f"Category {i}"] = [f"c{j % 4}" for j in range(rows)]
        b[f"Category {i}"] = [f"c{j % 4}" for j in range(rows)]
    for i in range(4):                     # text-vs-number clashes: warnings
        a[f"Reference {i}"] = [f"DWP-{j:03d}" for j in range(rows)]
        b[f"Reference {i}"] = list(range(900, 900 + rows))
    for i in range(6):                     # enough mapped columns to hit the pair cap
        a[f"Extra column number {i}"] = [f"e{j}" for j in range(rows)]
        b[f"Extra column number {i}"] = [f"f{j}" for j in range(rows)]
    for i in range(3):                     # placeholder headers
        a[f"_duplicated_{i}"] = [1] * rows
        b[f"_duplicated_{i}"] = [2] * rows
    pd.DataFrame(a).to_parquet(tmp_path / left)
    pd.DataFrame(b).to_parquet(tmp_path / right)

    result = tools_p2._check_join_union(tmp_path, left, right)

    assert len(result) <= tools_p2.MAX_TOOL_OUTPUT_CHARS and not result.endswith("...")
    assert "JOIN (many-to-many only)" in result and "UNION: '" in result
    assert result.count("warning:") == 4 and "... and " in result and "placeholder header" in result
