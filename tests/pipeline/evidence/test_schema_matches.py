import itertools

import pandas as pd

from lakegen.pipeline.evidence import table_inspection as tools_p2
from lakegen.pipeline.evidence.join_union.matching import schema_matching


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
    assert "UNION: 'left.parquet' unions with 'right.parquet'" in result
    assert "borough -> borough" in result
    assert "4 columns map 1-1 (100% of the left table's columns" in result
    assert "schema" not in result.casefold()


def test_check_join_union_states_the_join_key(tmp_path):
    rows = range(50)
    pd.DataFrame({"school_id": [f"S{i}" for i in rows], "enrollment": list(rows)}).to_parquet(
        tmp_path / "schools.parquet"
    )
    pd.DataFrame({"School ID": [f"S{i}" for i in rows], "attendance_rate": [0.9] * 50}).to_parquet(
        tmp_path / "attendance.parquet"
    )

    result = tools_p2._check_join_union(tmp_path, "schools.parquet", "attendance.parquet")

    assert "JOIN: 'schools.parquet' joins 'attendance.parquet'; candidate keys" in result
    assert "school_id (schools.parquet) = School ID (attendance.parquet)" in result


def test_check_join_union_states_join_without_union(monkeypatch, tmp_path):
    pd.DataFrame({"id": [1, 2], "b": [1, 1], "c": [1, 1]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"key": [1, 2], "y": [1, 1], "z": [1, 1]}).to_parquet(tmp_path / "right.parquet")
    monkeypatch.setattr(
        schema_matching,
        "match_columns",
        lambda *_args: {("id", "key"): 0.9, ("b", "y"): 0.1, ("c", "z"): 0.1},
    )

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert "JOIN: 'left.parquet' joins 'right.parquet'; candidate keys" in result
    assert "id (left.parquet) = key (right.parquet) -- 1:1, 2 shared key values" in result
    assert "NO UNION:" in result


def test_check_join_union_states_no_relationship(monkeypatch, tmp_path):
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "right.parquet")
    monkeypatch.setattr(schema_matching, "match_columns", lambda *_args: {("a", "x"): 0.2})

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert (
        "NO JOIN: no column pair between 'left.parquet' and 'right.parquet' has similar names"
        in result
    )
    assert "NO UNION:" in result


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


def test_check_join_union_compares_all_rows_and_columns(monkeypatch, tmp_path):
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


def test_match_columns_scores_names_with_valentine_coma_first(monkeypatch):
    calls = []

    def fake_valentine_match(frames, matcher, **kwargs):
        calls.append((frames, matcher, kwargs))
        return {}

    monkeypatch.setattr(schema_matching, "valentine_match", fake_valentine_match)

    schema_matching.match_columns(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))

    assert len(calls) == 1  # nothing is left to score on values
    assert list(calls[0][0][0].columns) == ["a"]
    assert list(calls[0][0][1].columns) == ["b"]
    assert calls[0][1].__class__.__name__ == "Coma"
    assert calls[0][2]["instance_sample_size"] == schema_matching.COMA_SAMPLE_ROWS


def _codes(n=60, offset=0, name="zona"):
    return pd.DataFrame({name: [f"Z{i + offset:03d}" for i in range(n)], "n": range(n)})


def test_a_renamed_column_is_matched_on_its_values():
    """`zona` / `zona_2001`, `postcode` / `pcd`, a header lost to `col_0`: the names say nothing, the
    values do."""
    matches = {tuple(m[:2]): m for m in schema_matching.match_pair(_codes(), _codes(name="col_0"))}

    match = matches[("zona", "col_0")]
    assert match.by_values and match[2] >= schema_matching.SCHEMA_MATCH_THRESHOLD
    assert match.name_score < schema_matching.SCHEMA_MATCH_THRESHOLD <= match.instance_score


def test_a_column_with_a_name_partner_is_not_scored_on_its_values():
    matches = schema_matching.match_columns(_codes(), _codes())

    assert matches[("zona", "zona")].instance_score is None
    assert not matches[("zona", "zona")].by_values


def test_amounts_and_row_numbers_are_not_matched_on_values():
    """Two columns of small whole numbers or of fractions resemble each other by chance."""
    q = pd.DataFrame({"amount": [i / 7 for i in range(80)], "OBJECTID": range(1, 81)})
    r = pd.DataFrame({"cost": [i / 7 for i in range(80)], "ref": range(1, 81)})

    assert schema_matching.match_pair(q, r) == []


def test_flags_with_two_values_are_not_matched_on_values():
    q = pd.DataFrame({"flag": ["y", "n"] * 30})
    r = pd.DataFrame({"x": ["y", "n"] * 30})

    assert schema_matching.match_pair(q, r) == []


def test_only_the_most_varied_columns_without_a_name_partner_are_scored_on_values(monkeypatch):
    width = schema_matching.MAX_VALUE_COLUMNS + 20
    # letters the two tables do not share, so no pair of names is similar
    q_names = ["".join(p) for p in itertools.islice(itertools.product("bcdfghjk", repeat=3), width)]
    r_names = ["".join(p) for p in itertools.islice(itertools.product("lmnpqrst", repeat=3), width)]
    q = pd.DataFrame({name: [f"v{j}-{i}" for j in range(30)] for i, name in enumerate(q_names)})
    r = pd.DataFrame({name: [f"v{j}-{i}" for j in range(30)] for i, name in enumerate(r_names)})
    scored_on_values = []
    real = schema_matching._run_coma

    def spy(a, b, *, use_instances):
        if use_instances:
            scored_on_values.append((a.shape[1], b.shape[1]))
        return real(a, b, use_instances=use_instances)

    monkeypatch.setattr(schema_matching, "_run_coma", spy)

    schema_matching.match_columns(q, r)

    assert scored_on_values == [
        (schema_matching.MAX_VALUE_COLUMNS, schema_matching.MAX_VALUE_COLUMNS)
    ]


def test_a_failing_value_pass_leaves_the_name_matches(monkeypatch):
    real = schema_matching._run_coma

    def broken(a, b, *, use_instances):
        if use_instances:
            raise RuntimeError("tf-idf failed")
        return real(a, b, use_instances=False)

    monkeypatch.setattr(schema_matching, "_run_coma", broken)

    matches = schema_matching.match_pair(_codes(), _codes().rename(columns={"zona": "col_0"}))

    assert [tuple(m[:2]) for m in matches] == [("n", "n")]


def test_long_text_cells_are_cut_before_valentine_reads_them():
    frame = pd.DataFrame({"geom": ["POLYGON(" + "1 1," * 5000 + ")"] * 5})

    sample = schema_matching._bounded_sample(frame)

    assert sample["geom"].str.len().max() == schema_matching.MAX_CELL_CHARS
    assert frame["geom"].str.len().min() > 1000  # the caller's frame is untouched


def test_a_column_match_unpacks_and_compares_as_a_plain_triple():
    match = schema_matching.ColumnMatch("a", "b", 0.9, 0.0, 1.0)

    left, right, score = match
    assert (left, right, score) == ("a", "b", 0.9) and match == ("a", "b", 0.9)
    assert match.by_values


def test_check_join_union_reports_unreadable_tables(tmp_path):
    result = tools_p2._check_join_union(tmp_path, "missing.parquet", "other.parquet")

    assert result.startswith("Error checking join/union between 'missing.parquet'")


# ---- the report over the whole tables ----------------------------------------------------------------


def _pair_files(tmp_path, n=6_000, row_group_size=1_000):
    frame = pd.DataFrame(
        {
            "id": [f"i{i}" for i in range(n)],
            "district": [f"d{i % 3}" for i in range(n)],
            "amount": [i % 97 for i in range(n)],
        }
    )
    frame.to_parquet(tmp_path / "left.parquet", row_group_size=row_group_size)
    frame.to_parquet(tmp_path / "right.parquet", row_group_size=row_group_size)


def test_the_join_sizes_are_counted_over_every_row_when_the_tables_are_sampled(
    monkeypatch, tmp_path
):
    from lakegen.pipeline.evidence.join_union import pair_check

    monkeypatch.setattr(pair_check, "PAIR_SAMPLE_ROWS", 1_000)
    _pair_files(tmp_path)

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert "was sampled (1,000 of 6,000 rows)" in result
    assert "the join sizes are counted over every row of the key columns" in result
    assert "id (left.parquet) = id (right.parquet) -- 1:1, 6,000 shared key values" in result
    assert "inner join = 6,000 rows" in result
    assert "~" not in result


def test_the_report_names_the_district_key_the_sandbox_would_stop(monkeypatch, tmp_path):
    from lakegen.pipeline.evidence.join_union import pair_check

    monkeypatch.setattr(pair_check, "PAIR_SAMPLE_ROWS", 1_000)
    n = 30_000
    frame = pd.DataFrame({"id": range(n), "district": [f"d{i % 3}" for i in range(n)]})
    frame.to_parquet(tmp_path / "left.parquet", row_group_size=1_000)
    frame.to_parquet(tmp_path / "right.parquet", row_group_size=1_000)

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert "inner join = 300,000,000 rows" in result  # 3 districts x 10,000 x 10,000
    assert "EXPLODES: 10,000x the larger table" in result
    assert "that join has 3 rows" in result


def test_the_union_verdict_survives_very_long_names(tmp_path):
    name = "Number of residents who received financial counseling or banking services "
    columns = {name * 2 + str(i): [f"v{j}-{i}" for j in range(40)] for i in range(12)}
    pd.DataFrame(columns).to_parquet(tmp_path / ("a" * 80 + ".parquet"))
    pd.DataFrame(columns).to_parquet(tmp_path / ("b" * 80 + ".parquet"))

    result = tools_p2._check_join_union(tmp_path, "a" * 80 + ".parquet", "b" * 80 + ".parquet")

    assert len(result) <= tools_p2.MAX_TOOL_OUTPUT_CHARS
    assert "UNION: " in result
    assert result.index("JOIN") < result.index("UNION: ")


def test_the_report_of_a_pair_is_kept_until_one_of_its_files_changes(monkeypatch, tmp_path):
    from lakegen.pipeline.evidence.join_union import pair_check

    pair_check._REPORTS.clear()
    _pair_files(tmp_path, n=100)
    analysed = []
    real = pair_check.analyse_pair
    monkeypatch.setattr(
        pair_check, "analyse_pair", lambda *a, **k: analysed.append(1) or real(*a, **k)
    )

    first = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")
    again = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")
    assert first == again and len(analysed) == 1

    pd.DataFrame({"id": ["x", "y"], "district": ["a", "b"], "amount": [1, 2]}).to_parquet(
        tmp_path / "right.parquet"
    )
    changed = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")
    assert changed != first and len(analysed) == 2


def test_an_error_is_not_cached(tmp_path):
    from lakegen.pipeline.evidence.join_union import pair_check

    pair_check._REPORTS.clear()
    assert tools_p2._check_join_union(tmp_path, "missing.parquet", "other.parquet").startswith(
        "Error"
    )
    assert not pair_check._REPORTS
