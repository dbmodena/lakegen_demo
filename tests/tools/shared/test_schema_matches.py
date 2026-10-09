import pandas as pd

from lakegen.tools.shared import judge as tools_p2
from lakegen.tools.shared import schema_matching


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
    pd.DataFrame(
        {"school_id": [f"S{i}" for i in rows], "enrollment": list(rows)}
    ).to_parquet(tmp_path / "schools.parquet")
    pd.DataFrame(
        {"School ID": [f"S{i}" for i in rows], "attendance_rate": [0.9] * 50}
    ).to_parquet(tmp_path / "attendance.parquet")

    result = tools_p2._check_join_union(
        tmp_path, "schools.parquet", "attendance.parquet"
    )

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
    monkeypatch.setattr(
        schema_matching, "match_columns", lambda *_args: {("a", "x"): 0.2}
    )

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert "NO JOIN: no column pair between 'left.parquet' and 'right.parquet' has similar names" in result
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


def test_match_columns_uses_schema_only_valentine_api(monkeypatch):
    calls = []

    def fake_valentine_match(frames, matcher, **kwargs):
        calls.append((frames, matcher, kwargs))
        return {}

    monkeypatch.setattr(schema_matching, "valentine_match", fake_valentine_match)

    schema_matching.match_columns(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))

    assert len(calls) == 1
    assert list(calls[0][0][0].columns) == ["a"]
    assert list(calls[0][0][1].columns) == ["b"]
    assert calls[0][1].__class__.__name__ == "Coma"
    assert calls[0][2]["instance_sample_size"] == schema_matching.COMA_SAMPLE_ROWS


def test_check_join_union_reports_unreadable_tables(tmp_path):
    result = tools_p2._check_join_union(tmp_path, "missing.parquet", "other.parquet")

    assert result.startswith("Error checking join/union between 'missing.parquet'")
