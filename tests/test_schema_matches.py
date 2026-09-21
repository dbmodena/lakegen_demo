import pandas as pd

from lakegen.agent_tools import schema_matching, tools_p2


def _evidence(matches):
    scores = [score for _, _, score in matches]
    return {
        "matches": matches,
        "sm_macro_avg": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "sm_micro_avg": max(scores, default=0.0),
        "sm_n_matches": len(matches),
        "sm_time": 0.0,
    }


def test_schema_gate_passes_on_average_or_best_score():
    assert schema_matching.passes_schema_gate({"sm_macro_avg": 0.5, "sm_micro_avg": 0.1})
    assert schema_matching.passes_schema_gate({"sm_macro_avg": 0.2, "sm_micro_avg": 0.5})
    assert not schema_matching.passes_schema_gate({"sm_macro_avg": 0.49, "sm_micro_avg": 0.49})


def test_union_evidence_aligns_pairs_at_or_above_threshold():
    evidence = _evidence([("a", "x", 0.9), ("b", "y", 0.5), ("c", "z", 0.2)])

    union = schema_matching.union_evidence(evidence, ["a", "b", "c", "d"])

    assert union["supported"]
    assert union["q_columns"] == ["a", "b"]
    assert union["r_columns"] == ["x", "y"]
    assert union["column_scores"] == [0.9, 0.5]
    assert union["union_column_ratio"] == 0.5


def test_low_average_blocks_union_but_best_pair_supports_join():
    evidence = _evidence([("id", "key", 0.9), ("b", "y", 0.1), ("c", "z", 0.1)])

    union = schema_matching.union_evidence(evidence, ["id", "b", "c"])
    join = schema_matching.join_evidence(evidence)

    assert not union["supported"]
    assert join["supported"]
    assert join["key"] == ("id", "key")
    assert join["alternatives"] == []


def test_join_evidence_without_matches_is_unsupported():
    join = schema_matching.join_evidence(_evidence([]))

    assert join == {"key": None, "score": 0.0, "supported": False, "alternatives": []}


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
    assert "4 of 4 columns of 'left.parquet' aligned" in result
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

    assert "JOIN: 'schools.parquet' joins 'attendance.parquet' on " in result
    assert "school_id (schools.parquet) = School ID (attendance.parquet)" in result


def test_check_join_union_states_join_without_union(monkeypatch, tmp_path):
    pd.DataFrame({"id": [1], "b": [1], "c": [1]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"key": [1], "y": [1], "z": [1]}).to_parquet(tmp_path / "right.parquet")
    monkeypatch.setattr(
        schema_matching,
        "match_columns",
        lambda *_args: {("id", "key"): 0.9, ("b", "y"): 0.1, ("c", "z"): 0.1},
    )

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert (
        "JOIN: 'left.parquet' joins 'right.parquet' on "
        "id (left.parquet) = key (right.parquet) (match score 0.900 >= 0.5)."
    ) in result
    assert "UNION: 'left.parquet' does not union with 'right.parquet'" in result


def test_check_join_union_states_no_relationship(monkeypatch, tmp_path):
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(tmp_path / "right.parquet")
    monkeypatch.setattr(
        schema_matching, "match_columns", lambda *_args: {("a", "x"): 0.2}
    )

    result = tools_p2._check_join_union(tmp_path, "left.parquet", "right.parquet")

    assert (
        "NO RELATIONSHIP: 'left.parquet' neither joins nor unions with 'right.parquet'"
    ) in result
    assert "JOIN:" not in result
    assert "UNION:" not in result


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

    def fake_valentine_match(q, r, matcher):
        calls.append((q, r, matcher))
        return {}

    monkeypatch.setattr(schema_matching, "valentine_match", fake_valentine_match)

    schema_matching.match_columns(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))

    assert len(calls) == 1
    assert list(calls[0][0].columns) == ["a"]
    assert list(calls[0][1].columns) == ["b"]
    assert calls[0][2].__class__.__name__ == "Coma"


def test_check_join_union_reports_unreadable_tables(tmp_path):
    result = tools_p2._check_join_union(tmp_path, "missing.parquet", "other.parquet")

    assert result.startswith("Error checking join/union between 'missing.parquet'")
