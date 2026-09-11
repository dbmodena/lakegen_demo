import pandas as pd

from lakegen.agent_tools import tools_p2


def test_find_schema_matches_skips_valentine_for_the_same_file(monkeypatch, tmp_path):
    table = tmp_path / "table.parquet"
    pd.DataFrame(
        {
            "identifier": [1, 2, 3],
            "name": ["one", "two", "three"],
        }
    ).to_parquet(table)
    monkeypatch.setattr(
        tools_p2,
        "valentine_match",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Valentine should not run for identical schemas")
        ),
    )

    result = tools_p2._find_schema_matches(tmp_path, table.name, table.name)

    assert result.startswith("Exact-schema joinability analysis")
    assert "identifier <-> identifier" in result
    assert "name <-> name" in result
    assert "Schema similarity: 1.0000" in result


def test_find_schema_matches_skips_valentine_for_distinct_files_with_same_schema(
    monkeypatch, tmp_path
):
    columns = {"identifier": [1, 2], "value": [10, 20]}
    pd.DataFrame(columns).to_parquet(tmp_path / "left.parquet")
    pd.DataFrame(columns).to_parquet(tmp_path / "right.parquet")
    monkeypatch.setattr(
        tools_p2,
        "valentine_match",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Valentine should not run for identical schemas")
        ),
    )

    result = tools_p2._find_schema_matches(
        tmp_path, "left.parquet", "right.parquet"
    )

    assert result.startswith("Exact-schema joinability analysis")
    assert "identifier <-> identifier" in result
