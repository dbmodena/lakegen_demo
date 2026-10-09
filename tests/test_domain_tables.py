"""Table ids reduce to the resource id the metrics compare."""

from __future__ import annotations

import pytest

from lakegen.experiments.analysis.results import final_ranking, selected_tables, solved, tokens
from lakegen.domain.tables import resource_id, table_id


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("parks.parquet", "parks"),
        ("parks.CSV", "parks"),
        ("parks.pq", "parks"),
        ("/data/lake/tables/parks.parquet", "parks"),
        ("parks.json", "parks.json"),
        ("parks", "parks"),
        ("a.b.parquet", "a.b"),
    ],
)
def test_table_id_drops_a_table_suffix_and_any_directory(value, expected):
    assert table_id(value) == expected


def test_resource_id_is_the_part_after_the_last_dataset_separator():
    assert resource_id("dataset___resource.parquet") == "resource"
    assert resource_id("a___b___c") == "c"
    assert resource_id("plain.parquet") == "plain"
    assert resource_id(123) == "123"


def test_final_ranking_takes_the_last_attempt_in_rank_order_without_repeats():
    result = {
        "ranking": [
            {"attempt": 1, "rank": 1, "resource_id": "old"},
            {"attempt": 2, "rank": 2, "resource_id": "ds___b"},
            {"attempt": 2, "rank": 1, "resource_id": "a"},
            {"attempt": 2, "rank": 3, "resource_id": "a.parquet"},
            {"attempt": 2, "rank": 4, "resource_id": ""},
            "not a dict",
        ],
        "tables": ["ignored.parquet"],
    }
    assert final_ranking(result) == ["a", "b"]


def test_final_ranking_falls_back_to_the_selection():
    assert final_ranking({"tables": ["x___y.parquet", "y.csv", "z"]}) == ["y", "z"]
    assert final_ranking({}) == []
    assert selected_tables({"tables": ["d___t.parquet", "t"]}) == ["t"]


def test_outcome_and_tokens_read_the_stored_result():
    assert solved({"code_evaluation": {"supported_correct": True}})
    assert not solved({"code_evaluation": {}}) and not solved({})
    spent = tokens({"tokens": {"p1_p2": 10, "p3": "20", "p4": None}})
    assert (spent.p1_p2, spent.p3, spent.p4, spent.total) == (10, 20, 0, 30)
    assert tokens({}).total == 0
