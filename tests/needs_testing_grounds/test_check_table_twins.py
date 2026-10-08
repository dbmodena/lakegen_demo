import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_table_twins", Path(__file__).resolve().parents[1] / "analysis" / "check_table_twins.py"
)
twins = importlib.util.module_from_spec(_SPEC)
sys.modules["check_table_twins"] = twins
_SPEC.loader.exec_module(twins)


_FIND = importlib.util.spec_from_file_location(
    "find_table_twins", Path(__file__).resolve().parents[1] / "analysis" / "find_table_twins.py"
)
find = importlib.util.module_from_spec(_FIND)
sys.modules["find_table_twins"] = find
_FIND.loader.exec_module(find)


@pytest.fixture
def base() -> pd.DataFrame:
    return pd.DataFrame({
        "Name": ["a", "b", "c", "d"],
        "Area": [1.5, 2.5, np.nan, 4.5],
        "Count": [1, 2, 3, 4],
        "FID": [0, 1, 2, 3],
    })


def _compare(tmp_path, left: pd.DataFrame, right: pd.DataFrame, **write_right) -> dict:
    left.to_parquet(tmp_path / "a.parquet")
    right.to_parquet(tmp_path / "b.parquet", **write_right)
    return twins.compare_content(tmp_path / "a.parquet", tmp_path / "b.parquet")


def test_same_file_is_identical_file(tmp_path, base):
    assert _compare(tmp_path, base, base)["verdict"] == "IDENTICAL_FILE"


def test_same_data_with_other_compression_is_identical_content(tmp_path, base):
    result = _compare(tmp_path, base, base, compression="gzip")
    assert not result["file_sha256_equal"]
    assert result["verdict"] == "IDENTICAL_CONTENT"


def test_int_and_float_with_equal_values_are_identical_content(tmp_path, base):
    other = base.assign(Count=base["Count"].astype(float))
    assert _compare(tmp_path, base, other)["verdict"] == "IDENTICAL_CONTENT"


def test_shuffled_rows_are_reordered(tmp_path, base):
    result = _compare(tmp_path, base, base.iloc[[3, 1, 0, 2]].reset_index(drop=True))
    assert result["verdict"] == "SAME_CONTENT_REORDERED"
    assert not result["row_order_equal"] and result["row_multiset_equal"]


def test_permuted_columns_are_reordered(tmp_path, base):
    result = _compare(tmp_path, base, base[["FID", "Name", "Count", "Area"]])
    assert result["verdict"] == "SAME_CONTENT_REORDERED"
    assert result["column_relation"] == "reordered"


def test_renamed_columns_are_found_by_content(tmp_path, base):
    result = _compare(tmp_path, base, base.rename(columns={"Name": "name", "Area": "Surface"}))
    assert result["verdict"] == "SAME_CONTENT_RENAMED_COLUMNS"
    renames = {(r["a"], r["b"]): r["cosmetic"] for r in result["renamed_columns"]}
    assert renames == {("Name", "name"): True, ("Area", "Surface"): False}


def test_one_changed_value_is_same_shape_different_values(tmp_path, base):
    other = base.copy()
    other.loc[1, "Count"] = 20
    result = _compare(tmp_path, base, other)
    assert result["verdict"] == "SAME_SHAPE_DIFFERENT_VALUES"
    assert result["differing_cells"] == 1 and result["differing_columns"] == {"Count": 1}


def test_null_in_another_position_counts_as_a_difference(tmp_path, base):
    other = base.copy()
    other.loc[[1, 2], "Area"] = [np.nan, 3.5]
    result = _compare(tmp_path, base, other)
    assert result["verdict"] == "SAME_SHAPE_DIFFERENT_VALUES"
    assert result["differing_columns"] == {"Area": 2}


def test_string_null_versus_value_is_counted(tmp_path, base):
    other = base.copy()
    other.loc[0, "Name"] = None
    result = _compare(tmp_path, base, other)
    assert result["differing_columns"] == {"Name": 1}


def test_replaced_column_is_different_even_with_equal_numeric_sums(tmp_path, base):
    # Same shape and same only-numeric-sum, but another geography column.
    left = base[["Name", "FID"]].assign(Region=["x", "y", "z", "w"])
    right = base[["Name", "FID"]].assign(Alliance=["p", "q", "r", "s"])
    result = _compare(tmp_path, left, right)
    assert result["verdict"] == "DIFFERENT"
    assert result["unmatched_a"] == ["Region"] and result["unmatched_b"] == ["Alliance"]


def test_different_row_counts_are_different(tmp_path, base):
    assert _compare(tmp_path, base, base.iloc[:3])["verdict"] == "DIFFERENT"


def test_missing_file_is_unavailable(tmp_path, base):
    base.to_parquet(tmp_path / "a.parquet")
    result = twins.compare_content(tmp_path / "a.parquet", tmp_path / "nope.parquet")
    assert result["verdict"] == "UNAVAILABLE"


def test_source_key_treats_both_arcgis_url_forms_as_one_item():
    item = "197d33a98b9147459852ce3a42ce05bf"
    api = twins.source_key(f"https://h.example/api/download/v1/items/{item}/csv?layers=18")
    dataset = twins.source_key(f"https://h.example/datasets/{item}_18.csv?outSR=%7B%7D")
    assert api == dataset
    assert twins.source_key(f"https://h.example/datasets/{item}_19.csv") != api
    assert twins.source_key("https://h.example/datasets/ORG::some-slug.csv?outSR=1")[1].startswith("slug:")


def test_split_table_id_accepts_file_names_and_rejects_bare_ids():
    assert twins.split_table_id("d1___r1.parquet") == ("d1", "r1")
    with pytest.raises(ValueError):
        twins.split_table_id("r1")


@pytest.mark.parametrize("content, metadata, expected", [
    ({"verdict": "IDENTICAL_FILE"},
     {"index_descriptive_equal": True, "index_differences": [], "ckan_differences": [], "bookkeeping_differences": ["tags"]},
     "SAME_TABLE"),
    ({"verdict": "IDENTICAL_FILE"},
     {"index_descriptive_equal": True, "index_differences": [], "ckan_differences": ["ckan_package.title"]},
     "SAME_DATA_CKAN_ONLY_DIFFERENCES"),
    ({"verdict": "SAME_CONTENT_REORDERED"},
     {"index_descriptive_equal": False, "index_differences": ["solr.description"], "ckan_differences": []},
     "SAME_DATA_DIFFERENT_DESCRIPTION"),
    ({"verdict": "IDENTICAL_FILE"},
     {"index_descriptive_equal": False, "index_differences": [], "ckan_differences": []},
     "SAME_DATA_METADATA_UNVERIFIED"),
    ({"verdict": "SAME_SHAPE_DIFFERENT_VALUES", "differing_cells_pct": 30.0, "max_relative_numeric_deviation": 0.5},
     {"index_descriptive_equal": True}, "VARIANT_SAME_SHAPE"),
    ({"verdict": "DIFFERENT"}, {"index_descriptive_equal": True}, "DIFFERENT_TABLES"),
])
def test_conclusion_ladder(content, metadata, expected):
    assert twins.conclude(content, metadata)[0] == expected


def test_source_key_outside_arcgis_uses_the_full_url():
    same = twins.source_key("https://h.example/data/a.csv?x=1")
    assert twins.source_key("https://h.example/data/a.csv?x=1") == same
    assert twins.source_key("https://h.example/data/a.csv?x=2") != same


def test_one_published_file_with_different_labels_is_not_a_reupload():
    metadata = {"source": {"relation": "same_source"}, "index_descriptive_equal": False,
                "index_differences": ["solr.title"], "ckan_differences": ["ckan_package.title"]}
    verdict, reason = twins.conclude({"verdict": "IDENTICAL_FILE"}, metadata)
    assert verdict == "SAME_FILE_DIFFERENT_LABELS" and "solr.title" in reason


def test_identical_description_and_same_file_is_a_reupload():
    metadata = {"source": {"relation": "same_file_name"}, "index_descriptive_equal": True,
                "index_differences": [], "ckan_differences": []}
    assert twins.conclude({"verdict": "IDENTICAL_FILE"}, metadata)[0] == "SAME_TABLE"


def test_identical_description_but_differently_named_files_needs_review():
    metadata = {"source": {"relation": "different_file_name"}, "index_descriptive_equal": True,
                "index_differences": [], "ckan_differences": []}
    assert twins.conclude({"verdict": "IDENTICAL_FILE"}, metadata)[0] == "SAME_DATA_DIFFERENT_FILE_NAMES"


@pytest.mark.parametrize("a, b, expected", [
    ("https://h/x/a.csv", "https://h/x/a.csv", "same_source"),
    ("https://h/up/2022-10-27T16-21-32Z-organogram.csv", "https://h/up/2022-10-27T16-23-15Z-organogram.csv", "same_file_name"),
    ("https://h/spend-2018-08.csv", "https://h/spend-2018-09.csv", "different_file_name"),
    ("https://h/api/download/v1/items/" + "a" * 32 + "/csv?layers=0", "https://h/datasets/ORG::some-slug.csv?x=1", "unresolved_slug"),
    (None, "https://h/a.csv", "unknown"),
])
def test_file_relation(a, b, expected):
    assert twins.file_relation(a, b) == expected


def test_same_content_but_different_files_and_descriptions_is_not_a_twin():
    metadata = {"source": {"relation": "different_source"}, "index_descriptive_equal": False,
                "index_differences": ["solr.title"], "ckan_differences": []}
    assert twins.conclude({"verdict": "IDENTICAL_FILE"}, metadata)[0] == "SAME_DATA_DIFFERENT_DESCRIPTION"


# --- whole-lake grouping -----------------------------------------------------------------

def _record(resource_id, name, url, period="", package="Pkg"):
    package_record = {"id": "p-" + resource_id, "title": package, "notes": "n", "organization": {"title": "Org"},
                      "license_id": "ogl", "extras": []}
    resource = {"id": resource_id, "name": name, "description": None, "format": "CSV",
                "datafile-date": period, "url": url}
    return package_record, resource


def _refine(monkeypatch, members, records, docs=None):
    monkeypatch.setattr(find, "fetch_solr_docs", lambda *_a, **_k: docs or {})
    cluster = {"members": [f"d___{m}.parquet" for m in members]}
    find.refine_by_metadata([cluster], solr=("http://solr", "uk"), ckan=records)
    return cluster


def test_two_weekly_snapshots_with_equal_numbers_are_not_twins(monkeypatch):
    records = {
        "r1": _record("r1", "as at 28-12-2020", "https://h/officials-28-12-2020.csv"),
        "r2": _record("r2", "as at 21-12-2020", "https://h/officials-21-12-2020.csv"),
    }
    cluster = _refine(monkeypatch, ["r1", "r2"], records)
    assert cluster["metadata_split"] and len(cluster["twin_groups"]) == 2
    assert "resource.name" in cluster["split_on"]


def test_one_file_catalogued_under_two_names_is_not_a_twin_but_is_reviewed(monkeypatch):
    records = {
        "r1": _record("r1", "Brownfield Register", "https://h/register.csv", package="A"),
        "r2": _record("r2", "Brownfield land register 2018", "https://h/register.csv", package="B"),
    }
    cluster = _refine(monkeypatch, ["r1", "r2"], records)
    assert cluster["metadata_split"] and len(cluster["twin_groups"]) == 2
    assert len(cluster["review"]["same_file_different_labels"]) == 1


def test_reupload_with_identical_description_is_a_twin(monkeypatch):
    records = {
        "r1": _record("r1", "2022-09-30 Organogram", "https://h/up/2022-10-27T16-21-32Z-organogram.csv", period="2022-09-30"),
        "r2": _record("r2", "2022-09-30 Organogram", "https://h/up/2022-10-27T16-23-15Z-organogram.csv", period="2022-09-30"),
    }
    cluster = _refine(monkeypatch, ["r1", "r2"], records)
    assert not cluster["metadata_split"] and cluster["twin_evidence"] == [["same_file_name"]]


def test_identical_description_with_differently_named_files_is_reviewed_not_merged(monkeypatch):
    records = {
        "r1": _record("r1", "Spend return", "https://h/spend-2018-08.csv"),
        "r2": _record("r2", "Spend return", "https://h/spend-2018-09.csv"),
    }
    cluster = _refine(monkeypatch, ["r1", "r2"], records)
    assert cluster["metadata_split"]
    assert len(cluster["review"]["same_description_different_file_names"]) == 1


def test_period_alone_keeps_same_named_tables_apart(monkeypatch):
    records = {
        "r1": _record("r1", "Spend return", "https://h/a.csv", period="2018-08-31"),
        "r2": _record("r2", "Spend return", "https://h/b.csv", period="2018-09-30"),
    }
    cluster = _refine(monkeypatch, ["r1", "r2"], records)
    assert cluster["metadata_split"] and cluster["split_on"] == ["resource.datafile-date"]


def test_member_without_metadata_never_merges(monkeypatch):
    records = {"r1": _record("r1", "x", "https://h/a.csv")}
    cluster = _refine(monkeypatch, ["r1", "r2"], records)
    assert cluster["metadata_split"] and not cluster["metadata_known"]
