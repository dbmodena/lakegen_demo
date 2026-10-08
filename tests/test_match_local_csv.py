from lakegen.discovery.candidates import match_local_csv


def test_match_local_csv_picks_the_resource_not_its_datasets_first_file():
    # UK files are named <dataset_id>___<resource_id>; every one of a dataset's
    # resources contains the dataset id.
    files = ["ds-1___res-a.parquet", "ds-1___res-b.parquet", "ds-2___res-c.parquet"]

    assert match_local_csv({"dataset_id": "ds-1", "resource_id": "res-b"}, files) == (
        "ds-1___res-b.parquet"
    )


def test_match_local_csv_matches_files_named_by_resource_id():
    files = ["22gm-5ceg.parquet", "abcd-1234.parquet"]
    doc = {"dataset_id": "abcd-1234", "resource_id": "abcd-1234"}

    assert match_local_csv(doc, files) == "abcd-1234.parquet"


def test_match_local_csv_falls_back_to_resource_then_dataset_substrings():
    files = ["Parking__ds-9.csv", "prefix_res-z_suffix.csv"]

    assert match_local_csv({"dataset_id": "ds-9", "resource_id": "res-z"}, files) == (
        "prefix_res-z_suffix.csv"
    )
    assert match_local_csv({"dataset_id": "ds-9", "resource_id": "res-y"}, files) == (
        "Parking__ds-9.csv"
    )
    assert match_local_csv({"dataset_id": "ds-8", "resource_id": "res-y"}, files) is None
    assert match_local_csv({}, files) is None
