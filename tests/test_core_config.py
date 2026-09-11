from pathlib import Path

import pandas as pd

from lakegen.core import config


def test_uk_resolvers_prefer_cleaned_assets(monkeypatch, tmp_path):
    uk = tmp_path / "data" / "uk"
    old_tables = uk / "datasets" / "parquet"
    clean_tables = uk / "clean_datasets" / "parquet"
    metadata = uk / "metadata"
    old_tables.mkdir(parents=True)
    clean_tables.mkdir(parents=True)
    metadata.mkdir()
    pd.DataFrame({"old": [1]}).to_parquet(old_tables / "old.parquet")
    pd.DataFrame({"clean": [1]}).to_parquet(clean_tables / "clean.parquet")
    (metadata / "metadata_retrieved_only.json").write_text("[]")
    (metadata / "metadata_retrieved_cleaned.json").write_text("[]")
    (metadata / "datasets_metadata.csv").write_text("table,rows\nclean,1\n")
    monkeypatch.setattr(config, "BASE_DIR", Path(tmp_path))

    assert config.resolve_portal_tables_dir("uk") == clean_tables
    assert config.resolve_portal_metadata_path("uk") == (
        metadata / "metadata_retrieved_cleaned.json"
    )
    assert config.resolve_portal_dataset_statistics_path("uk") == (
        metadata / "datasets_metadata.csv"
    )


def test_nyc_resolver_keeps_existing_dataset_layout(monkeypatch, tmp_path):
    tables = tmp_path / "data" / "nyc" / "datasets" / "parquet"
    tables.mkdir(parents=True)
    pd.DataFrame({"value": [1]}).to_parquet(tables / "table.parquet")
    monkeypatch.setattr(config, "BASE_DIR", Path(tmp_path))

    assert config.resolve_portal_tables_dir("nyc") == tables
