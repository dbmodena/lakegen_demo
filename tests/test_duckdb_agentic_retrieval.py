import json

import pandas as pd
import pytest

from lakegen.core.catalogue import (
    catalogue_title,
    is_uninformative,
    strip_format_noise,
)
from lakegen.retrieval import (
    DuckDBAgenticRetriever,
    RetrievalConfig,
    RetrievalMode,
    TableRetrievalService,
)


def test_duckdb_agentic_retrieval_catalogs_counts_and_samples(tmp_path):
    pd.DataFrame({
        "school_name": ["Galileo", "Newton", "Curie"],
        "connection_type": ["fiber", "dsl", "fiber"],
        "mbps": [1000, 20, 500],
    }).to_parquet(tmp_path / "school_connectivity.parquet")
    pd.DataFrame({"street": ["Main"], "trees": [4]}).to_parquet(
        tmp_path / "urban_trees.parquet"
    )

    retriever = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic", duckdb_sample_rows=1), tmp_path
    )
    hits = retriever.retrieve(
        "Which schools have fiber connectivity?", ["fiber"], top_k=5
    )

    assert [hit.document["resource_id"] for hit in hits] == [
        "school_connectivity.parquet"
    ]
    evidence = hits[0].document["duckdb_evidence"]
    assert evidence["match_count"] == 2
    assert len(evidence["samples"]) == 1
    assert hits[0].rank == hits[0].lexical_rank == 1


def test_uniform_service_requires_directory_and_does_not_call_solr(tmp_path):
    config = RetrievalConfig(mode=RetrievalMode.DUCKDB_AGENTIC)
    with pytest.raises(ValueError, match="requires a local table_dir"):
        TableRetrievalService(object(), config)

    pd.DataFrame({"incident_kind": ["collision"]}).to_parquet(
        tmp_path / "road_events.parquet"
    )
    service = TableRetrievalService(object(), config, table_dir=str(tmp_path))
    hits = service.retrieve(
        question="road collision incidents", keywords=["collision"], top_k=1
    )
    assert hits[0].document["dataset_id"] == "road_events"


def test_ranking_prefers_distinct_term_coverage_over_raw_match_volume(tmp_path):
    pd.DataFrame({
        "kind": ["pedestrian plaza"],
        "partner": ["community organization"],
        "borough": ["Queens"],
    }).to_parquet(tmp_path / "relevant.parquet")
    pd.DataFrame({
        "organization": ["community organization"] * 10_000,
        "program": ["outreach"] * 10_000,
    }).to_parquet(tmp_path / "large_but_partial.parquet")

    hits = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic"), tmp_path
    ).retrieve(
        "pedestrian plazas partnered with community organizations",
        ["pedestrian", "plazas", "community", "organizations"],
        top_k=2,
    )

    assert hits[0].document["resource_id"] == "relevant.parquet"
    assert hits[0].document["duckdb_evidence"]["primary_coverage"] == 1.0
    assert hits[1].document["duckdb_evidence"]["primary_coverage"] == 0.5


def test_content_search_is_not_restricted_to_schema_matching_columns(tmp_path):
    pd.DataFrame({
        "feature_type": ["pedestrian plaza"],
        "operator": ["community partner"],
    }).to_parquet(tmp_path / "opaque-id.parquet")

    hit = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic"), tmp_path
    ).retrieve("Find pedestrian plazas", ["pedestrian", "plaza"], top_k=1)[0]

    evidence = hit.document["duckdb_evidence"]
    assert evidence["primary_coverage"] == 1.0
    assert evidence["term_counts"]["pedestrian"] == 1
    assert evidence["term_counts"]["plaza"] == 1


def test_descriptive_catalog_prefilters_opaque_parquet_ids(tmp_path):
    table_dir = tmp_path / "portal" / "datasets" / "parquet"
    metadata_dir = tmp_path / "portal" / "metadata"
    table_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    pd.DataFrame({"value": ["unrelated"]}).to_parquet(table_dir / "aaaa-aaaa.parquet")
    pd.DataFrame({"route": ["A1"]}).to_parquet(table_dir / "zzzz-zzzz.parquet")
    (metadata_dir / "metadata_retrieved_only.json").write_text(json.dumps([
        {
            "resource": {
                "id": "aaaa-aaaa",
                "name": "Building permits",
                "description": "Construction applications",
                "columns_name": ["value"],
                "columns_description": [],
            },
            "classification": {"domain_category": "Housing"},
        },
        {
            "resource": {
                "id": "zzzz-zzzz",
                "name": "Bicycle Routes",
                "description": "Street bicycle network and protected bike lanes",
                "columns_name": ["route"],
                "columns_description": ["Identifier for each bicycle route"],
            },
            "classification": {
                "domain_category": "Transportation",
                "domain_tags": ["bicycles"],
            },
        },
    ]), encoding="utf-8")

    hits = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic", duckdb_max_files=1), table_dir
    ).retrieve("List bicycle routes", ["bicycle", "route"], top_k=1)

    assert hits[0].document["resource_id"] == "zzzz-zzzz.parquet"
    assert hits[0].document["title"] == "Bicycle Routes"
    assert "protected bike lanes" in hits[0].document["description"]


def test_uk_ckan_catalog_maps_package_and_resource_ids(tmp_path):
    table_dir = tmp_path / "uk" / "datasets" / "parquet"
    metadata_dir = tmp_path / "uk" / "metadata"
    table_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    package_id = "11111111-1111-1111-1111-111111111111"
    resource_id = "22222222-2222-2222-2222-222222222222"
    filename = f"{package_id}___{resource_id}.parquet"
    pd.DataFrame({"quarter": ["Q1"], "paid": [95]}).to_parquet(
        table_dir / filename
    )
    (metadata_dir / "metadata_retrieved_only.json").write_text(json.dumps([
        {
            "id": package_id,
            "name": "prompt-payment-data",
            "title": "Government prompt payment data",
            "notes": "Percentage of supplier invoices paid within five days.",
            "theme-primary": "government",
            "organization": {"title": "HM Revenue and Customs"},
            "tags": [{"display_name": "supplier payments"}],
            "groups": [{"title": "Public spending"}],
            "resources": [{
                "id": resource_id,
                "name": "Prompt payments 2025 quarterly results",
                "description": "CSV results for the 2025 financial year",
            }],
        }
    ]), encoding="utf-8")

    hits = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic"), table_dir
    ).retrieve("supplier invoices paid promptly", ["supplier", "payment"], top_k=1)

    assert hits[0].document["resource_id"] == filename
    # Package title first, then the resource's own name (lakegen.core.catalogue).
    assert hits[0].document["title"] == (
        "Government prompt payment data Prompt payments 2025 quarterly results"
    )
    assert "Percentage of supplier invoices" in hits[0].document["description"]
    assert "HM Revenue and Customs" in hits[0].document["tags"]


def test_uk_cleaned_catalog_is_preferred_over_legacy_metadata(tmp_path):
    table_dir = tmp_path / "uk" / "clean_datasets" / "parquet"
    metadata_dir = tmp_path / "uk" / "metadata"
    table_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    package_id = "11111111-1111-1111-1111-111111111111"
    resource_id = "22222222-2222-2222-2222-222222222222"
    filename = f"{package_id}___{resource_id}.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(table_dir / filename)
    (metadata_dir / "metadata_retrieved_only.json").write_text(json.dumps([{
        "id": package_id, "title": "Legacy title",
        "resources": [{"id": resource_id, "name": "Legacy resource"}],
    }]))
    (metadata_dir / "metadata_retrieved_cleaned.json").write_text(json.dumps([{
        "id": package_id, "title": "Clean title",
        "resources": [{
            "id": resource_id, "name": "Clean retained resource",
            "description": "filtered retained table",
        }],
    }]))

    hit = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic"), table_dir
    ).retrieve("retained filtered table", ["retained"], top_k=1)[0]

    # Both halves come from the cleaned file, neither from the legacy one.
    assert hit.document["title"] == "Clean title Clean retained resource"
    assert "Legacy" not in hit.document["title"]


def test_global_footer_ranking_finds_schema_match_after_first_250(tmp_path):
    for index in range(250):
        pd.DataFrame({"opaque": ["unrelated"]}).to_parquet(
            tmp_path / f"a{index:03d}.parquet"
        )
    pd.DataFrame({"bicycle_route": ["R1"]}).to_parquet(
        tmp_path / "z_relevant.parquet"
    )

    retriever = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic", duckdb_max_files=250), tmp_path
    )
    hits = retriever.retrieve("Find bicycle routes", ["bicycle", "route"], top_k=1)

    assert hits[0].document["resource_id"] == "z_relevant.parquet"
    assert "bicycle_route" in hits[0].document["duckdb_evidence"]["matched_columns"]


def test_real_schema_ranks_files_without_descriptive_metadata(tmp_path):
    pd.DataFrame({"value": ["none"]}).to_parquet(tmp_path / "a_first.parquet")
    pd.DataFrame({"school_district": [12]}).to_parquet(tmp_path / "z_school.parquet")

    retriever = DuckDBAgenticRetriever(
        RetrievalConfig(
            mode="duckdb_agentic",
            duckdb_max_files=1,
            duckdb_probe_files=1,
            duckdb_probe_rows_per_file=1,
        ),
        tmp_path,
    )

    assert retriever._catalog(["school"])[0].path.name == "z_school.parquet"
    hits = retriever.retrieve("school districts", ["school"], top_k=1)
    assert hits[0].document["resource_id"] == "z_school.parquet"


def test_bounded_probe_promotes_term_found_only_in_values(tmp_path):
    pd.DataFrame({"opaque": ["unrelated"]}).to_parquet(tmp_path / "a.parquet")
    pd.DataFrame({"opaque": ["hidden narwhal"]}).to_parquet(tmp_path / "b.parquet")

    hits = DuckDBAgenticRetriever(
        RetrievalConfig(
            mode="duckdb_agentic",
            duckdb_max_files=1,
            duckdb_probe_files=1,
            duckdb_probe_rows_per_file=1,
        ),
        tmp_path,
    ).retrieve("Find narwhal records", ["narwhal"], top_k=1)

    assert hits[0].document["resource_id"] == "b.parquet"
    assert hits[0].document["duckdb_evidence"]["term_counts"]["narwhal"] == 1


def test_max_files_is_applied_after_preliminary_ranking(tmp_path):
    pd.DataFrame({"value": ["none"]}).to_parquet(tmp_path / "a.parquet")
    pd.DataFrame({"permit_number": [1]}).to_parquet(tmp_path / "z.parquet")
    retriever = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic", duckdb_max_files=1), tmp_path
    )

    catalog = retriever._catalog(["permit"])
    assert len(catalog) == 2
    assert catalog[0].path.name == "z.parquet"
    assert retriever.retrieve("permit", ["permit"], top_k=1)[0].document[
        "resource_id"
    ] == "z.parquet"


def test_value_scans_respect_max_rows_per_file(tmp_path):
    pd.DataFrame({"opaque": ["none", "none", "forbidden-tail-value"]}).to_parquet(
        tmp_path / "opaque.parquet"
    )
    hits = DuckDBAgenticRetriever(
        RetrievalConfig(
            mode="duckdb_agentic",
            duckdb_max_scan_rows_per_file=2,
            duckdb_probe_rows_per_file=2,
        ),
        tmp_path,
    ).retrieve("forbidden tail value", ["forbidden"], top_k=1)

    assert hits == []


def test_preliminary_catalog_order_is_deterministic(tmp_path):
    for name in ("c.parquet", "a.parquet", "b.parquet"):
        pd.DataFrame({"opaque": ["none"]}).to_parquet(tmp_path / name)
    retriever = DuckDBAgenticRetriever(RetrievalConfig(), tmp_path)

    first = [entry.path.name for entry in retriever._catalog(["missing"])]
    second = [entry.path.name for entry in retriever._catalog(["missing"])]
    assert first == second == ["a.parquet", "b.parquet", "c.parquet"]


# --- catalogue titles ---------------------------------------------------------
# data.gov.uk names an ArcGIS-harvested resource after its format ("CSV"), so
# taking the resource name first left ~30% of the UK lake titled "CSV" with the
# real name unused in the package title. See lakegen.core.catalogue.

def test_catalogue_title_ignores_a_resource_named_after_its_format():
    assert catalogue_title("Regions (December 2024) Boundaries EN BFC", "CSV") == (
        "Regions (December 2024) Boundaries EN BFC"
    )
    assert catalogue_title("Farm Census District Electoral Area 2019", "geojson") == (
        "Farm Census District Electoral Area 2019"
    )


def test_catalogue_title_keeps_a_resource_name_that_tells_siblings_apart():
    """1,426 UK tables share this package title; the resource name is the only
    thing separating them, so both have to survive."""
    assert catalogue_title(
        "Organogram of Staff Roles & Salaries", "2021-12-31 Organogram (Junior)"
    ) == "Organogram of Staff Roles & Salaries 2021-12-31 Organogram (Junior)"


def test_catalogue_title_does_not_repeat_a_resource_name_equal_to_the_package():
    assert catalogue_title("Air Quality 2016", "Air Quality 2016") == "Air Quality 2016"


def test_catalogue_title_falls_back_only_when_nothing_usable_remains():
    assert catalogue_title("", "CSV", fallback="air-quality-2016") == "air-quality-2016"
    assert catalogue_title(None, None) == ""


def test_catalogue_title_strips_a_trailing_file_extension():
    """924 UK resources are named after their file: the stem is the real name."""
    assert catalogue_title(
        "NI Water Consented Wastewater", "2022 NI Water Results.csv"
    ) == "NI Water Consented Wastewater 2022 NI Water Results"


def test_catalogue_title_removes_a_format_token_inside_a_real_name():
    """237 UK resources carry the format mid-name; the rest of the name stays."""
    assert catalogue_title(
        "Organogram of Staff Roles & Salaries", "Organogram - Senior CSV data"
    ) == "Organogram of Staff Roles & Salaries Organogram - Senior data"
    assert catalogue_title("Spend", "December 2020 >£25k Spend CSV") == (
        "Spend December 2020 >£25k Spend"
    )


def test_catalogue_title_drops_a_name_that_is_only_format_and_filler():
    assert catalogue_title("Spend over £25k", "CSV Download") == "Spend over £25k"
    assert catalogue_title("Spend over £25k", "Download the data file") == "Spend over £25k"


def test_is_uninformative_keeps_any_name_with_a_word_of_its_own():
    assert is_uninformative("CSV")
    assert is_uninformative("CSV Download")
    assert not is_uninformative("Organogram - Senior CSV data")
    assert not is_uninformative("2022 Results.csv")


def test_strip_format_noise_leaves_filler_inside_a_real_name():
    """"Data" decides worthlessness but is not removed from a name that has one."""
    assert strip_format_noise("Land Registry Price Paid Data") == (
        "Land Registry Price Paid Data"
    )
    assert strip_format_noise("Trees (CSV)") == "Trees"


def test_strip_format_noise_leaves_a_format_glued_to_another_word():
    """Splitting inside a token would mangle real words, so "2018CSV" and
    "csvformat" survive. Two UK resources out of 15,759, both of which still
    carry their package title."""
    assert strip_format_noise("December 2018CSV") == "December 2018CSV"
    assert strip_format_noise("sports-pitches-csvformat") == "sports-pitches-csvformat"


def test_catalogue_title_handles_missing_and_untrimmed_values():
    assert catalogue_title("  Spaced Title  ", None) == "Spaced Title"
    assert catalogue_title(None, "  2021 Return ") == "2021 Return"


def test_uk_ckan_catalog_titles_use_the_package_name_not_the_format(tmp_path):
    """An ArcGIS-harvested resource is named "CSV"; the package title is the
    only real name it has, and it has to reach the index."""
    table_dir = tmp_path / "uk" / "datasets" / "parquet"
    metadata_dir = tmp_path / "uk" / "metadata"
    table_dir.mkdir(parents=True)
    metadata_dir.mkdir()
    package_id = "33333333-3333-3333-3333-333333333333"
    resource_id = "44444444-4444-4444-4444-444444444444"
    pd.DataFrame({"lad23cd": ["E06"], "shape_area": [1.5]}).to_parquet(
        table_dir / f"{package_id}___{resource_id}.parquet"
    )
    (metadata_dir / "metadata_retrieved_only.json").write_text(json.dumps([
        {
            "id": package_id,
            "name": "lad-dec-2023-boundaries",
            "title": "Local Authority Districts (December 2023) Boundaries UK BSC",
            "notes": "Digital vector boundaries for local authority districts.",
            "organization": {"title": "Office for National Statistics"},
            "resources": [{"id": resource_id, "name": "CSV", "description": ""}],
        }
    ]), encoding="utf-8")

    hits = DuckDBAgenticRetriever(
        RetrievalConfig(mode="duckdb_agentic"), table_dir
    ).retrieve("local authority district boundaries", ["boundaries"], top_k=1)

    assert hits[0].document["title"] == (
        "Local Authority Districts (December 2023) Boundaries UK BSC"
    )
