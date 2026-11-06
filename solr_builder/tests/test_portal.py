import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from solr_builder import portal as docs


def test_clean_html_returns_plain_text():
    raw = (
        "<div style='font-family:Lato, &quot;Avenir&quot;'><p>Boundaries&nbsp;for "
        '<a href="https://x">LSOAs</a></p>\n<br/>Fish &amp; chips</div>'
    )

    assert docs.clean_html(raw) == "Boundaries for LSOAs Fish & chips"
    assert docs.clean_html(None) == ""


def test_clean_html_drops_style_blocks_and_keeps_comparisons():
    raw = "<style>p {color: red}</style>Aged <5 years <!-- hidden --> and a < b"

    assert docs.clean_html(raw) == "Aged <5 years and a < b"
    assert not docs.has_markup(docs.clean_html(raw))
    assert docs.has_markup("<span>x</span>")


def test_merge_descriptions_says_a_repeated_text_once():
    assert docs.merge_descriptions("<p>Same text.</p>", "Same text.") == "Same text."
    assert docs.merge_descriptions("Package notes.", "This file: March.") == (
        "Package notes. This file: March."
    )
    assert docs.merge_descriptions("Short.", "Short. And more.") == "Short. And more."
    assert docs.merge_descriptions(None, "") == ""


def test_empty_parquet_column_name_is_aligned_with_what_duckdb_exposes(tmp_path):
    path = tmp_path / "table.parquet"
    pq.write_table(pa.table({"a": [1], "": [2], "b": ["x"]}), path)
    names = [field.name for field in pq.ParquetFile(path).schema_arrow]
    engine = [
        c[0]
        for c in duckdb.connect()
        .execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)])
        .description
    ]

    assert docs.needs_engine_names(names)
    assert docs.resolve_column_names(names, engine) == engine == ["a", "C1", "b"]
    # Without DuckDB's answer the same rule is applied directly.
    assert docs.resolve_column_names(names) == ["a", "C1", "b"]
    assert docs.resolve_column_names(["a", "b"]) == ["a", "b"]
    assert docs.resolve_column_names(["a", " "]) == ["a", "C1"]


def test_uk_document_names_the_table_and_keeps_column_lists_aligned():
    package = {
        "id": "pkg-1",
        "name": "regions",
        "title": "Regions (December 2024) Boundaries EN BFC",
        "notes": "<p>Boundaries&nbsp;of regions</p>",
        "organization": {"title": "Office for National Statistics"},
        "tags": [{"display_name": "geo"}, {"name": "geo"}, {"name": "boundaries"}],
        "metadata_created": "2024-01-02T03:04:05",
    }
    resource = {
        "id": "res-1",
        "name": "CSV",
        "description": "",
        "url": "https://example.org/regions.csv",
        "created": "2024-01-02T03:04:05Z",
    }

    document = docs.build_uk_document(
        package,
        resource,
        column_names=["RGN24CD", "C1"],
        column_types=["large_string", "int64"],
        generation="2026-09-21T20:00:00.000Z",
    )

    assert document["title"] == "Regions (December 2024) Boundaries EN BFC"
    assert document["description"] == "Boundaries of regions"
    assert document["tags"] == ["geo", "boundaries"]
    assert document["columns.name"] == document["columns.label"] == ["RGN24CD", "C1"]
    assert document["columns.type"] == ["large_string", "int64"]
    assert document["metadata_created"] == "2024-01-02T03:04:05.000Z"
    assert document["dataset_url"] == "https://www.data.gov.uk/dataset/pkg-1"
    assert document["source"] == "UK Open Data" and document["format"] == "parquet"
    assert document["indexed_ts"] == "2026-09-21T20:00:00.000Z"
    assert "modified_at" not in document  # absent values are not written


def test_uk_document_rejects_misaligned_columns():
    with pytest.raises(ValueError, match="same length"):
        docs.build_uk_document(
            {"id": "p"},
            {"id": "r"},
            column_names=["a"],
            column_types=[],
            generation="g",
        )
