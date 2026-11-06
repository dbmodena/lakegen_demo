from solr_builder.build_portal_core import solr_document
from solr_builder.build_uk_fixed import diff_report


def test_portal_document_keeps_column_lists_aligned_and_drops_empty_values():
    record = {
        "resource_id": "r1",
        "title": "Trees",
        "notes": "",
        "created_at": "2024-03-01T10:00:00",
        "columns": [{"name": "id", "type": "number"}, {"label": "Species"}],
    }

    document = solr_document(record, "2026-10-29T00:00:00Z")

    assert "notes" not in document and "columns" not in document
    assert document["columns.name"] == ["id", "column_1"]
    assert document["columns.label"] == ["id", "Species"]
    assert document["columns.type"] == ["number", "text"]
    assert "columns.description" not in document  # not every column has one
    assert document["indexed_ts"] == "2026-10-29T00:00:00Z"


def test_uk_diff_names_new_stale_and_changed_documents():
    current = {
        "a": {"resource_id": "a", "title": "Old", "description": "<p>Text</p>"},
        "gone": {"resource_id": "gone", "title": "Stale", "representation_version": "v1"},
    }
    built = [
        {"resource_id": "a", "title": "New", "description": "Text"},
        {"resource_id": "b", "title": "Added"},
    ]

    report = diff_report(current, built)

    assert report["new_ids"] == 1 and report["stale_ids"] == ["gone"]
    assert report["fields_changed"] == {"title": 1, "description": 1}
    assert report["description_changes"] == {"markup or entities removed": 1}
    assert report["without_vector_but_labelled"] == 1
