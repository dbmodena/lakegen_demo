"""A lake's catalogue entries: the Solr document of a table, its local file, and its metadata for the agents."""

from __future__ import annotations

from pathlib import Path

from lakegen.core.catalogue import clean_catalogue_text


def match_local_csv(doc: dict, all_files: list[str]) -> str | None:
    dataset_id = str(doc.get("dataset_id") or "").strip()
    resource_id = str(doc.get("resource_id") or "").strip()
    # The file named after this exact resource wins: UK files are named
    # <dataset_id>___<resource_id>, NYC files <resource_id>. A substring match
    # alone maps every resource of a UK dataset to the dataset's first file, so
    # it is only a fallback, and the resource id is tried before the dataset id.
    stems = {resource_id.casefold()} if resource_id else set()
    if dataset_id and resource_id:
        stems.add(f"{dataset_id}___{resource_id}".casefold())
    if stems:
        exact = next(
            (name for name in all_files if Path(name).stem.casefold() in stems),
            None,
        )
        if exact is not None:
            return exact
    for identifier in (resource_id, dataset_id):
        if identifier:
            found = next((name for name in all_files if identifier in name), None)
            if found is not None:
                return found
    return None


def text(value: object) -> str:
    """``str`` that leaves a missing value empty instead of spelling it "None"."""
    return "" if value is None else str(value).strip()


def solr_metadata_from_doc(doc: dict) -> dict[str, object]:
    tags = doc.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]

    columns = doc.get("columns", [])
    if not isinstance(columns, list):
        columns = []
    # Solr documents restore an absent column description as None, and
    # ``.get(key, "")`` only defaults an absent key, so a bare ``str`` here
    # printed "None" after every column in the agent's context.
    structured_columns = [
        {
            "name": text(column.get("name")),
            "description": clean_catalogue_text(column.get("description")),
            "type": text(column.get("type")),
        }
        for column in columns
        if isinstance(column, dict) and text(column.get("name"))
    ]
    # Catalogue text is cleaned once, here, where every search path turns a
    # hit into metadata, so the agent never reads portal HTML. Column names
    # and types are left as indexed: they must match the real schema.
    return {
        "title": clean_catalogue_text(doc.get("title")),
        "publisher": clean_catalogue_text(doc.get("publisher")),
        "resource_name": clean_catalogue_text(
            doc.get("resource_name", doc.get("title"))
        ),
        "description": clean_catalogue_text(doc.get("description")),
        "tags": [clean_catalogue_text(tag) for tag in tags],
        "columns.name": [column["name"] for column in structured_columns],
        "columns.description": [
            column["description"]
            for column in structured_columns
            if column["description"]
        ],
        "columns.type": [
            column["type"] for column in structured_columns if column["type"]
        ],
        # Retain per-column alignment for the bounded agent-facing preview.
        "columns": structured_columns,
        "column_count": len(columns),
    }
