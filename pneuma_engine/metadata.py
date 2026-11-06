"""The catalogue text Pneuma indexes as each table's context.

Two metadata layouts are read: the flat one of the Socrata portals (one entry
per resource) and the nested data.gov.uk one (datasets holding resources, each
resource's file named ``<dataset_id>___<resource_id>``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def metadata_text(item: dict[str, Any]) -> str:
    """Build a resource's context text from the flat portal metadata.

    Args:
        item: One metadata entry, with ``resource`` and ``classification`` objects.

    Returns:
        Name, description, tags and column names, one per line, empty ones left out.
    """
    resource = _as_dict(item.get("resource"))
    classification = _as_dict(item.get("classification"))
    parts = [
        str(resource.get("name") or "").strip(),
        str(resource.get("description") or "").strip(),
        "Tags: " + ", ".join(map(str, classification.get("tags") or [])),
        "Columns: " + ", ".join(map(str, resource.get("columns_name") or [])),
    ]
    return "\n".join(part for part in parts if part and part not in {"Tags: ", "Columns: "})


def uk_metadata_text(dataset: dict[str, Any], resource: dict[str, Any]) -> str:
    """Build a resource's context text from the nested data.gov.uk metadata.

    Args:
        dataset: The dataset entry holding the resource.
        resource: One entry of the dataset's ``resources``.

    Returns:
        Dataset title and notes, resource name and description, publisher and
        tags, one per line, empty ones left out.
    """
    tags = dataset.get("tags") if isinstance(dataset.get("tags"), list) else []
    tag_names = [
        str(tag.get("display_name") or tag.get("name") or "").strip()
        for tag in tags
        if isinstance(tag, dict)
    ]
    organization = _as_dict(dataset.get("organization"))
    parts = [
        str(dataset.get("title") or dataset.get("name") or "").strip(),
        str(dataset.get("notes") or "").strip(),
        "Resource: " + str(resource.get("name") or "").strip(),
        str(resource.get("description") or "").strip(),
        "Publisher: " + str(organization.get("title") or organization.get("name") or "").strip(),
        "Tags: " + ", ".join(name for name in tag_names if name),
    ]
    return "\n".join(
        part for part in parts if part and part not in {"Resource: ", "Publisher: ", "Tags: "}
    )


def metadata_entries(items: list[Any], paths_by_stem: dict[str, Path]) -> list[tuple[Path, str]]:
    """Resolve flat or nested metadata entries to the registered table files.

    Args:
        items: The metadata file's entries.
        paths_by_stem: Registered table files by file stem.

    Returns:
        ``(table file, context text)`` for every entry naming a registered table
        and holding some text.
    """
    entries: list[tuple[Path, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        resource = item.get("resource")
        if isinstance(resource, dict):
            candidates = [(str(resource.get("id")), metadata_text(item))]
        else:
            candidates = _nested_candidates(item)
        entries.extend(
            (path, value)
            for stem, value in candidates
            if (path := paths_by_stem.get(stem)) is not None and value
        )
    return entries


def _nested_candidates(dataset: dict[str, Any]) -> list[tuple[str, str]]:
    """``(file stem, context text)`` for each resource of a data.gov.uk dataset."""
    dataset_id = dataset.get("id")
    resources = dataset.get("resources")
    if not dataset_id or not isinstance(resources, list):
        return []
    return [
        (f"{dataset_id}___{resource['id']}", uk_metadata_text(dataset, resource))
        for resource in resources
        if isinstance(resource, dict) and resource.get("id")
    ]


def normalize_metadata_for_pneuma(value: str) -> tuple[str, int]:
    """Escape SQL apostrophes in Pneuma's disposable metadata staging CSV.

    Pneuma 0.0.4 interpolates CSV values into a DuckDB string literal instead
    of using query parameters. DuckDB decodes each doubled apostrophe back to
    one apostrophe when storing the context, so the persisted text keeps its
    original meaning while the source metadata remains untouched.

    Args:
        value: A table ID or context text.

    Returns:
        The escaped value, and how many apostrophes were doubled.
    """
    return value.replace("'", "''"), value.count("'")
