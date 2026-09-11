"""Versioned textual representations of table metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


METADATA_V1 = "metadata-v1"


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    values: Iterable[Any] = value if isinstance(value, (list, tuple)) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def metadata_v1(document: dict[str, Any]) -> str:
    """Serialize title, description, tags, and schema metadata.

    This intentionally excludes table rows. Keeping the representation named
    and stable makes later TARGET-style ablations (title-only, schema-only,
    sampled rows, or richer metadata) comparable.
    """
    columns = document.get("columns")
    if isinstance(columns, list):
        column_names = _strings(
            [column.get("name") for column in columns if isinstance(column, dict)]
        )
        column_descriptions = _strings(
            [
                column.get("description")
                for column in columns
                if isinstance(column, dict)
            ]
        )
    else:
        column_names = _strings(
            document.get("columns.name") or document.get("columns_name")
        )
        column_descriptions = _strings(
            document.get("columns.description")
            or document.get("columns_description")
        )

    sections = [
        ("Title", _strings(document.get("title"))),
        ("Description", _strings(document.get("description"))),
        ("Tags", _strings(document.get("tags"))),
        ("Column names", column_names),
        ("Column descriptions", column_descriptions),
    ]
    rendered_sections = [
        f"{label}: {' | '.join(values)}" for label, values in sections if values
    ]
    if not rendered_sections:
        return ""
    return "\n".join(
        (
            "Represent this table metadata for information retrieval.",
            *rendered_sections,
        )
    )


REPRESENTATION_BUILDERS = {METADATA_V1: metadata_v1}


def represent_table(document: dict[str, Any], version: str = METADATA_V1) -> str:
    try:
        builder = REPRESENTATION_BUILDERS[version]
    except KeyError as exc:
        raise ValueError(f"Unknown table representation version: {version!r}") from exc
    return builder(document)
