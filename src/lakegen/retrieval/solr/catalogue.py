"""A lake's catalogue entries: the Solr document of a table, its local file, and its metadata for the agents.

The entry carries the titles a portal has (``title``; ``resource_name`` when the file has a name of its own, as
Valencia's Valencian names; ``dataset_title`` when the indexed title is the file's name and the lake's own catalogue
names the dataset, as on data.gov.uk) and who stands behind the table (``publisher``, and the ODS portals'
``responsible_entity``), whole: the selector shows them as they are.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from lakegen.utils.naming import clean_catalogue_text

logger = logging.getLogger(__name__)


def match_local_csv(doc: dict, all_files: list[str]) -> str | None:
    """The local table file of a catalogue document.

    The file named after the exact resource first, then the first file whose
    name contains the resource id, then the dataset id.

    Args:
        doc: The Solr document (``resource_id``, ``dataset_id``).
        all_files: The lake's table file names.

    Returns:
        The file name, or None when no file matches.
    """
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


def single_text(value: object) -> str:
    """Catalogue text that Solr may hold as a list (Valencia's resource names): its items joined, cleaned."""
    if isinstance(value, (list, tuple)):
        parts = [clean_catalogue_text(item) for item in value]
        return " / ".join(dict.fromkeys(part for part in parts if part))
    return clean_catalogue_text(value)


def solr_metadata_from_doc(doc: dict) -> dict[str, object]:
    """The metadata the agents read for a table, from its Solr document, catalogue text cleaned.

    Args:
        doc: The Solr document.

    Returns:
        Titles, publisher, description, tags and the columns (names, descriptions,
        types, and the aligned per-column entries).
    """
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
        "title": single_text(doc.get("title")),
        "publisher": single_text(doc.get("publisher")),
        "resource_name": single_text(doc.get("resource_name", doc.get("title"))),
        "responsible_entity": single_text(doc.get("responsible_entity")),
        "description": clean_catalogue_text(doc.get("description")),
        "tags": [clean_catalogue_text(tag) for tag in tags],
        "columns.name": [column["name"] for column in structured_columns],
        "columns.description": [
            column["description"] for column in structured_columns if column["description"]
        ],
        "columns.type": [column["type"] for column in structured_columns if column["type"]],
        # Retain per-column alignment for the bounded agent-facing preview.
        "columns": structured_columns,
        "column_count": len(columns),
    }


# --- the dataset's own title, from the lake's catalogue ---------------------
#
# The UK core indexes each file under its resource name ("CSV" for 4,545 of
# them); the dataset's title lives in the lake's CKAN package list. The divided
# pipeline reads that list once per process and gives such entries a
# ``dataset_title``, and a publisher when the document has none.


@lru_cache(maxsize=8)
def _ckan_titles(path: str, mtime: float) -> dict[str, tuple[str, str]]:
    try:
        packages = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("catalogue %s not read: %s", path, exc)
        return {}
    if isinstance(packages, dict):
        packages = packages.get("datasets") or packages.get("result") or list(packages.values())
    titles: dict[str, tuple[str, str]] = {}
    for package in packages if isinstance(packages, list) else []:
        if not isinstance(package, dict) or not package.get("id") or not package.get("title"):
            continue
        organization = package.get("organization")
        publisher = (
            organization.get("title") or organization.get("name")
            if isinstance(organization, dict)
            else ""
        )
        titles[str(package["id"])] = (
            clean_catalogue_text(package["title"]),
            clean_catalogue_text(publisher),
        )
    return titles


def ckan_titles(path: Path) -> dict[str, tuple[str, str]]:
    """Dataset id -> (title, publisher) from a CKAN package list; {} when the file is missing or not one."""
    path = Path(path)
    if not path.is_file():
        return {}
    return _ckan_titles(str(path), path.stat().st_mtime)


def lake_titles(lake: str) -> dict[str, tuple[str, str]]:
    """The dataset titles of a lake's catalogue file; {} for an unknown lake or one without a CKAN list."""
    from lakegen.config.registry import get_lake

    try:
        return ckan_titles(get_lake(lake).metadata_path)
    except KeyError:
        return {}


def with_dataset_title(
    metadata: dict[str, object], doc: dict, titles: dict[str, tuple[str, str]]
) -> dict[str, object]:
    """Add the dataset's title (``dataset_title``) when it is not the indexed title, and a publisher when missing."""
    found = titles.get(str(doc.get("dataset_id") or "").strip()) if titles else None
    if not found:
        return metadata
    title, publisher = found
    if title and title.casefold() != str(metadata.get("title") or "").casefold():
        metadata["dataset_title"] = title
    if publisher and not metadata.get("publisher"):
        metadata["publisher"] = publisher
    return metadata
