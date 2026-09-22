"""Solr documents for the UK portal, built from CKAN metadata and parquet schemas.

The UK index holds one document per parquet file. Three details of how a
document is built decide whether the table can be found and understood, and
each was wrong in the first UK build:

* ``title`` names the family and the file (see ``lakegen.core.catalogue``), not
  just the resource name, which for ArcGIS harvests is the literal "CSV".
* ``description`` is plain text. CKAN notes are HTML, and left as they are the
  markup is indexed as words ("div", "span", "style") and shown to the agent.
* ``columns.name``, ``columns.label`` and ``columns.type`` are parallel lists
  that ``LocalSolrClient`` zips back into columns by position. Solr silently
  drops an empty-string value, so a parquet column with an empty name shifted
  every later type onto the wrong column.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import html
import re
from typing import Any

from lakegen.core.catalogue import catalogue_title

SOURCE_LABEL = "UK Open Data"
DOCUMENT_FORMAT = "parquet"

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLOCK_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# A tag starts with a letter or "/". "<5 years" and "a < b" are text.
_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_html(value: Any) -> str:
    """Plain text of a CKAN description.

    The same steps orqa applies to its own metadata (unescape, drop tags,
    collapse whitespace), so questions written against orqa's text match the
    text indexed here. Two deliberate differences: the content of ``<script>``
    and ``<style>`` blocks is dropped rather than kept as words, and only real
    tags are removed, so ``"<5 years"`` survives.
    """
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = _COMMENT_RE.sub(" ", text)
    text = _BLOCK_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def has_markup(text: str) -> bool:
    """Does this text still contain HTML tags or comments?"""
    return bool(_TAG_RE.search(text) or _COMMENT_RE.search(text))


def merge_descriptions(*texts: Any) -> str:
    """Join descriptions, dropping one that another already contains.

    CKAN often repeats the package notes as the resource description; joining
    them naively says everything twice.
    """
    parts: list[str] = []
    for text in (clean_html(item) for item in texts):
        if not text:
            continue
        folded = text.casefold()
        if any(folded in part.casefold() for part in parts):
            continue
        parts = [part for part in parts if part.casefold() not in folded]
        parts.append(text)
    return " ".join(parts)


def solr_timestamp(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def named_values(values: Any) -> list[str]:
    """Distinct display names of CKAN tags (or any list of named objects)."""
    result: list[str] = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, dict):
            value = value.get("display_name") or value.get("title") or value.get("name")
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def needs_engine_names(names: Sequence[str]) -> bool:
    """True when the SQL engine will not expose these column names as they are."""
    return any(not name.strip() for name in names) or len(set(names)) != len(names)


def resolve_column_names(
    names: Sequence[str], engine_names: Sequence[str] | None = None
) -> list[str]:
    """Column names as the agent's SQL engine sees them, aligned by position.

    A parquet file may hold a column named "". DuckDB exposes it as ``C<index>``;
    Solr cannot store it at all. ``engine_names`` is the list DuckDB reports for
    the same file; without it DuckDB's rule for an empty name is applied
    directly. A name that is still blank gets the same fallback, so the list
    never contains a value Solr would drop.
    """
    if not needs_engine_names(names):
        return list(names)
    resolved = (
        list(engine_names)
        if engine_names is not None and len(engine_names) == len(names)
        else list(names)
    )
    return [name if name.strip() else f"C{index}" for index, name in enumerate(resolved)]


def build_uk_document(
    package: dict[str, Any],
    resource: dict[str, Any],
    *,
    column_names: Sequence[str],
    column_types: Sequence[str],
    generation: str,
) -> dict[str, Any]:
    """The Solr document for one CKAN resource stored as one parquet file."""
    if len(column_names) != len(column_types):
        raise ValueError("column_names and column_types must be the same length")
    package_id = str(package.get("id") or "").strip()
    organization = package.get("organization") or {}
    names = list(column_names)
    document = {
        "dataset_id": package_id,
        "resource_id": str(resource.get("id") or "").strip(),
        "source": SOURCE_LABEL,
        "title": catalogue_title(
            package.get("title"), resource.get("name"), fallback=package.get("name")
        ),
        "description": merge_descriptions(
            package.get("notes"), resource.get("description")
        ),
        "publisher": str(
            organization.get("title") or organization.get("name") or ""
        ).strip(),
        "tags": named_values(package.get("tags")),
        "created_at": solr_timestamp(resource.get("created")),
        "modified_at": solr_timestamp(resource.get("metadata_modified")),
        "metadata_created": solr_timestamp(package.get("metadata_created")),
        "metadata_modified": solr_timestamp(package.get("metadata_modified")),
        "dataset_url": f"https://www.data.gov.uk/dataset/{package_id}",
        "download_url": str(resource.get("url") or "").strip(),
        "format": DOCUMENT_FORMAT,
        "columns.name": names,
        "columns.label": list(names),
        "columns.type": list(column_types),
        "indexed_ts": generation,
    }
    return {key: value for key, value in document.items() if value not in (None, "", [])}
