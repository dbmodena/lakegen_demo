"""From Pneuma's path-like table ids back to LakeGen's catalogue documents."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

DocumentResolver = Callable[[str], dict[str, Any] | None]


class SolrPneumaDocumentResolver:
    """Map Pneuma's path-like table IDs back to LakeGen catalog documents.

    Every document of the core is read once and indexed under its aliases: the
    id fields, the URL fields, their file names and stems, and the
    ``<dataset_id>___<resource_id>`` stem of lakes that group resources.

    Args:
        solr: The lake's Solr client; only ``iter_documents`` is used.

    Attributes:
        FIELDS: The document fields read.
        ALIAS_FIELDS: The fields a table id can be matched on.
    """

    FIELDS = (
        "id",
        "resource_id",
        "dataset_id",
        "title",
        "description",
        "tags",
        "columns.name",
        "columns.description",
        "columns.type",
        "schema",
        "dataset_url",
        "download_url",
        "url",
        "permalink",
        "link",
        "source",
        "portal",
        "provenance",
    )

    ALIAS_FIELDS = ("id", "resource_id", "dataset_id", "download_url", "url", "source")

    def __init__(self, solr: Any) -> None:
        # Indexed by alias rather than scanned per lookup: content search asks
        # this question once per candidate file, not once per Pneuma hit. The
        # catalog position is kept so a contested alias still resolves to the
        # document a linear scan would have found first -- ``_aliases`` returns a
        # set, whose iteration order is not stable across processes.
        self._index: dict[str, tuple[int, dict[str, Any]]] = {}
        for position, document in enumerate(
            solr.iter_documents(fields=self.FIELDS, sort_field="resource_id")
        ):
            for field_name in self.ALIAS_FIELDS:
                for alias in self._aliases(document.get(field_name)):
                    self._index.setdefault(alias, (position, document))
            # Lakes that group resources under a dataset (UK) name each file
            # ``<dataset_id>___<resource_id>``, a stem no single field holds.
            # Pneuma returns that file's path, so the pair is an alias too.
            file_stem = self._file_stem(document)
            if file_stem:
                self._index.setdefault(file_stem, (position, document))

    @staticmethod
    def _file_stem(document: dict[str, Any]) -> str:
        """``<dataset_id>___<resource_id>``, casefolded; empty without both."""
        dataset_id = str(document.get("dataset_id") or "").strip()
        resource_id = str(document.get("resource_id") or "").strip()
        if not dataset_id or not resource_id:
            return ""
        return f"{dataset_id}___{resource_id}".casefold()

    @staticmethod
    def _aliases(value: Any) -> set[str]:
        """The value, its file name and its stem, casefolded; none for a blank value."""
        if value is None or not str(value).strip():
            return set()
        text = str(value).strip()
        path = Path(text)
        return {text.casefold(), path.name.casefold(), path.stem.casefold()}

    def __call__(self, table_id: str) -> dict[str, Any] | None:
        """The catalog document of ``table_id``, the earliest in the catalog on a tie; None when unknown."""
        found = [self._index[alias] for alias in self._aliases(table_id) if alias in self._index]
        if not found:
            return None
        return min(found, key=lambda item: item[0])[1]
