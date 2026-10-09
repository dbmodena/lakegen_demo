"""The data lakes: their tables on disk and their catalogue entries; in time the ``Lake`` of the registry.

``get_all_table_files(table_dir)`` lists a lake's tables as the pipeline
sees them; ``catalogue`` maps a catalogue document to its local file and to
the metadata the agents read. The readers are in ``table_io``.
"""

from __future__ import annotations

from pathlib import Path

from lakegen.lakes.catalogue import match_local_csv, solr_metadata_from_doc
from lakegen.lakes.table_io import list_table_files


def get_all_table_files(table_dir: str | Path) -> list[str]:
    return list_table_files(table_dir)


# The name callers outside this repository use.
get_all_csv_files = get_all_table_files

__all__ = ["get_all_csv_files", "get_all_table_files", "list_table_files", "match_local_csv", "solr_metadata_from_doc"]
