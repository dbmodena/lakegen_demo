"""The data lakes: their tables on disk, and in time the ``Lake`` of the registry.

``get_all_table_files(table_dir)`` lists a lake's tables as the pipeline
sees them. The readers stay in ``lakegen.core.table_io`` until they move here.
"""

from __future__ import annotations

from pathlib import Path

from lakegen.core.table_io import list_table_files


def get_all_table_files(table_dir: str | Path) -> list[str]:
    return list_table_files(table_dir)


# The name callers outside this repository use.
get_all_csv_files = get_all_table_files

__all__ = ["get_all_csv_files", "get_all_table_files", "list_table_files"]
