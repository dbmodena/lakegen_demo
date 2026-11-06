"""Table identifiers: the forms a table is named by, and the one the metrics compare.

A lake's table is a file: ``<dataset>___<resource>.parquet`` on the CKAN
portals, ``<name>.parquet`` elsewhere. A benchmark's gold ids and the
retrieval rankings name the resource part alone, so a selected table is
reduced to its resource id before it is compared with them.

These are the API's normalisers, the ones every stored metric was computed
with. The paper analysis (``records.resource_id``, which also strips
whitespace) and the service's CSV log (a regex over the suffixes, gold ids
taken whole) agree with them on every id stored in the jobs under
/data/bilel/lakegen/jobs (checked on 2026-10-08: 1,090 selected tables, 3,084
ranked ids, 1,354 gold ids), which is why they are not kept as variants.
"""

from __future__ import annotations

from pathlib import Path

TABLE_SUFFIXES = frozenset({".csv", ".parquet", ".pq"})


def table_id(value: object) -> str:
    """A table file's id: its name without a table suffix, whatever the suffix's case.

    A path keeps only its last part; a name with any other suffix is kept whole.

    Args:
        value: A table name or path.

    Returns:
        The id.
    """
    name = Path(str(value)).name
    return Path(name).stem if Path(name).suffix.casefold() in TABLE_SUFFIXES else name


def resource_id(value: object) -> str:
    """The resource part of a table id, the form gold ids and rankings share.

    What follows the last ``___`` of a ``dataset___resource`` name; the whole
    id when there is none.

    Args:
        value: A table name, path or id.

    Returns:
        The resource id.
    """
    return table_id(value).rsplit("___", 1)[-1]
