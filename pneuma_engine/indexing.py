"""Patches that make Pneuma 0.0.4's indexing fit large lakes.

Applied before Pneuma runs, by the bootstrap and by the server:

* :func:`tune_duckdb_connections` bounds every DuckDB connection Pneuma opens:
  low-memory settings, and capped reads of the Parquet-backed external views
  that :func:`configure_bounded_external_views` names;
* :func:`configure_pneuma_indexing` inserts into Chroma in batches and recovers
  from the non-finite vectors an embedding model sometimes returns.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any

import duckdb

logger = logging.getLogger(__name__)

DEFAULT_EXTERNAL_VIEW_ROW_LIMIT = 10_000
DEFAULT_EXTERNAL_VIEW_COLUMN_LIMIT = 64
EMBEDDING_NON_FINITE_FALLBACK_PREFIX = "Represent this dataset for retrieval. "
_NON_FINITE_MARKERS = ("unsupported value: nan", "non-finite vector", "zero-norm vector")
_INSERT_DF = re.compile(r"\bFROM\s+insert_df\b", flags=re.IGNORECASE)
_SELECT_ALL = re.compile(r"\s*SELECT\s+\*\s+FROM\s+'((?:''|[^'])+)'\s*", flags=re.IGNORECASE)


@dataclass
class BoundedViews:
    """The external views whose reads are capped while tables are summarized.

    Attributes:
        table_ids: Pneuma table IDs of the bounded views.
        row_limit: Rows a read of a bounded view returns.
        column_limit: Columns a read of a wider view keeps, spread across its schema.
        sampling: ``prefix`` for a view's first rows, ``reservoir`` for a
            repeatable sample of the whole table.
        truncated_table_ids: Views whose column cut was already logged.
    """

    table_ids: set[str] = field(default_factory=set)
    row_limit: int = DEFAULT_EXTERNAL_VIEW_ROW_LIMIT
    column_limit: int = DEFAULT_EXTERNAL_VIEW_COLUMN_LIMIT
    sampling: str = "prefix"
    truncated_table_ids: set[str] = field(default_factory=set)


BOUNDED_VIEWS = BoundedViews()


class BoundedConnection:
    """A DuckDB connection that caps Pneuma's reads of bounded external views.

    Args:
        connection: The DuckDB connection every call is delegated to.
    """

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def __enter__(self) -> BoundedConnection:
        self._connection.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self._connection.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def sql(self, query: str, *args: Any, **kwargs: Any) -> Any:
        """Run ``query``, giving Pneuma's ``insert_df`` a name and bounding view reads.

        Args:
            query: The SQL Pneuma sends.
            *args: Passed on to DuckDB.
            **kwargs: Passed on to DuckDB.

        Returns:
            DuckDB's relation for the (possibly bounded) query.
        """
        if _INSERT_DF.search(query):
            self._register_insert_df(inspect.currentframe().f_back)
        match = _SELECT_ALL.fullmatch(query)
        if match and (table_id := match.group(1).replace("''", "'")) in BOUNDED_VIEWS.table_ids:
            query = self._bounded_query(query, table_id)
        return self._connection.sql(query, *args, **kwargs)

    def _register_insert_df(self, caller_frame: FrameType | None) -> None:
        """Register the DataFrame Pneuma's caller holds as ``insert_df``."""
        insert_df = caller_frame.f_locals.get("insert_df") if caller_frame is not None else None
        if insert_df is None:
            raise RuntimeError("Pneuma referenced insert_df but no caller DataFrame was found")
        self._connection.register("insert_df", insert_df)

    def _bounded_query(self, query: str, table_id: str) -> str:
        escaped_id = table_id.replace("'", "''")
        # The table ID is Pneuma's own, escaped as a DuckDB string literal.
        columns = self._connection.sql(f"SELECT * FROM '{escaped_id}' LIMIT 0").columns  # nosec B608
        if len(columns) > BOUNDED_VIEWS.column_limit:
            query = _projected_query(columns, escaped_id, table_id)
        if BOUNDED_VIEWS.sampling == "reservoir":
            # Pneuma draws its 5 row samples from these rows, so they come from
            # the whole table, the same ones on every run.
            return (
                f"{query.rstrip()} USING SAMPLE "
                f"reservoir({BOUNDED_VIEWS.row_limit} ROWS) REPEATABLE (0)"
            )
        return f"{query.rstrip()} LIMIT {BOUNDED_VIEWS.row_limit}"


def _projected_query(columns: list[str], escaped_id: str, table_id: str) -> str:
    """Select ``column_limit`` columns spread evenly across a very wide view.

    Covering the whole schema deterministically keeps a wide table from being
    summarized by its first fields only.

    Args:
        columns: The view's columns.
        escaped_id: The table ID, escaped for a DuckDB string literal.
        table_id: The table ID, for the log line.

    Returns:
        The projecting ``SELECT``, without a row bound.
    """
    last = len(columns) - 1
    limit = BOUNDED_VIEWS.column_limit
    selected = [columns[round(i * last / (limit - 1))] for i in range(limit)]
    projection = ", ".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in selected)
    if table_id not in BOUNDED_VIEWS.truncated_table_ids:
        logger.info(
            "[summarize] bounded_columns table=%s original=%d selected=%d",
            Path(table_id).name,
            len(columns),
            len(selected),
        )
        BOUNDED_VIEWS.truncated_table_ids.add(table_id)
    # Column names are quoted as identifiers, the view as a string literal.
    return f"SELECT {projection} FROM '{escaped_id}'"  # nosec B608


def tune_duckdb_connections() -> None:
    """Apply low-memory settings and view bounds to every connection opened inside Pneuma."""
    original_connect = duckdb.connect

    def connect(*args: Any, **kwargs: Any) -> BoundedConnection:
        config = dict(kwargs.pop("config", {}) or {})
        config.setdefault("threads", "4")
        config.setdefault("preserve_insertion_order", "false")
        return BoundedConnection(original_connect(*args, config=config, **kwargs))

    duckdb.connect = connect  # type: ignore[assignment]


def configure_bounded_external_views(
    db_path: Path, *, row_limit: int, column_limit: int, sampling: str = "prefix"
) -> int:
    """Bound the reads of the index's Parquet-backed external views during summaries.

    Args:
        db_path: Pneuma's ``storage.db``.
        row_limit: Rows a read of a view returns.
        column_limit: Columns a read of a wider view keeps.
        sampling: ``prefix`` or ``reservoir``; see :class:`BoundedViews`.

    Returns:
        How many views are bounded.
    """
    with duckdb.connect(str(db_path), read_only=True) as connection:
        table_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM table_status WHERE hash LIKE 'external:%'"
            ).fetchall()
        ]
    BOUNDED_VIEWS.table_ids = {
        table_id
        for table_id in table_ids
        if Path(table_id).is_file() and Path(table_id).suffix.casefold() == ".parquet"
    }
    BOUNDED_VIEWS.row_limit = row_limit
    BOUNDED_VIEWS.column_limit = column_limit
    BOUNDED_VIEWS.sampling = sampling
    return len(BOUNDED_VIEWS.table_ids)


def validate_embedding(vector: Any) -> list[float]:
    """Check that an embedding is usable for vector search.

    Args:
        vector: The embedding as the provider returned it.

    Returns:
        The embedding as a list of floats.

    Raises:
        ValueError: If it is empty, holds a non-finite value or has zero norm.
    """
    result = [float(value) for value in vector]
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("Embedding model returned an empty or non-finite vector")
    if math.sqrt(sum(value * value for value in result)) < 1e-12:
        raise ValueError("Embedding model returned a zero-norm vector")
    return result


def _is_non_finite_embedding_error(error: BaseException) -> bool:
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None:
        messages.append(str(current).casefold())
        current = current.__cause__
    detail = " ".join(messages)
    return any(marker in detail for marker in _NON_FINITE_MARKERS)


def _embed(embed_model: Any, items: list[str], model: str) -> list[list[float]]:
    """Embed ``items``, isolating and recovering the ones that come back non-finite.

    A failing batch is halved until the failing text stands alone; that text is
    embedded again with a neutral prefix, which moves the model off the NaN.

    Args:
        embed_model: The OpenAI-compatible client Pneuma embeds with.
        items: Texts to embed.
        model: Embedding model name.

    Returns:
        One vector per text, in order.

    Raises:
        RuntimeError: If the provider returns a different number of vectors.
    """
    try:
        response = embed_model.embeddings.create(input=items, model=model)
        vectors = [validate_embedding(item.embedding) for item in response.data]
        if len(vectors) != len(items):
            raise RuntimeError(
                "Embedding provider returned a different number of vectors "
                f"({len(vectors)}) than inputs ({len(items)})"
            )
        return vectors
    except Exception as error:
        if not _is_non_finite_embedding_error(error):
            raise
        if len(items) > 1:
            midpoint = len(items) // 2
            return _embed(embed_model, items[:midpoint], model) + _embed(
                embed_model, items[midpoint:], model
            )
        original = items[0]
        response = embed_model.embeddings.create(
            input=[EMBEDDING_NON_FINITE_FALLBACK_PREFIX + original], model=model
        )
        if len(response.data) != 1:
            raise RuntimeError(
                "Embedding provider did not return one vector for the non-finite fallback"
            ) from error
        vector = validate_embedding(response.data[0].embedding)
        logger.warning(
            "[index] recovered_non_finite_embedding sha256=%s",
            hashlib.sha256(original.encode("utf-8")).hexdigest(),
        )
        return [vector]


def _add_in_batches(add: Any, collection: Any, batch_size: int, **columns: Any) -> None:
    """Call Chroma's ``add`` on at most ``batch_size`` documents at a time."""
    ids = columns["ids"]
    item_count = 1 if isinstance(ids, str) else len(ids)
    if item_count <= batch_size:
        add(collection, **columns)
        return
    logger.info("[index] chroma_insert documents=%d batch_size=%d", item_count, batch_size)
    for start in range(0, item_count, batch_size):
        end = min(start + batch_size, item_count)
        add(
            collection,
            **{
                name: None if value is None else value[start:end] for name, value in columns.items()
            },
        )
        logger.info("[index] chroma_insert %d/%d", end, item_count)


def configure_pneuma_indexing(*, embedding_batch_size: int, chroma_insert_batch_size: int) -> None:
    """Make Pneuma's indexing robust to non-finite embeddings and Chroma batch limits.

    Args:
        embedding_batch_size: Documents sent per embedding request.
        chroma_insert_batch_size: Documents inserted per Chroma operation.
    """
    import pneuma.index_generator.index_generator as pneuma_index_generator

    original_add = pneuma_index_generator.Collection.add

    def chunked_collection_add(
        collection: Any,
        ids: Any,
        embeddings: Any = None,
        metadatas: Any = None,
        documents: Any = None,
        images: Any = None,
        uris: Any = None,
    ) -> None:
        _add_in_batches(
            original_add,
            collection,
            chroma_insert_batch_size,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            images=images,
            uris=uris,
        )

    def robust_prompt_openai_embed(
        embed_model: Any, documents: list[str], model: str = "text-embedding-3-small"
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(documents), embedding_batch_size):
            vectors.extend(
                _embed(embed_model, documents[offset : offset + embedding_batch_size], model)
            )
        return vectors

    pneuma_index_generator.Collection.add = chunked_collection_add
    pneuma_index_generator.prompt_openai_embed = robust_prompt_openai_embed
