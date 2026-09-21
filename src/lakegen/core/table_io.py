from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SUPPORTED_TABLE_SUFFIXES = {".csv", ".parquet", ".pq"}
PARQUET_SUFFIXES = {".parquet", ".pq"}


def is_supported_table(path: str | Path) -> bool:
    return Path(path).suffix.casefold() in SUPPORTED_TABLE_SUFFIXES


def list_table_files(table_dir: str | Path) -> list[str]:
    path = Path(table_dir)
    if not path.is_dir():
        return []
    return sorted(
        entry.name
        for entry in path.iterdir()
        if entry.is_file() and is_supported_table(entry)
    )


def detect_csv_separator(path: str | Path) -> str:
    try:
        with Path(path).open("r", encoding="utf-8", errors="ignore") as csv_file:
            first_line = csv_file.readline()
        if first_line.count(";") > first_line.count(","):
            return ";"
    except OSError:
        pass
    return ","


def read_table(
    path: str | Path,
    *,
    nrows: int | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    table_path = Path(path)
    suffix = table_path.suffix.casefold()

    if suffix in PARQUET_SUFFIXES:
        if nrows is None:
            return pd.read_parquet(table_path, columns=columns)
        parquet_file = pq.ParquetFile(table_path)
        batches = parquet_file.iter_batches(
            batch_size=max(1, nrows),
            columns=list(columns) if columns is not None else None,
        )
        first_batch = next(batches, None)
        if first_batch is None:
            schema_columns = list(columns) if columns is not None else parquet_file.schema.names
            return pd.DataFrame(columns=schema_columns)
        return first_batch.to_pandas().head(nrows)

    if suffix == ".csv":
        return pd.read_csv(
            table_path,
            sep=detect_csv_separator(table_path),
            nrows=nrows,
            usecols=list(columns) if columns is not None else None,
            low_memory=False,
        )

    raise ValueError(f"Unsupported table format: {table_path.suffix or '<none>'}")


def table_row_count(path: str | Path) -> int | None:
    """Return an inexpensive exact row count when the file format provides one."""
    table_path = Path(path)
    if table_path.suffix.casefold() in PARQUET_SUFFIXES:
        return pq.ParquetFile(table_path).metadata.num_rows
    return None


def read_table_sample(
    path: str | Path,
    max_rows: int,
    *,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int | None]:
    """Return ``(frame, total_rows)`` with at most ``max_rows`` rows, without loading a huge table.

    The whole table when it fits. Otherwise a parquet file with several row groups is sampled from
    row groups spaced evenly through the file (sorted data is not read from the top only, and only
    those groups are decoded); a single-row-group parquet file or a CSV can only be read from the
    top, so its first ``max_rows`` rows come back. ``total_rows`` is None when the format does not
    report it cheaply (CSV).
    """
    table_path = Path(path)
    max_rows = max(1, int(max_rows))
    total = table_row_count(table_path)
    if total is not None and total <= max_rows:
        return read_table(table_path, columns=columns), total
    if table_path.suffix.casefold() in PARQUET_SUFFIXES:
        parquet_file = pq.ParquetFile(table_path)
        groups = parquet_file.num_row_groups
        if groups > 1:
            rows_per_group = max(1, (total or max_rows) // groups)
            wanted = min(groups, max(1, -(-max_rows // rows_per_group)))
            if wanted == 1:
                picked = [groups // 2]
            else:  # spaced so that both the first and the last row group are included
                picked = sorted({round(i * (groups - 1) / (wanted - 1)) for i in range(wanted)})
            frame = parquet_file.read_row_groups(
                picked, columns=list(columns) if columns is not None else None
            ).to_pandas()
            if len(frame) > max_rows:
                frame = frame.sample(max_rows, random_state=0).reset_index(drop=True)
            return frame, total
    return read_table(table_path, nrows=max_rows, columns=columns), total


def iter_table_chunks(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    chunk_rows: int = 100_000,
) -> Iterator[pd.DataFrame]:
    table_path = Path(path)
    suffix = table_path.suffix.casefold()

    if suffix in PARQUET_SUFFIXES:
        parquet_file = pq.ParquetFile(table_path)
        for batch in parquet_file.iter_batches(
            batch_size=chunk_rows,
            columns=list(columns) if columns is not None else None,
        ):
            yield batch.to_pandas()
        return

    if suffix == ".csv":
        yield from pd.read_csv(
            table_path,
            sep=detect_csv_separator(table_path),
            usecols=list(columns) if columns is not None else None,
            chunksize=chunk_rows,
            low_memory=False,
        )
        return

    raise ValueError(f"Unsupported table format: {table_path.suffix or '<none>'}")


def table_load_command(path: str | Path) -> str:
    table_path = Path(path)
    quoted_path = repr(str(table_path))
    if table_path.suffix.casefold() in PARQUET_SUFFIXES:
        return f"pd.read_parquet({quoted_path})"
    if table_path.suffix.casefold() == ".csv":
        separator = detect_csv_separator(table_path)
        return f"pd.read_csv({quoted_path}, sep={separator!r})"
    raise ValueError(f"Unsupported table format: {table_path.suffix or '<none>'}")
