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
