import logging
import string
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import polars as pl
import polars.selectors as cs

whitespace_translator = str.maketrans(string.whitespace, " " * len(string.whitespace))

LRU_CACHE_SIZE = 1024


@lru_cache(maxsize=LRU_CACHE_SIZE)
def calculate_xash(token: str, hash_size: int = 128) -> int:
    """Calculates the XASH hash of a token.

    Setting is the same as provided by XASH/MATE authors.

    Args:
        token: The input string token to hash.
        hash_size: The size of the hash in bits (default: 128).

    Returns:
        The calculated XASH integer.
    """

    number_of_ones = 5
    char = list(string.ascii_lowercase + string.digits + " ")

    segment_size_dict = {64: 1, 128: 3, 256: 6, 512: 13}
    segment_size = segment_size_dict[hash_size]

    n_bits_for_chars = 37 * segment_size
    length_bit_start = n_bits_for_chars
    n_bits_for_length = hash_size - length_bit_start
    token_size = len(token)

    # - Character position encoding
    result = 0
    # Pick the 5 most infrequent characters
    counts = Counter(token).items()
    sorted_counts = sorted(counts, key=lambda char_occurances: char_occurances[::-1])
    selected_chars = [char for char, _ in sorted_counts[:number_of_ones]]
    # Encode the position of the 5 most infrequent characters
    for c in selected_chars:
        if c not in char:
            continue
        # Calculate the mean position of the character and set the one bit in the corresponding segment
        indices = [i for i, ltr in enumerate(token) if ltr == c]
        mean_index = sum(indices) / len(indices)
        normalized_mean_index = mean_index / token_size
        segment = max(int(normalized_mean_index * segment_size - 1e-6), 0)  # Legacy fix
        location = char.index(c) * segment_size + segment
        result = result | 2**location

    # Rotate position encoding
    shift_distance = (
        length_bit_start
        * (token_size % (hash_size - length_bit_start))
        // (hash_size - length_bit_start)
    )
    left_bits = result << shift_distance
    wrapped_bits = result >> (n_bits_for_chars - shift_distance)
    cut_overlapping_bits = 2**n_bits_for_chars

    result = (left_bits | wrapped_bits) % cut_overlapping_bits

    # - Add length bit
    length_bit = 1 << (length_bit_start + token_size % n_bits_for_length)
    result = result | length_bit

    return result


def init_logger(logfile: Optional[Path] = None, stdout: bool = False):
    """Initializes the BLEND logger.

    Args:
        logfile: Optional path to a log file.
        stdout: Whether to log to stdout (default: False).

    Returns:
        The configured logger object.
    """
    logger = logging.getLogger("BLEND")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    if logfile and not any(
        isinstance(handler, logging.FileHandler) for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(logging.DEBUG)  # Set minimum level for file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout and not any(
        isinstance(handler, logging.StreamHandler) for handler in logger.handlers
    ):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)  # Set minimum level for console
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger


@lru_cache(maxsize=LRU_CACHE_SIZE)
def clean(
    s: Any,
    lowercase: bool = False,
    replace_whitespaces: bool = False,
    bad_tokens: Optional[tuple[str]] = None,
):
    """Cleans a string or other value.

    Args:
        s: The input value to clean.
        lowercase: Whether to convert to lowercase (default: False).
        replace_whitespaces: Whether to replace all whitespaces with spaces (default: False).
        bad_tokens: A tuple of tokens to consider as empty/bad. Defaults to ("nan", "null", "none").

    Returns:
        The cleaned string, or an empty string if it's a bad token.
    """
    if bad_tokens is None:
        bad_tokens = ("nan", "null", "none")

    s = str(s)
    if lowercase:
        s = s.lower()
    if replace_whitespaces:
        s = s.translate(whitespace_translator)
    if s in bad_tokens:
        return ""
    return s.strip()


def _truncate(s: str, max_length: int | None = 128) -> str:
    """Truncates a string to the specified length.

    Args:
        s: The string to truncate.
        max_length: The maximum length. If positive, keep the first max_length characters.
            If negative, keep the last max_length characters.
            If None, keep the whole string.

    Returns:
        The truncated string.
    """
    if max_length is None:
        return s

    if max_length > 0:
        return s[:max_length]

    return s[max_length:]


def _clean(
    column_name: str,
    lowercase: bool = True,
    replace_whitespaces: bool = True,
    filter_bad_tokens: bool = True,
    bad_tokens: Optional[list[str]] = None,
) -> pl.Expr:
    """Creates a Polars expression to clean a column.

    Args:
        column_name: The name of the column to clean.
        lowercase: Whether to convert to lowercase (default: True).
        replace_whitespaces: Whether to replace newlines/tabs with spaces (default: True).
        filter_bad_tokens: Whether to filter out bad tokens (default: True).
        bad_tokens: List of tokens to filter out. Defaults to ["nan", "null", "none"].

    Returns:
        A Polars expression for cleaning the column.
    """
    if not bad_tokens:
        bad_tokens = ["nan", "null", "none"]

    e = pl.col(column_name).cast(pl.String)

    if lowercase:
        e = e.str.to_lowercase()
    if replace_whitespaces:
        e = e.str.replace_all(r"[\t\n\r]", " ")
        e = e.str.strip_chars()
    if filter_bad_tokens:
        e = pl.when(e.is_in(set(bad_tokens))).then(pl.lit("")).otherwise(e)

    return e


def remove_null_rows(df: pl.DataFrame, *exclude_columns) -> pl.DataFrame:
    return df.filter(~pl.all_horizontal(pl.all().exclude(*exclude_columns).is_null()))


def remove_null_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df[[s.name for s in df if not (s.null_count() == df.height)]]


def load_table(
    path: Path, opts: Optional[dict] = None, lazy: bool = False
) -> pl.DataFrame | pl.LazyFrame:
    if opts is None:
        opts = {}
    format_ = path.suffix.replace(".", "")

    match format_:
        case "csv":
            return pl.scan_csv(path, **opts) if lazy else pl.read_csv(path, **opts)
        case "parquet":
            return (
                pl.scan_parquet(path, **opts) if lazy else pl.read_parquet(path, **opts)
            )
        case _:
            raise ValueError(f"Unknown table format in {path}: {format_}")


def parse_table(
    path: Path,
    load_opts: Optional[dict] = None,
    clean_args: Optional[dict] = None,
    xash_size: int = 128,
    max_cell_length: int = 128,
) -> tuple[str, pl.DataFrame | str]:
    """Load and parse the table at the specified path.

    The output table is in the DataXFormer format, plus the
    columns introduced by the BLEND authors:

    | TableId | ColumnId | RowId | CellValue | Quadrant | SuperKey |

    That is, each cell is reported along with its column and row IDs,
    its quadrant value and its SuperKey (the XASH value under OR with all
    the other row XASHes)

    Args:
        path: A pathlib.Path object pointing to the tables position.
        load_opts: A dictionary of options, passed to pl.scan_csv or pl.scan_parquet
        clean_args: A dictionary of options for the cell cleaning function.
            See blend.utils.clean.
        xash_size: The size of the XASH value as bits (allowed values are 64, 128, 256, 512)
        max_cell_length: The size of the stored cell value as bytes (default 128).
            Only the first max_cell_length of each string will be stored.
            If it is negative, only the last max_cell_length will be stored.

    Returns:
        A tuple with the table ID (obtained from its path) and the parsed rows.

    Raises:
        ValueError: If the table is in an unknown format.
        pl.exceptions.NoDataError: If the table is empty.
    """
    if load_opts is None:
        load_opts = {}

    if clean_args is None:
        clean_args = {}

    table_id = path.stem

    try:
        table_df = load_table(path, load_opts, lazy=False)

        # BUG: here lazy-mode seems to be the worst choice:
        # filtering the all-nulls rows on the already collected
        # dataframe is much faster than doing this on a LazyFrame
        #
        # we need to keep track of the real row index of each record
        # even after dropping nulls, thus we create a new column to this aim
        table_df = (
            table_df.with_row_index(name="bl_row_index")
            .pipe(remove_null_rows, "bl_row_index")
            .pipe(remove_null_columns)
        )

        if table_df.shape[0] * table_df.shape[1] == 0:
            raise pl.exceptions.NoDataError("Empty table.")

    except (
        pl.exceptions.ComputeError,
        pl.exceptions.SchemaError,
        pl.exceptions.NoDataError,
        ValueError,
    ) as e:
        return table_id, f"{type(e)}::{str(e)}"

    # identify the numeric columns for the correlation part
    numeric_cols = set(table_df.select(cs.numeric()).columns)

    exprs = []
    for col_counter, col_name in enumerate(
        c for c in table_df.columns if c != "bl_row_index"
    ):
        is_numeric = col_name in numeric_cols
        if is_numeric:
            quadrant_expr = (
                pl.when(pl.col(col_name).is_not_null())
                .then(pl.col(col_name) >= pl.col(col_name).mean())
                .otherwise(None)
            )
        else:
            quadrant_expr = pl.lit(None, pl.Boolean)

        clean_expr = _clean(col_name, **clean_args)
        if isinstance(max_cell_length, int) and max_cell_length > 0:
            clean_expr = clean_expr.str.head(max_cell_length)
        elif isinstance(max_cell_length, int) and max_cell_length < 0:
            clean_expr = clean_expr.str.tail(max_cell_length)

        exprs.append(
            pl.struct(
                [
                    clean_expr.alias("bl_cell_value"),
                    quadrant_expr.alias("bl_quadrant"),
                    pl.lit(col_counter).alias("bl_column_id"),
                ]
            ).alias(col_name)
        )

    all_data = (
        table_df.lazy()
        .select(
            [
                pl.lit(table_id).alias("bl_table_id"),
                pl.col("bl_row_index").alias("bl_row_id"),
                *exprs,
            ]
        )
        # Unpivot the table to go from Wide to Long format
        .unpivot(
            index=["bl_table_id", "bl_row_id"],
            variable_name="original_col_name",
            value_name="packed_data",
        )
        # Expand the struct back into individual columns
        .unnest("packed_data")
        .filter(pl.col("bl_cell_value") != "")
        .select(
            "bl_table_id", "bl_column_id", "bl_row_id", "bl_quadrant", "bl_cell_value"
        )
        .collect()
    )

    if xash_size < 0:
        final_data = all_data.with_columns(pl.lit(int(0).to_bytes(), pl.Binary))
    else:
        superkey_data = all_data.group_by("bl_row_id").agg(
            pl.map_groups(
                ["bl_cell_value"],
                lambda values: calculate_superkey_for_row(
                    values[0].to_list(), xash_size
                ),
                return_dtype=pl.Binary,
                returns_scalar=True,
            ).alias("bl_super_key")
        )

        final_data = all_data.join(superkey_data, on="bl_row_id", coalesce=True)

    assert isinstance(final_data, pl.DataFrame)
    final_data = final_data.rename(lambda c: c.removeprefix("bl_"))

    return table_id, final_data


def calculate_superkey_for_row(cell_values: list, xash_size: int) -> bytes:
    superkey = 0
    for value in cell_values:
        if value is None:
            print(cell_values)
        superkey |= calculate_xash(value, xash_size)
    return superkey.to_bytes(16, byteorder="big")
