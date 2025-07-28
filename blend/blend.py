import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from time import time
from typing import Dict, List, Literal, Tuple

import duckdb
import polars as pl
import polars.selectors as cs
from duckdb import DuckDBPyConnection
from tqdm import tqdm

from .Plan import Plan
from .DBHandler import DBHandler
from .Operators.Seekers import SingleColumnOverlap, Keyword

from .utils import calculate_xash, clean

BASE_TEMPORAL_GRANULARITY = {
    "temporal": {
        "dtypes": [pl.Date, pl.Datetime],
        "levels": ["day", "month", "quarter", "year"],
    }
}


# To use ProcessPoolExecutor this cannot be put inside the class,
# because it cannot be pickled from there
def parse_table(
    table_path: str,
    granularities: Dict | None,
    format: Literal["csv", "parquet"],
    limit_table_rows: int | None,
):

    table_id = os.path.basename(table_path)

    try:
        match format:
            case "csv":
                table_df = pl.scan_csv(
                    table_path + '.csv', ignore_errors=True, n_rows=limit_table_rows
                )
            case "parquet":
                table_df = pl.scan_parquet(table_path + '.parquet', n_rows=limit_table_rows)
        table_df = (
            table_df.with_row_index(name="blend_row_index").drop_nulls().collect()
        )
    except (pl.exceptions.ComputeError, pl.exceptions.SchemaError) as e:
        print(f"Failure. Table: {table_path}. Error: {e}")
        return {}

    if table_df.shape[0] == 0 or table_df.shape[1] == 0:
        return {}
    
    data_dict = defaultdict(list)

    # identify the numeric columns
    # for the correlation part
    numeric_cols = [
        column_name
        for column_name in table_df.select(cs.numeric()).columns
        # if not any(
        #     nnct in column_name.lower()
        #     for nnct in non_numeric_column_tokens
        # )
    ]

    superkeys = defaultdict(lambda: defaultdict(int))

    # maybe using a different list for each attribute
    # (i.e. TableId, ...) in this phase, when loading
    # creating the dataframe and then loading it to duckdb we may see a speed up
    for col_counter, col_name in enumerate(table_df.columns[1:]):
        is_numeric_col = col_name in numeric_cols
        column = table_df.select("blend_row_index", col_name)
        if is_numeric_col:
            # QCR uses the mean or the median?
            mean = column.to_series(1).mean()

        # for each row, check if it's possible to cast it to
        # another granularity level, and save both
        # granularity class and level in the index
        g, gspec = None, None
        levels = ["base"]
        granularities = {} if not granularities else granularities

        # TODO fix for integrating blend_row_index
        for gclass, gspec in granularities.items():
            if column.dtype in gspec["dtypes"]:
                g = gclass
                levels = gspec["levels"]
                break

        for level in levels:
            # for each granularity level, cast the column
            if level != "base":
                column = table_df.to_series(col_counter)
                match g:
                    case "temporal":
                        match level:
                            case "day":
                                column = column.dt.strftime("%Y-%m-%d")
                            case "month":
                                column = column.dt.strftime("%Y-%m")
                            case "quarter":
                                column = column.dt.quarter().cast(str)
                            case "year":
                                column = column.dt.strftime("%Y")

            for row_counter, item in column.rows():
                tokenized = clean(str(item))

                # compute the quadrant (for the Correlation Seeker)
                quadrant = item >= mean if is_numeric_col and item else None

                data_dict["CellValue"].append(tokenized)
                data_dict["TableId"].append(table_id)
                data_dict["ColumnId"].append(col_counter)
                data_dict["RowId"].append(row_counter)
                data_dict["GranularityClass"].append(g)
                data_dict["Granularity"].append(level)
                data_dict["Quadrant"].append(quadrant)

                superkeys[g][row_counter] = superkeys[g][row_counter] | calculate_xash(
                    tokenized
                )

    # transform the superkey in binary
    superkeys_as_binary = {
        key: bytes(f"{superkey:0128b}".encode())
        for key, superkey in superkeys[g].items()
    }

    for rc in data_dict["RowId"]:
        data_dict["SuperKey"].append(superkeys_as_binary[rc])

    return data_dict


def drop_table(dbcon: DuckDBPyConnection):
    dbcon.sql("""
        DROP TABLE IF EXISTS AllTables CASCADE;
    """)


def create_table(dbcon: DuckDBPyConnection):
    # actually, DuckDB doesn't consider the n value in VARCHAR(n),
    # is provided just for compatibility
    dbcon.sql("""
        CREATE TABLE AllTables (
        CellValue           VARCHAR,
        TableId             VARCHAR, 
        ColumnId            UINTEGER,
        RowId               UINTEGER, 
        GranularityClass    VARCHAR,
        Granularity         VARCHAR,
        Quadrant            BOOLEAN,
        SuperKey            BYTEA,
        PRIMARY KEY (TableId, ColumnId, RowId, Granularity)
        );""")


def save_data_to_duckdb(dbcon: DuckDBPyConnection, data: Dict):
    schema = [
        ("CellValue", pl.String),
        ("TableId", pl.String),
        ("ColumnId", pl.UInt32),
        ("RowId", pl.UInt32),
        ("GranularityClass", pl.String),
        ("Granularity", pl.String),
        ("Quadrant", pl.Boolean),
        ("SuperKey", pl.Binary),
    ]

    # with pl.Series here, are we correctly
    # adopting lazy format?
    df = pl.LazyFrame(
        [pl.Series(c, data[c], d) for c, d in schema],
        schema=schema,
        # orient="row",
    )

    dbcon.sql("INSERT INTO AllTables SELECT * FROM df;")


def create_column_indexes(dbcon: DuckDBPyConnection):
    dbcon.sql("CREATE INDEX TableId_idx ON AllTables (TableId);")
    dbcon.sql("CREATE INDEX CellValue_idx ON AllTables (CellValue);")


class BLEND:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = ":memory:" if not db_path else db_path
        self._in_memory = not db_path
        self._db_handler: DBHandler | None = None

        if self._in_memory:
            raise NotImplementedError(
                "In-memory database still not correctly handled. Provide a DB path: db_path=..."
            )

        if os.path.exists(db_path):
            self._db_handler = DBHandler(db_path)

    def create_index(
        self,
        data_path: str,
        table_ids: List[str] | None = None,
        max_workers: int | None = None,
        batch_size: int = 1_000,
        granularities: Dict | None = None,
        format: Literal["csv", "parquet"] = "csv",
        limit_table_rows: int | None = None,
        verbose: bool = False,
    ) -> Tuple:
        """
        Create the BLEND index for the given tables.
        The index is named AllTables and is stored under the path passed in the init function.

        :param data_path: the directory where tables are stored.
        :param table_ids: a subset of table IDs (without any type extension) to use for the index creation. If None, all tables are used.
        :param max_workers: number of parallel workers to use during index creation. If None, only one process is used.
        :param batch_size: number of batch rows to commit during index creation.
        :param granularities: <do not use this param, under design>
        :param limit_table_rows: maximum number of rows per table. If None, entire table is loaded.
        :return: a tuple with insertion, indexes creation and total time.
        """

        dbcon: DuckDBPyConnection | None = None

        # take table IDs
        table_ids = list(
            filter(
                lambda _id: True if table_ids is None else _id in table_ids,
                map(
                    lambda _id: re.sub(r"(.csv|.parquet)$", "", _id),
                    os.listdir(data_path),
                ),
            )
        )

        # each entry of the dict is a values list
        # for one of the field in the BLEND index
        # in theory, Polars stores data in columnar format,
        # and passing to it data split by columns should
        # improve performance (not verified yet)
        data_dict = defaultdict(list)

        start_t = time()
        with ProcessPoolExecutor(max_workers) as executor:
            futures = {
                executor.submit(
                    parse_table,
                    os.path.join(data_path, table_path),
                    granularities,
                    format,
                    limit_table_rows,
                )
                for table_path in table_ids
            }

            # connect to the database once
            # the executor is launched
            dbcon = duckdb.connect(self._db_path)

            drop_table(dbcon)
            create_table(dbcon)

            for future in tqdm(
                futures, desc="Parsing and storing tables: ", disable=not verbose
            ):
                try:
                    new_data_dict = future.result()
                    if not new_data_dict:
                        continue

                    for k, v in new_data_dict.items():
                        data_dict[k].extend(v)

                    if len(data_dict["RowId"]) >= batch_size:
                        save_data_to_duckdb(dbcon, data_dict)
                        data_dict.clear()
                except TimeoutError:
                    continue

        if data_dict:
            save_data_to_duckdb(dbcon, data_dict)

        end_ins_t = time()

        if verbose:
            print("Tables ingestion completed.")
            print("Creating indexes...")
        create_column_indexes(dbcon)

        if not self._in_memory:
            dbcon.commit()
            dbcon.close()

        end_idx_t = time()

        self._db_handler = DBHandler(self._db_path)

        if verbose:
            print("Index creation completed.")
        return (end_ins_t - start_t, end_idx_t - end_ins_t, end_idx_t - start_t)

    def single_column_join_search(
        self, values: List[str], k: int, granularity: str = "base"
    ):
        plan = Plan(db=self._db_handler)
        plan.add("single_column_overlap", SingleColumnOverlap(values, k, granularity))
        return plan.run()
    
    def keyword_search(
        self, values: List[str], k: int
    ):
        plan = Plan(db=self._db_handler)
        plan.add("keyword", Keyword(values, k))
        return plan.run()


