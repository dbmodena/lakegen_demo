import random
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import duckdb
import polars as pl


class DBHandler(object):
    """Handles interactions with the DuckDB database for the BLEND index.

    Attributes:
        connection: The DuckDB connection object.
        cursor: The DuckDB cursor object.
        dbms: The database management system being used (default: "duckdb").
        use_pandas: Whether to return results as pandas DataFrames (default: True).
        db_path: The path to the DuckDB database file.
        index_table: The name of the table used for indexing.
        db_name: The name of the database.
        use_ml_optimizer: Whether to use the ML optimizer.
        frequency_dict: A dictionary mapping tokens to their frequencies.
    """

    def __init__(
        self,
        db_path: Path,
        index_table: Optional[str] = None,
        use_ml_optimizer: bool = False,
        freq_dict_path: Optional[Path] = None,
    ) -> None:
        """Initializes the DBHandler.

        Args:
            db_path: The path to the DuckDB database file.
            index_table: The name of the table used for indexing. Defaults to "all_tables".
            use_ml_optimizer: Whether to use the ML optimizer. Defaults to False.
            freq_dict_path: Path to the frequency dictionary CSV file. Required if use_ml_optimizer is True.

        Raises:
            FileNotFoundError: If the parent directory of db_path does not exist.
            AssertionError: If use_ml_optimizer is True but freq_dict_path is not provided.
        """
        self.connection = None
        self.cursor = None
        self.dbms = "duckdb"  # we'll use only duckdb

        self.db_path = db_path
        if not self.db_path.parent.exists():
            raise FileNotFoundError(
                f"DB directory doesn't exist: {self.db_path.parent}"
            )

        self.index_table = index_table if index_table is not None else "all_tables"
        self.db_name = self.db_path.stem.replace("-", "_")

        self.use_ml_optimizer = use_ml_optimizer

        # BLEND supports also the possibility to
        # optimize the general plan, but a frequency
        # dictionary is needed
        if self.use_ml_optimizer:
            assert freq_dict_path, (
                "Frequencies file must be provided to use ML optimizer"
            )

            df = pl.read_csv(
                freq_dict_path, schema={"tokenized": pl.String, "frequency": pl.Int64}
            )
            self.frequency_dict = dict(zip(df["tokenized"], df["frequency"]))
        else:
            self.frequency_dict = {}

    def drop_index_table(self):
        """Drops the index table if it exists."""
        with duckdb.connect(self.db_path) as con:
            con.sql(f"""
                DROP TABLE IF EXISTS {self.index_table} CASCADE;
                CHECKPOINT {self.db_name};
            """)

    def create_index_table(self):
        """Creates the index table."""
        with duckdb.connect(self.db_path) as con:
            con.sql(f"""
                CREATE TABLE {self.index_table} (
                table_id             VARCHAR,
                column_id            UINTEGER,
                row_id               UINTEGER,
                quadrant             BOOLEAN,
                cell_value           VARCHAR,
                super_key            BYTEA,
                PRIMARY KEY (table_id, row_id, column_id)
            );""")

    def create_column_indexes(self):
        """Creates indexes on the cell_value column."""
        with duckdb.connect(self.db_path) as con:
            # con.sql(f"CREATE INDEX table_id_idx ON {self.index_table} (table_id);")
            con.sql(f"CREATE INDEX cell_value_idx ON {self.index_table} (cell_value);")

    def save_data_to_duckdb(self, data: pl.DataFrame | list[pl.DataFrame] | Path):
        """Saves data to the DuckDB database.

        Args:
            data: A polars DataFrame, a list of polars DataFrames, or a Path to a parquet file.
        """
        if isinstance(data, pl.DataFrame):
            data = [data]

        with duckdb.connect(self.db_path) as con:
            if isinstance(data, list):
                for df in data:
                    con.sql(f"INSERT INTO {self.index_table} SELECT * FROM df;")
            elif isinstance(data, Path):
                filename = data.absolute().as_posix()
                con.sql(
                    f"INSERT INTO {self.index_table} SELECT * FROM read_parquet('{filename}');"
                )

    def close(self) -> None:
        """Closes the database connection (placeholder)."""
        pass

    def clean_query(self, query: str) -> str:
        """Replaces the 'all_tables' index name with the actual index table name.

        Args:
            query: The SQL query string.

        Returns:
            The modified query string.
        """
        return query.replace("all_tables", f"{self.index_table}")

    def execute_and_fetchall(self, query: str) -> list[Union[tuple, list]]:
        """Executes a query and returns all results.

        Args:
            query: The SQL query string.

        Returns:
            A list of tuples or lists containing the query results.
        """
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(super_key)", "super_key")

        results = []
        connection = duckdb.connect(self.db_path, read_only=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()  # ty: ignore
        finally:
            connection.close()
        return results  # ty: ignore

    def execute_and_fetchyield(self, query: str, params: Optional[tuple] = None):
        """Executes a query and yields results in batches.

        Args:
            query: The SQL query string.
            params: Optional parameters for the query.

        Yields:
            Rows from the query result.
        """
        query = self.clean_query(query)
        query = query.replace("TO_BITSTRING(super_key)", "super_key")

        conn = duckdb.connect(self.db_path, read_only=True)

        try:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                while rows := cursor.fetchmany(size=500):
                    for row in rows:
                        yield row
        finally:
            conn.close()

    def get_table_from_index(self, table_id: str) -> pl.DataFrame:
        sql = f"""
        SELECT cell_value, column_id, row_id
        FROM all_tables
        WHERE table_id = {table_id}
        """

        results = self.execute_and_fetchall(sql)

        df = pl.DataFrame(
            results, schema={"cell_value": str, "column_id": int, "row_id": int}
        )

        df = df.unique()
        df = df.pivot(index="row_id", on="column_id", values="cell_value")

        return df

    def extract_token_frequencies_from_db(self) -> pl.DataFrame:
        sql = """
            SELECT cell_value AS tokenized, COUNT(cell_value) AS frequency
            FROM all_tables
            GROUP BY cell_value
            ORDER BY cell_value
        """

        results = self.execute_and_fetchall(sql)
        return pl.DataFrame(
            results, schema=[("tokenized", str), ("frequency", int)], orient="row"
        )

    def table_ids_to_sql(self, table_ids: list[int]) -> str:
        if len(table_ids) == 0:
            return "SELECT 0 AS table_id WHERE 1 = 0"

        if self.dbms == "postgres":
            return f"""
            SELECT * FROM (
                VALUES {" ,".join([f"('{table_id}')" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}(table_id)
            """

        return f"""
            SELECT table_id FROM (
            {" UNION ALL ".join([f"SELECT '{table_id}' AS table_id" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}
        """

    def get_token_frequencies(self, tokens: Iterable[str]) -> dict[str, int]:
        tokens = self.clean_value_collection(set(tokens))

        return {token: self.frequency_dict.get(token, 1) for token in tokens}

    def remove_table_from_index(self, table_id: str):
        sql = f"DELETE FROM all_tables WHERE table_id = '{table_id}'"

        self.execute_and_fetchall(sql)

    @staticmethod
    def clean_value_collection(values: Iterable[Any]) -> list[str]:
        return [
            str(v).replace("'", "''").strip() for v in values if str(v).lower() != "nan"
        ]

    @staticmethod
    def create_sql_list_str(values: Iterable[Any]) -> str:
        values = set(map(lambda x: str(x).replace("'", ""), values))
        return "'{}'".format("' , '".join(values))

    @staticmethod
    def create_sql_list_numeric(values: Iterable[Number]) -> str:
        return "{}".format(" , ".join(map(str, values)))

    @staticmethod
    def random_subquery_name() -> str:
        return f"subquery{random.random() * 1000000:.0f}"
