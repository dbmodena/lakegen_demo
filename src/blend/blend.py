from numbers import Number
from pathlib import Path
from typing import Any, Callable, Optional

import polars as pl

from .db import DBHandler
from .operators import combiners, seekers
from .plan import Plan
from .utils import _truncate, clean

__all__ = ["BLEND"]


class BLEND:
    def __init__(
        self,
        db_path: Path,
        index_table: str = "all_tables",
        clean_function: Optional[Callable] = None,
        clean_args: Optional[dict] = None,
        xash_size: int = 128,
        max_cell_length: Optional[int] = 128,
        use_ml_optimizer: bool = False,
        freq_dict_path: Optional[Path] = None,
    ) -> None:
        """Instantiate a BLEND indexer and retriever.

        Args:
            db_path: A Path object leading to a duckdb file position.
            clean_function: The clean function for any cell value. It accepts any type and returns a string.
            clean_args: The clean function arguments, passed to it with any call.
            xash_size: The XASH size used for the super key.
            max_cell_length: The size of the stored cell value as bytes (default 128).
                Only the first max_cell_length of each string will be stored. If it is negative,
                only the last max_cell_length will be stored.
        """
        self._db_path = db_path
        self.db_handler: DBHandler = DBHandler(
            self._db_path, index_table, use_ml_optimizer, freq_dict_path
        )

        # Clean function and relative parameters
        self._clean_function = clean_function if clean_function else clean
        self._clean_args = clean_args if clean_args else {}

        self.xash_size = xash_size
        self.max_cell_length = max_cell_length

    def remove_table(self, table_id: str):
        self.db_handler.remove_table_from_index(table_id)

    def get_table(self, table_id: str) -> pl.DataFrame:
        return self.db_handler.get_table_from_index(table_id)

    def extract_token_frequencies_from_db(self) -> pl.DataFrame:
        return self.db_handler.extract_token_frequencies_from_db()

    def close(self):
        self.db_handler.close()

    def keyword_search(self, values: list[Any], k: int, clean: bool = True):
        """Execute a keyword search on the given query values.

        Args:
            values: A list of string keywords.
            k: The number of results to return.
            clean: If True, apply the default clean function on the input values.

        Returns:
            A list of tuples <table id, overlap size (distinct)>.
        """
        if clean:
            values = [self._clean_function(v, **self._clean_args) for v in values]
        values = [_truncate(v, self.max_cell_length) for v in values]
        plan = Plan(self.db_handler)
        plan.add("keyword", seekers.K(values, k))

        return plan.run()

    def single_column_join_search(
        self, column: list[Any], k: int, clean: bool = True
    ) -> list[tuple[str, int, int]]:
        """Execute a single-column join search on the given column values.

        Args:
            column: A list of strings representing the query column.
            k: The number of results to return.
            clean: If True, apply the default clean function on the input values.

        Returns:
            A list of tuples <table id, column number, overlap size (distinct)>.
        """
        if clean:
            column = [self._clean_function(cell, **self._clean_args) for cell in column]
        column = [_truncate(cell, self.max_cell_length) for cell in column]
        plan = Plan(self.db_handler)
        plan.add("single_column_join", seekers.SC(column, k))

        return plan.run()

    def multi_column_join_search(
        self,
        table: list[list[Any]] | pl.DataFrame,
        k: int,
        clean: bool = True,
        verbose: bool = False,
    ) -> list[tuple[str, list, float]]:
        """Execute a multi-column join search on the given table.

        This method is built on top of the MATE discovery algorithm.

        Args:
            table: A list-of-rows representing a table or a Polars DataFrame.
            k: The number of results to return.
            clean: If True, apply the default clean function on the input values.
            verbose: If True, print verbose output.

        Returns:
            A list of tuples <table id, column numbers, joinability score>.
        """
        if not isinstance(table, pl.DataFrame):
            table = pl.DataFrame(table, orient="row")

        if clean:
            table = table.with_columns(
                [
                    pl.col(col).map_elements(
                        lambda s: self._clean_function(s, **self._clean_args), pl.String
                    )
                    for col in table.columns
                ]
            )

        if isinstance(self.max_cell_length, int):
            if self.max_cell_length > 0:
                table = table.with_columns(
                    [
                        pl.col(col).str.head(self.max_cell_length)
                        for col in table.columns
                    ]
                )
            elif self.max_cell_length < 0:
                table = table.with_columns(
                    [
                        pl.col(col).str.tail(self.max_cell_length)
                        for col in table.columns
                    ]
                )

        plan = Plan(self.db_handler)
        plan.add("multi_column_join", seekers.MC(table, k, self.xash_size, verbose))
        return plan.run()

    def correlation_search(
        self,
        keys: list[Any],
        targets: list[Number],
        k: int = 10,
        hash_size: int = 256,
        clean: bool = True,
    ):
        """Execute a join-correlation search on the given key and target columns.

        This method is built on top of the QCR Join-Correlation search schema.

        Args:
            keys: A list of strings representing a key column.
            targets: A list of numbers representing a target column.
            k: The number of results to return.
            hash_size: The dimension of the hash size used by the QCR approach.
            clean: If True, apply the default clean function on the input values.
        """
        if clean:
            keys = [self._clean_function(k, **self._clean_args) for k in keys]
        keys = [_truncate(k, self.max_cell_length) for k in keys]

        plan = Plan(self.db_handler)
        plan.add("correlation", seekers.C(keys, targets, k, hash_size))

        return plan.run()

    def union_search(
        self, table: list[list[Any]] | pl.DataFrame, k: int, clean: bool = True
    ):
        """Execute a union search on the given table.

        This method exeutes a union of the results given by a single-column search
        on all the table columns.

        Args:
            table: A list-of-rows representing a table.
            k: The number of results to return.
            clean: If True, apply the default clean function on the input values.
        """
        if not isinstance(table, pl.DataFrame):
            table = pl.DataFrame(table, orient="row")

        if clean:
            table = table.with_columns(
                [
                    pl.col(col).map_elements(
                        lambda s: self._clean_function(s, **self._clean_args), pl.String
                    )
                    for col in table.columns
                ]
            )

        if isinstance(self.max_cell_length, int):
            if self.max_cell_length > 0:
                table = table.with_columns(
                    [
                        pl.col(col).str.head(self.max_cell_length)
                        for col in table.columns
                    ]
                )
            elif self.max_cell_length < 0:
                table = table.with_columns(
                    [
                        pl.col(col).str.tail(self.max_cell_length)
                        for col in table.columns
                    ]
                )

        plan = Plan(self.db_handler)
        for n_column, column in enumerate(table.columns):
            plan.add(str(n_column), seekers.SC(table.get_column(column), k * 10))

        plan.add(
            "union",
            combiners.Counter(k=k),
            inputs=list(map(str, range(len(table.columns)))),
        )

        return plan.run()
