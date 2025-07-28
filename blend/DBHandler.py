import random
from numbers import Number

from typing import Iterable, List, Tuple, Union

import duckdb
import pandas as pd


# TODO handle config file
class DBHandler(object):
    def __init__(
        self,
        db_path: str | None = None,
        index_table: str = "AllTables",
        use_ml_optimizer: bool = False,
        freq_dict_path: str | None = None,
    ) -> None:
        self.connection = None
        self.cursor = None
        self.index_table = None
        self.dbms = "duckdb"  # I'll use only duckdb

        self.db_path = ":memory:" if not db_path else db_path
        self.in_memory = self.db_path == ":memory:"

        if self.in_memory:
            # need to handle read-only and other things
            raise NotImplementedError("In-memory database not implemented yet.")

        self.index_table = index_table
        self.use_ml_optimizer = use_ml_optimizer

        # self.connection = duckdb.connect(self.db_path, read_only=not self.in_memory)
        # self.cursor = self.connection.cursor()

        if self.use_ml_optimizer:
            assert freq_dict_path, (
                "Frequencies file must be provided to use ML optimizer"
            )

            df = pd.read_csv(freq_dict_path)
            self.frequency_dict = dict(zip(df["tokenized"], df["frequency"]))
        else:
            self.frequency_dict = {}

    def close(self) -> None:
        # if self.cursor is not None:
        #     self.cursor.close()
        # if self.connection is not None:
        #     self.connection.close()
        # self.cursor = None
        # self.connection = None
        pass

    def clean_query(self, query: str) -> str:
        return query.replace("AllTables", f"{self.index_table}")

    def execute_and_fetchall(self, query: str) -> List[Union[Tuple, List]]:
        """Returns results"""
        query = self.clean_query(query)
        # if self.dbms == "postgres":
        #     query = query.replace("TO_BITSTRING(superkey)", "superkey")

        query.replace("TO_BITSTRING(superkey)", "superkey")

        # do not keep the cursor
        # self.cursor.execute(query)
        # results = self.cursor.fetchall()

        with duckdb.connect(self.db_path, read_only=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()

        return results

    def get_table_from_index(self, table_id: int) -> pd.DataFrame:
        sql = f"""
        SELECT CellValue, ColumnId, RowId
        FROM AllTables
        WHERE TableId = {table_id}
        """

        results = self.execute_and_fetchall(sql)

        df = pd.DataFrame(
            results, columns=["CellValue", "ColumnId", "RowId"], dtype=str
        )
        df = df.drop_duplicates()
        df = df.pivot(index="RowId", columns="ColumnId", values="CellValue")
        df.index.name = None
        df.columns.name = None

        return df

    def table_ids_to_sql(self, table_ids: Iterable[int]) -> str:
        if len(table_ids) == 0:
            return "SELECT 0 AS TableId WHERE 1 = 0"

        if self.dbms == "postgres":
            return f"""
            SELECT * FROM (
                VALUES {" ,".join([f"({table_id})" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}(TableId)
            """
        elif self.dbms == "vertica":
            return f"""
            SELECT TableId
            FROM (
                SELECT Explode(Array[{", ".join(f"{table_id}" for table_id in table_ids)}])
                OVER (Partition Best) AS (Index_In_Array, TableId)
            ) {DBHandler.random_subquery_name()}
            """

        return f"""
            SELECT TableId FROM (
            {" UNION ALL ".join([f"SELECT {table_id} AS TableId" for table_id in table_ids])}
            ) AS {DBHandler.random_subquery_name()}
        """

    def get_token_frequencies(self, tokens: Iterable[str]) -> dict[str, int]:
        tokens = DBHandler.clean_value_collection(set(tokens))

        return {token: DBHandler.frequency_dict.get(token, 1) for token in tokens}

    @staticmethod
    def clean_value_collection(values: Iterable[any]) -> List[str]:
        return [
            str(v).replace("'", "''").strip() for v in values if str(v).lower() != "nan"
        ]

    @staticmethod
    def create_sql_list_str(values: Iterable[any]) -> str:
        values = [str(x).replace("'", "") for x in values]
        return "'{}'".format("' , '".join(set(values)))

    @staticmethod
    def create_sql_list_numeric(values: Iterable[Number]) -> str:
        values = [str(x) for x in values]
        return "{}".format(" , ".join(values))

    @staticmethod
    def random_subquery_name() -> str:
        return f"subquery{random.random() * 1000000:.0f}"
