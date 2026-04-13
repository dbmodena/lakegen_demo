from typing import Iterable

from ...db import DBHandler
from .seeker_base import Seeker


class SingleColumnOverlap(Seeker):
    def __init__(self, input_query_values: Iterable[str], k: int = 10) -> None:
        super().__init__(k)

        self.input = set(input_query_values)

        # The final output results list is sorted by
        # the COUNT(DISTINCT), applying a set semantic
        self.base_sql = """
        SELECT table_id, column_id, 
            COUNT(DISTINCT cell_value) AS TokensCount, 
            FROM all_tables
        WHERE cell_value IN ($TOKENS$) $ADDITIONALS$
        GROUP BY table_id, column_id
        ORDER BY COUNT(DISTINCT cell_value) DESC
        LIMIT $TOPK$
        """

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        sql = self.base_sql.replace("$TOPK$", str(self.k))
        sql = sql.replace("$ADDITIONALS$", additionals)
        sql = sql.replace(
            "$TOKENS$", db.create_sql_list_str(db.clean_value_collection(self.input))
        )

        return sql

    def cost(self) -> int:
        return 4

    def ml_cost(self, db: DBHandler) -> float:
        return self._predict_runtime([list(self.input)], db)
