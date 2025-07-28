from typing import Iterable

# Typing imports
from blend.DBHandler import DBHandler
from blend.Operators.Seekers.SeekerBase import Seeker


class SingleColumnOverlap(Seeker):
    def __init__(
        self, input_query_values: Iterable[str], k: int = 10, granularity: str = "base"
    ) -> None:
        super().__init__(k, granularity)

        self.input = set(input_query_values)
        self.base_sql = """
        SELECT TableId, ColumnId, COUNT(DISTINCT CellValue) FROM AllTables
        WHERE CellValue IN ($TOKENS$) $ADDITIONALS$
        GROUP BY TableId, ColumnId
        ORDER BY COUNT(DISTINCT CellValue) DESC
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
