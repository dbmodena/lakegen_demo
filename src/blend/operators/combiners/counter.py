from ...db import DBHandler
from .combiner_base import Combiner


class Counter(Combiner):
    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        sql = """
        SELECT table_id FROM (
        (
        """
        for i, input_ in enumerate(self._inputs):
            sql += input_.create_sql_query(db, additionals=additionals)
            sql += ")"
            if i < len(self._inputs) - 1:
                sql += " UNION ALL "
                sql += "("
        sql += f"""
        ) AS {db.random_subquery_name()}
        GROUP BY table_id
        ORDER BY COUNT(table_id) DESC
        LIMIT {self.k}
        """

        return sql
