from typing import Optional
from abc import ABC, abstractmethod

from ..db import DBHandler


class Operator(ABC):
    DB: Optional[DBHandler] = None

    def __init__(self, k: int):
        self.k = k

    def run(self, additionals: str = "") -> list:
        if self.DB is None:
            raise TypeError(
                "DBHandler is None. Database hanlder has not been initialized yet."
            )

        sql = self.create_sql_query(self.DB, additionals=additionals)
        result = self.DB.execute_and_fetchall(sql)
        return [r for r in result[: self.k]]

    def set_db(self, db: DBHandler):
        self.DB = db

    @abstractmethod
    def cost(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def ml_cost(self, db: DBHandler) -> float:
        raise NotImplementedError

    @abstractmethod
    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        raise NotImplementedError
