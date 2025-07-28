from abc import ABC, abstractmethod

# Typing imports
from blend.DBHandler import DBHandler
from typing import List


class Operator(ABC):
    DB: DBHandler | None = None  # DBHandler()

    def __init__(self, k: int, granularity: str):
        self.k = k
        self.granularity = granularity

    def run(self, additionals: str = "") -> List[int]:
        sql = self.create_sql_query(self.DB, additionals=additionals)
        result = self.DB.execute_and_fetchall(sql)
        return [r for r in result[: self.k]]

    @abstractmethod
    def cost(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def ml_cost(self, db: DBHandler) -> float:
        raise NotImplementedError

    @abstractmethod
    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        raise NotImplementedError
