from abc import ABC
from typing import Optional

from ..operator_base import Operator


class Combiner(Operator, ABC):
    def __init__(self, k: int = 10) -> None:
        super().__init__(k)

        self._inputs: Optional[list[Operator]] = None

    def set_inputs(self, inputs: list[Operator]) -> None:
        if inputs is None or len(inputs) == 0:
            raise ValueError("Combiner must have at least one input.")
        if any(not isinstance(input_, Operator) for input_ in inputs):
            raise TypeError("All inputs must be Operators.")

        self._inputs: list[Operator] = inputs

    def cost(self) -> int:
        return sum(input_.cost() for input_ in self._inputs)

    def ml_cost(self, db) -> float:
        return sum(input_.ml_cost(db) for input_ in self._inputs)
