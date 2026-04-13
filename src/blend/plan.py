from typing import Iterable, Optional

from .db import DBHandler
from .operators import Combiner, Operator, Seeker


class Plan(object):
    def __init__(self, db: DBHandler) -> None:
        self.DB = db
        self._operators = dict()
        self._terminal_candidates: set[str] = set()

    def add(
        self, name: str, operator: Operator, inputs: Optional[Iterable[str]] = None
    ) -> None:
        """Add an operator to the plan.

        Args:
            name: The name of the operator.
            operator: The operator instance to add.
            inputs: A list of names of input operators for the current operator.

        Raises:
            ValueError: If the operator name already exists or if a Seeker has inputs.
            TypeError: If the operator is not an instance of Operator.
            KeyError: If an input operator is not found in the plan.
        """
        if inputs is None:
            inputs = []
        inputs = list(inputs)

        if name in self._operators:
            raise ValueError(f"Operator {name} already exists in the plan.")

        if not isinstance(operator, Operator):
            raise TypeError(f"Expected Operator, got {type(operator)}")

        if isinstance(operator, Seeker):
            if len(inputs) != 0:
                raise ValueError(f"Seeker {name} cannot have inputs.")

        if isinstance(operator, Combiner):
            operator: Combiner = operator
            try:
                input_operators = [self._operators[input_name] for input_name in inputs]
            except KeyError as e:
                raise KeyError(
                    f"Operator {name} has an invalid input: {e}. "
                    f"Please add the input operator before adding the combiner."
                )

            operator.set_inputs(list(input_operators))

        # Add the operator to the plan
        operator.set_db(self.DB)
        self._terminal_candidates.add(name)
        self._operators[name] = operator
        self._terminal_candidates -= set(inputs)

    def run(self) -> list:
        """Run the plan.

        Returns:
            The results of the plan execution.

        Raises:
            ValueError: If no terminal candidates or multiple terminal candidates are found.
        """
        if len(self._terminal_candidates) == 0:
            raise ValueError("No terminal candidates found.")

        if len(self._terminal_candidates) > 1:
            raise ValueError("Multiple terminal candidates found.")

        terminal = next(iter(self._terminal_candidates))

        return self._operators[terminal].run()
