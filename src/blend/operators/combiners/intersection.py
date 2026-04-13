from functools import cmp_to_key

from ...db import DBHandler
from .combiner_base import Combiner


class Intersection(Combiner):
    def cost(self) -> int:
        # Assuming most calculations are pruned by the first input
        return min(input_.cost() for input_ in self._inputs)

    def ml_cost(self, db: DBHandler) -> float:
        # Assuming most calculations are pruned by the first input
        return min(input_.ml_cost(db) for input_ in self._inputs)

    def create_sql_query(self, db: DBHandler, additionals: str = "") -> str:
        def lazy_comparator(operator1, operator2):
            if operator1.cost() != operator2.cost():
                # Rule based optimization
                return operator1.cost() - operator2.cost()
            else:
                # ML based optimization
                return operator1.ml_cost(db) - operator2.ml_cost(db)

        sorted_inputs = list(sorted(self._inputs, key=cmp_to_key(lazy_comparator)))

        intersect_additionals = ""
        for input_ in sorted_inputs[:-1]:
            result = input_.run(additionals + intersect_additionals)
            print(input, len(result))
            if len(result) == 0:
                return "SELECT table_id FROM all_tables WHERE 1=0"

            # for those cases were is not returned a simple list
            # of TableIDs (like MultiColumnSeeker) extract only them
            if isinstance(result[0], list) or isinstance(result[0], tuple):
                result = [r[0] for r in result]

            intersect_additionals = (
                # f" AND table_id IN ({db.create_sql_list_numeric(result)}) "
                f" AND table_id IN ({db.create_sql_list_str(result)}) "
            )

        sorted_inputs[-1].k = self.k
        return sorted_inputs[-1].create_sql_query(
            db, additionals=additionals + intersect_additionals
        )
