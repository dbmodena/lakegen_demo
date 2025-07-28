# typing imports
from typing import Iterable

from blend.Operators.Seekers import SingleColumnOverlap
from blend.Plan import Plan


def SingleColumnJoinSearch(
    query_values: Iterable[any], k: int = 10, granularity: str = "base"
) -> Plan:
    plan = Plan()
    plan.add("single_column_overlap", SingleColumnOverlap(query_values, k, granularity))
    return plan
