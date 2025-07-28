from blend.Plan import Plan
from blend.Operators.Seekers import Keyword

# typing imports
from typing import List

def KeywordSearch(query_values: List[str], k: int = 10) -> Plan:
    plan = Plan()
    plan.add("keyword", Keyword(query_values, k))
    return plan
