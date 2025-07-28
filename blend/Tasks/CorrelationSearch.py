from numbers import Number

# typing imports
from typing import List, Literal

from blend.Operators.Seekers import Correlation
from blend.Plan import Plan


def CorrelationSearch(
    source_column: List[str],
    target_column: List[Number],
    k: int = 10,
    granularity: str = "base",
    hash_size: Literal[256, 512, 1024] = 256,
    verbosity: int = 1,
) -> Plan:
    """
    Args:
        source_column: The key values.
        target_column: The numeric values.
        k: Number of results to return.
        verbosity: an integer between 1 and 4.
            With 1, returns only the table IDs.
            With 2, also key and numeric column IDs.
            With 3 and 4, the score, as an absolute integer and as a [-1, 1] float, respectively.

    Returns:
        [TODO:description]
    """
    plan = Plan()
    plan.add(
        "correlation",
        Correlation(source_column, target_column, k, granularity, hash_size, verbosity),
    )
    return plan
