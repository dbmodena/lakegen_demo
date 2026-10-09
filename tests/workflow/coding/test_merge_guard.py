"""The sandbox's merge guard and the join/union check apply one rule to the size of a merge."""
import re

import pytest

from lakegen.domain.merge_limits import MERGE_BLOWUP_FACTOR, MERGE_BLOWUP_MIN_ROWS, merge_blows_up
from lakegen.workflow.coding.agent import _CARTESIAN_GUARD_PREAMBLE


def _guard():
    """The guard's size check, run on its own (the preamble also patches pandas, which a test must not)."""
    source = re.search(r"def _lakegen_check_merge_blowup.*?\n\n\n", _CARTESIAN_GUARD_PREAMBLE, re.S).group(0)
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - the guard's own source, bounded by the regex above
    return namespace["_lakegen_check_merge_blowup"]


CASES = [
    (100, 100, 100),
    (1_000, 1_000, 40_000),                                                  # 40x but small
    (1_000, 1_000, MERGE_BLOWUP_MIN_ROWS),                                   # at the row floor
    (1_000, 1_000, MERGE_BLOWUP_MIN_ROWS + 1),                               # past it, 50x
    (10_000, 2_000, 10_000 * MERGE_BLOWUP_FACTOR),                           # exactly the factor
    (10_000, 2_000, 10_000 * MERGE_BLOWUP_FACTOR + 1),                       # one row past it
    (2_000, 10_000, 10_000 * MERGE_BLOWUP_FACTOR + 1),                       # larger side on the right
    (0, 0, 0),
    (1, 1, 51_000),
]


@pytest.mark.parametrize(("left", "right", "result"), CASES)
def test_the_guard_stops_exactly_the_merges_the_shared_rule_calls_blow_ups(left, right, result):
    guard = _guard()
    try:
        guard(left, right, result, "k", "pd.merge")
        stopped = False
    except RuntimeError as error:
        stopped = "CARTESIAN_PRODUCT_DETECTED" in str(error)
    assert stopped == merge_blows_up(result, left, right)
