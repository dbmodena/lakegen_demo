"""Composed query validator -- ties together the value-grounding check and
the date-parse sanity check, plus one rule that only makes sense combined: a
legitimately-typed but suspiciously empty/zero result COMBINED WITH an
unresolved grounding warning escalates to a flagged outcome. A bare
empty-result check alone is too blunt -- a real zero is sometimes correct;
it's the COMBINATION with a grounding violation that makes the "0.0 by
coincidence" shape (the bug that motivated this whole effort) reliably
detectable.

Validated over four rounds of live 100-question scratchpad testing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from lakegen.value_grounding import (
    GroundingViolation, check_case_sensitivity_gaps, check_date_parse_sanity, check_value_grounding,
    extract_literal_comparisons,
)


@dataclass
class ValidationOutcome:
    grounding_violations: list[GroundingViolation] = field(default_factory=list)
    empty_or_zero_result: bool = False
    flagged: bool = False  # the composed rule: empty/zero result + HIGH grounding violation
    flag_reason: str = ""


def _looks_empty_or_zero(result) -> bool:
    if result is None:
        return True
    if isinstance(result, (int, float)) and float(result) == 0.0:
        return True
    if isinstance(result, pd.DataFrame) and result.empty:
        return True
    if isinstance(result, dict):
        return all(_looks_empty_or_zero(v) for v in result.values()) if result else True
    if isinstance(result, list):
        # [{'total_amount_paid': 0.0}] -- the common __LAKEGEN_EVAL_JSON__
        # "final answer" wrapper shape this pipeline actually produces;
        # recurse into it the same way as a bare dict rather than always
        # returning False for any list (a real bug caught during
        # validation: this shape was not being flagged at all before).
        return all(_looks_empty_or_zero(v) for v in result) if result else True
    return False


# Public: the reviewed coder also needs this to tell a compliant-but-empty result
# (a plan problem) from broken code.
looks_empty_or_zero = _looks_empty_or_zero


def validate_query(code: str, df: pd.DataFrame, result=None) -> ValidationOutcome:
    comparisons = extract_literal_comparisons(code)
    violations = check_value_grounding(comparisons, df)
    violations += check_date_parse_sanity(code, df)
    violations += check_case_sensitivity_gaps(code, df)
    outcome = ValidationOutcome(grounding_violations=violations)
    outcome.empty_or_zero_result = _looks_empty_or_zero(result)

    high_violations = [v for v in violations if v.confidence == "HIGH"]
    if outcome.empty_or_zero_result and high_violations:
        outcome.flagged = True
        outcome.flag_reason = (
            "Result is empty/zero AND a HIGH-confidence value-grounding violation was found "
            f"({high_violations[0].message}) -- this result is likely wrong, not genuinely zero."
        )
    return outcome
