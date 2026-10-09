import pandas as pd

from lakegen.query_validator import _looks_empty_or_zero, validate_query


def test_looks_empty_or_zero_handles_the_actual_answer_wrapper_shape():
    """[{'total_amount_paid': 0.0}] is the real __LAKEGEN_EVAL_JSON__ final-
    answer shape this pipeline produces -- confirmed missing before this
    fix, since bare list handling fell through to `return False`."""
    assert _looks_empty_or_zero([{"total_amount_paid": 0.0}]) is True
    assert _looks_empty_or_zero([{"total_amount_paid": 585559.83}]) is False
    assert _looks_empty_or_zero([]) is True
    assert _looks_empty_or_zero(None) is True
    assert _looks_empty_or_zero(0.0) is True
    assert _looks_empty_or_zero(42) is False


def test_composed_check_flags_the_malformed_header_bug():
    df = pd.DataFrame({
        "_duplicated_2": ["Dragonfly House Rent and SC"] * 5,
        "585,559.83": ["10142.26"] * 5,
    })
    code = (
        "x = pd.to_datetime(df['_duplicated_2'], format='%d/%m/%Y', errors='coerce')\n"
        "mask = x.dt.year == 2018"
    )
    outcome = validate_query(code, df, [{"total_amount_paid": 0.0}])
    assert outcome.flagged
    assert "0.0" not in outcome.flag_reason or "empty/zero" in outcome.flag_reason


def test_composed_check_does_not_flag_a_legitimate_nonzero_result():
    df = pd.DataFrame({"amount": [100.0, 200.0]})
    code = "mask = df['amount'] > 50"
    outcome = validate_query(code, df, [{"total": 300.0}])
    assert not outcome.flagged


def test_composed_check_does_not_flag_a_genuine_zero_without_grounding_issue():
    """A real zero (no filter-literal problem detected) must NOT be flagged
    -- the whole point of requiring BOTH conditions together."""
    df = pd.DataFrame({"amount": [100.0, 200.0]})
    code = "mask = df['amount'] > 999999"  # legitimately finds nothing, SOFT not HIGH
    outcome = validate_query(code, df, [{"total": 0.0}])
    assert not outcome.flagged
