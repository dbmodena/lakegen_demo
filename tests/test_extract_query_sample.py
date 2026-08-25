from pathlib import Path

import pytest

import extract_query_sample as sample


def _case(**updates):
    case = {
        "id": "q1",
        "question": "What is the total?",
        "reference_code": "result = Table_0['value'].sum()",
        "reference_result": 3,
        "expected_result_type": "number",
        "expected_result_description": "A total.",
        "relevant_table_ids": ["table"],
        "table_aliases": {"Table_0": "table"},
        "engine": "PANDAS",
        "query_kind": "single_table",
    }
    case.update(updates)
    return case


def _validate(monkeypatch, tmp_path, executions, **updates):
    iterator = iter(executions)
    monkeypatch.setattr(
        sample,
        "execute_pandas_reference",
        lambda **_kwargs: next(iterator),
    )
    return sample.validate_case(
        _case(**updates),
        tables_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        validation_runs=len(executions),
    )


def test_valid_reference_is_frozen_after_independent_runs(monkeypatch, tmp_path):
    validated, reasons = _validate(
        monkeypatch,
        tmp_path,
        [{"status": "success", "result": 3} for _ in range(3)],
    )

    assert reasons == []
    assert validated["reference_result"] == 3
    assert validated["gold_validation"]["deterministic"] is True
    assert validated["gold_validation"]["declared_result_match"] is True


def test_execution_error_rejects_reference(monkeypatch, tmp_path):
    validated, reasons = _validate(
        monkeypatch,
        tmp_path,
        [{"status": "invalid_reference", "error": "missing column"}],
    )

    assert validated is None
    assert reasons[0] == "reference_execution_error"


def test_declared_drift_is_recorded_but_only_strict_mode_rejects(
    monkeypatch, tmp_path
):
    validated, reasons = _validate(
        monkeypatch,
        tmp_path,
        [{"status": "success", "result": 4}],
    )
    assert reasons == []
    assert validated["gold_validation"]["declared_result_drift"] is True

    monkeypatch.setattr(
        sample,
        "execute_pandas_reference",
        lambda **_kwargs: {"status": "success", "result": 4},
    )
    strict, strict_reasons = sample.validate_case(
        _case(), tables_dir=tmp_path, cache_dir=tmp_path / "cache",
        validation_runs=1, require_declared_match=True,
    )
    assert strict is None
    assert strict_reasons == ["declared_result_drift"]


def test_non_deterministic_results_are_rejected(monkeypatch, tmp_path):
    validated, reasons = _validate(
        monkeypatch,
        tmp_path,
        [
            {"status": "success", "result": 3},
            {"status": "success", "result": 4},
        ],
    )

    assert validated is None
    assert reasons == ["non_deterministic_result"]


def test_equivalent_result_representation_is_accepted(monkeypatch, tmp_path):
    validated, reasons = _validate(
        monkeypatch,
        tmp_path,
        [
            {"status": "success", "result": 3},
            {"status": "success", "result": [{"total": 3}]},
        ],
    )

    assert reasons == []
    assert validated is not None


def test_missing_alias_and_table_are_rejected(tmp_path):
    case = _case(table_aliases={})
    validated, reasons = sample.validate_case(
        case, tables_dir=tmp_path, cache_dir=tmp_path / "cache",
        validation_runs=1,
    )
    assert validated is None
    assert "missing_table_aliases" in reasons

    case = _case(table_aliases={"Table_0": "missing"})
    validated, reasons = sample.validate_case(
        case, tables_dir=tmp_path, cache_dir=tmp_path / "cache",
        validation_runs=1,
    )
    assert validated is None
    assert reasons[0] == "reference_execution_error"


def test_limit_without_order_and_unseeded_sampling_are_rejected(tmp_path):
    assert "limit_without_order_by" in sample._static_reproducibility_reasons(
        "query = 'SELECT * FROM t LIMIT 5'"
    )
    validated, reasons = sample.validate_case(
        _case(reference_code="result = Table_0.sample(1)"),
        tables_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        validation_runs=1,
    )
    assert validated is None
    assert "non_deterministic_reference" in reasons


def test_sampling_more_than_valid_pool_fails():
    with pytest.raises(ValueError, match="only 1 eligible"):
        sample.sample_valid_cases(
            [_case()], count=2, seed=42, source="source.json"
        )
