import pytest

import build_benchmark as benchmark


def _payload(
    *, easy_single=0, easy_multi=0, medium_single=0, medium_multi=0,
    hard_single=0, hard_multi=0, engine="PANDAS",
):
    counts = {
        ("easy", "single"): easy_single, ("easy", "multi"): easy_multi,
        ("medium", "single"): medium_single, ("medium", "multi"): medium_multi,
        ("hard", "single"): hard_single, ("hard", "multi"): hard_multi,
    }
    group = {"_meta": {"tables": {"Table_0": "table_0", "Table_1": "table_1"}}}
    index = 0
    for (difficulty, scope), amount in counts.items():
        for _ in range(amount):
            aliases = ["Table_0"] if scope == "single" else ["Table_0", "Table_1"]
            group[str(index)] = {
                "client_id": f"q-{index}", "status": "success", "question": f"Question {index}",
                "difficulty": difficulty, "tables": aliases, "code": "result = 1",
                "question_keywords": ["question"], "query_result": 1,
                "expected_result_type": "number",
            }
            index += 1
    return {engine: {"generated": {"group": group}}}


def test_builds_exact_difficulty_and_one_third_multi_table_sample():
    payload = _payload(
        easy_single=40, easy_multi=20, medium_single=40,
        medium_multi=20, hard_single=40, hard_multi=20,
    )
    result = benchmark.build_benchmark(payload, count=100, seed=7, source="uk.json")

    cases = result["cases"]
    assert len(cases) == 100
    assert {case["reference_code"] for case in cases} == {"result = 1"}
    assert result["sample_metadata"]["difficulty_quotas"] == {
        "easy": 34, "medium": 33, "hard": 33,
    }
    assert sum(
        result["sample_metadata"]["strata"][difficulty]["multi_table"]
        for difficulty in ("easy", "medium", "hard")
    ) == 34
    assert not {"tables", "engine", "query_kind", "source_group", "difficulty", "table_scope"} & cases[0].keys()
    assert result["sample_metadata"]["difficulty_quotas"] == {"easy": 34, "medium": 33, "hard": 33}


def test_excludes_non_successful_records_and_requires_generated_code():
    payload = _payload(easy_single=1, easy_multi=1, medium_single=1, medium_multi=1, hard_single=1, hard_multi=1)
    record = payload["PANDAS"]["generated"]["group"]["0"]
    record["status"] = "failed"
    del payload["PANDAS"]["generated"]["group"]["1"]["code"]

    with pytest.raises(ValueError, match="Cannot satisfy"):
        benchmark.build_benchmark(payload, count=6)


def test_accepts_successful_records_from_an_engine_other_than_pandas():
    payload = _payload(
        easy_single=1, easy_multi=1, medium_single=1,
        medium_multi=1, hard_single=1, hard_multi=1, engine="SQL",
    )

    result = benchmark.build_benchmark(payload, count=3)

    assert len(result["cases"]) == 3
    assert "engine_filter" not in result["sample_metadata"]
    assert all("engine" not in case for case in result["cases"])


def test_uses_exact_one_third_multi_table_split():
    payload = _payload(
        easy_single=40, easy_multi=20, medium_single=40,
        medium_multi=10, hard_single=40, hard_multi=19,
    )

    result = benchmark.build_benchmark(payload, count=100)

    assert sum(result["sample_metadata"]["strata"][difficulty]["multi_table"] for difficulty in ("easy", "medium", "hard")) == 34
    assert result["sample_metadata"]["table_scope_requirements"] == {
        "multi_table": 34,
        "single_table": 66,
    }


def test_exposes_dataset_specific_default_paths():
    assert benchmark.DEFAULT_DATASET == "nyc"
    assert benchmark.DEFAULT_PATHS["uk"][1] == benchmark.Path("benchmark/100q_uk.json")
    assert benchmark.DEFAULT_PATHS["nyc"][1] == benchmark.Path("benchmark/100q_nyc.json")


def test_preserves_accepted_table_alternatives():
    payload = _payload(easy_single=1)
    record = payload["PANDAS"]["generated"]["group"]["0"]
    record["retrieval"] = {
        "contract_version": 1,
        "tables": {
            "table_0": {"accepted_table_ids": ["table_alt", "table_alt"]},
        },
    }

    case = benchmark._normalize(benchmark._records(payload)[0])

    assert case["accepted_table_alternatives"] == {
        "table_0": ["table_alt"],
    }


def test_omits_empty_accepted_table_alternatives_from_benchmark():
    payload = _payload(easy_multi=1)
    record = payload["PANDAS"]["generated"]["group"]["0"]
    record["retrieval"] = {
        "contract_version": 1,
        "tables": {
            "table_0": {"accepted_table_ids": []},
            "table_1": {"accepted_table_ids": []},
        },
    }

    case = benchmark.build_benchmark(payload, count=1)["cases"][0]
    normalized = benchmark._normalize(benchmark._records(payload)[0])

    assert "accepted_table_alternatives" not in case
    assert normalized["accepted_table_alternatives"] is None
