"""Pipeline goldens of the autonomous service, ``lakegen.service.run_question``.

Every scenario runs the service with scripted phases and snapshots the call
log, the QueryResult, the log rows and the manifest. The scenarios cover each
discovery variant (unified/divided x agentic/orchestrated, divided with Jev)
and each way a run can go: first-try success, a coder rejection that sends
discovery back, the last-attempt fallback, the automatic coder sweep, a
synthesized refusal, a semantic-judge adjudication, a selection-contract
block, execution errors and empty results.
"""

from __future__ import annotations

from typing import Callable

import pytest

from lakegen.phases.jev_selector import TRACE_HEADER as JEV_TRACE_HEADER

from golden.harness import (
    SERVICE_NAMES, Case, Script, discovery_result, experiment, normalise,
    phase1_result, phase3_result, prepare_run, prepared, make_runtime,
)

SCENARIOS: dict[str, Callable[[], Case]] = {}


def scenario(name: str):
    def register(build: Callable[[], Case]) -> Callable[[], Case]:
        SCENARIOS[name] = build
        return build
    return register


SELECTED = (["gold.csv"], ["parks"], {"gold.csv": {"title": "gold"}}, "The table covers the question.", "trace", 5)
ANSWER = ("There are 42 parks.", 3)
NUMBER_CONTEXT = {"SOURCE_EXPECTED_RESULT_TYPE": "number", "SOURCE_REFERENCE_RESULT": [{"total": 42}]}


@scenario("unified_agentic_success")
def _():
    return Case(experiment(), Script(phase12=[SELECTED], phase3=[phase3_result()], phase4=[ANSWER]))


@scenario("divided_agentic_success")
def _():
    return Case(
        experiment(discovery_architecture="divided"),
        Script(
            phase1=[phase1_result()],
            phase2=[(["gold.csv"], ["gold.csv", "other.csv"], {"gold.csv": {"title": "gold"}}, "The selection covers the question.", "trace", 4)],
            phase3=[phase3_result()], phase4=[ANSWER],
        ),
        files=("gold.csv", "other.csv"),
    )


@scenario("unified_orchestrated_success")
def _():
    return Case(
        experiment(tool_access="orchestrated_context"),
        Script(unified_orchestrated=[discovery_result(["gold.csv"])], phase3=[phase3_result()], phase4=[ANSWER]),
    )


@scenario("divided_orchestrated_success")
def _():
    return Case(
        experiment(discovery_architecture="divided", tool_access="orchestrated_context"),
        Script(
            phase1=[phase1_result()],
            prepare_context=[(prepared(["gold.csv", "other.csv"]), {"gold.csv": {"title": "gold"}})],
            select_prepared=[(["gold.csv"], "Selected.", "", 3)],
            phase3=[phase3_result()], phase4=[ANSWER],
        ),
        files=("gold.csv", "other.csv"),
    )


@scenario("divided_jev_success")
def _():
    return Case(
        experiment(discovery_architecture="divided", discovery={"selector": "jev"}),
        Script(
            phase1=[phase1_result()],
            phase2=[(["gold.csv"], ["gold.csv"], {}, "Jev chose gold.csv.", JEV_TRACE_HEADER + "\ngold.csv 0.91", 0)],
            phase3=[phase3_result()], phase4=[ANSWER],
        ),
    )


@scenario("unified_coder_rejection_rediscovery")
def _():
    return Case(
        experiment(),
        Script(
            phase12=[
                (["bad.csv", "good.csv"], ["parks"], {}, "Both tables look relevant.", "trace", 5),
                (["good.csv"], ["parks", "2024"], {}, "good.csv alone covers 2024.", "trace", 5),
            ],
            phase3=[
                phase3_result(
                    rejected_reason="bad.csv has no 2024 rows.", rejection_details={"category": "temporal_coverage"},
                    raw_result=None, structured_result=None,
                    rejection_keep_tables=["good.csv"], rejection_skip_tables=["bad.csv"],
                ),
                phase3_result(),
            ],
            phase4=[ANSWER],
        ),
        files=("bad.csv", "good.csv"),
    )


@scenario("divided_coder_rejection_rediscovery")
def _():
    def phase2(*, carried_tables, **_kwargs):
        selected = list(carried_tables) or ["good.csv", "bad.csv"]
        return (selected, selected, {}, "The selection covers the question.", "trace", 4)

    return Case(
        experiment(discovery_architecture="divided"),
        Script(
            phase1=[phase1_result()],
            phase2=[phase2],
            phase3=[
                phase3_result(
                    rejected_reason="good.csv covers part of the answer.", raw_result=None, structured_result=None,
                    rejection_keep_tables=["good.csv"], rejection_skip_tables=["bad.csv"],
                ),
                phase3_result(),
            ],
            phase4=[ANSWER],
        ),
        files=("good.csv", "bad.csv"),
    )


@scenario("unified_last_attempt_fallback")
def _():
    def phase12(*, state, **_kwargs):
        state.all_candidates = [
            table for table in ["bad.csv", "good.csv"] if table.casefold() not in state.excluded_tables
        ]
        state.rejection_keep_tables = []
        state.rejection_skip_tables = ["bad.csv"] if "bad.csv" in state.all_candidates else []
        return (list(state.all_candidates), ["parks"], {}, "REJECT_KEYWORDS: still missing the organisation", "trace", 5)

    return Case(
        experiment(),
        Script(phase12=[phase12], phase3=[phase3_result()], phase4=[ANSWER]),
        files=("bad.csv", "good.csv"),
    )


@scenario("divided_last_attempt_fallback")
def _():
    def phase2(*, selection_state, **_kwargs):
        selection_state.rejection_keep_tables = []
        selection_state.rejection_skip_tables = ["bad.csv"]
        return ([], ["bad.csv", "good.csv"], {}, "REJECT_KEYWORDS: bad.csv has no matching rows", "trace", 4)

    return Case(
        experiment(discovery_architecture="divided"),
        Script(phase1=[phase1_result()], phase2=[phase2], phase3=[phase3_result()], phase4=[ANSWER]),
        files=("bad.csv", "good.csv"),
    )


@scenario("unified_sweep")
def _():
    def phase3(*_args, coder_context_level, **_kwargs):
        value = 41 if coder_context_level.value == "minimal" else 42
        return phase3_result(tokens=10, clean_code=f"print({value})", raw_result=str(value), structured_result=value)

    verdict = {
        "disposition": "incorrect", "requested_disposition": "incorrect",
        "requirements": [{"id": "r1", "text": "count the parks per district"}],
        "requirement_assessments": [{"id": "r1", "status": "failed"}],
        "critique": {"objections": [{"severity": "major", "text": "41 is not the count of parks"}]},
        "code_analysis": {"hardcoding_risk": "low"},
    }
    return Case(
        experiment(automatic_test_coder=True),
        Script(phase12=[SELECTED], phase3=[phase3], phase4=[ANSWER], judge=[(verdict, 5)]),
        log_context=NUMBER_CONTEXT,
    )


@scenario("unified_sweep_full_rejection_restart")
def _():
    discovery_calls = []

    def phase12(**_kwargs):
        discovery_calls.append(1)
        table = "first.csv" if len(discovery_calls) <= 2 else "second.csv"
        return ([table], ["value"], {}, "selection", "trace", 5)

    def phase3(*args, coder_context_level, **_kwargs):
        rejected = args[1][0] == "first.csv" and coder_context_level.value == "full"
        return phase3_result(
            rejected_reason="wrong tables" if rejected else "",
            raw_result=None if rejected else "42", structured_result=None if rejected else 42,
        )

    return Case(
        experiment(automatic_test_coder=True),
        Script(phase12=[phase12], phase3=[phase3], phase4=[ANSWER]),
        log_context={"SOURCE_EXPECTED_RESULT_TYPE": "number", "SOURCE_REFERENCE_RESULT": 42},
        files=("first.csv", "second.csv"),
    )


@scenario("unified_refusal")
def _():
    return Case(
        experiment(),
        Script(phase12=[SELECTED], phase3=[phase3_result()], phase4=[("The necessary data is not available.", 2)]),
    )


@scenario("unified_judge_adjudication")
def _():
    judgment = {
        "disposition": "alternative_correct", "requested_disposition": "alternative_correct",
        "requirements": [{"id": "r1", "text": "count the parks per district"}],
        "requirement_assessments": [{"id": "r1", "status": "verified"}],
        "critique": {"objections": []}, "code_analysis": {"hardcoding_risk": "low"},
    }
    return Case(
        experiment(),
        Script(
            phase12=[SELECTED],
            phase3=[phase3_result(clean_code="print(41)", raw_result="41", structured_result=41)],
            phase4=[ANSWER], judge=[(judgment, 7)],
        ),
        log_context={**NUMBER_CONTEXT, "SOURCE_EXPECTED_RESULT_DESCRIPTION": "the number of parks"},
    )


@scenario("unified_gate_block")
def _():
    return Case(
        experiment(),
        Script(
            phase12=[
                (["a.csv"], ["parks"], {}, "a.csv looks right.", "trace", 5),
                (["b.csv"], ["parks", "year"], {}, "b.csv has the year column.", "trace", 5),
            ],
            phase3=[
                phase3_result(
                    error="The selection contract was not verified.", raw_result=None, structured_result=None,
                    execution_error={"stage": "selection_contract_gate", "category": "selection_contract_gate", "retryable": False},
                    coder_context_audit={
                        "status": "invalid",
                        "validation_diagnostics": [{"category": "unknown_column", "table": "a.csv", "evidence": {"unknown": ["year"]}}],
                    },
                ),
                phase3_result(),
            ],
            phase4=[ANSWER],
        ),
        files=("a.csv", "b.csv"),
    )


@scenario("unified_execution_error_retry")
def _():
    coder_calls = []

    def phase3(*_args, **_kwargs):
        coder_calls.append(1)
        return phase3_result(
            error="result needs revision", raw_result=None, structured_result=None,
            execution_error={"stage": "result_validation", "category": "result_needs_revision", "retryable": len(coder_calls) == 1},
        )

    return Case(experiment(), Script(phase12=[SELECTED], phase3=[phase3]))


@scenario("divided_orchestrated_empty_context")
def _():
    return Case(
        experiment(discovery_architecture="divided", tool_access="orchestrated_context"),
        Script(phase1=[phase1_result()], prepare_context=[(prepared([]), {})]),
    )


@scenario("unified_empty_result_retry")
def _():
    return Case(
        experiment(),
        Script(
            phase12=[SELECTED],
            phase3=[phase3_result(raw_result="", structured_result=None), phase3_result()],
            phase4=[ANSWER],
        ),
    )


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_service_golden(name, golden, monkeypatch, tmp_path):
    import lakegen.service as service

    case = SCENARIOS[name]()
    replacements, log, capture = prepare_run(monkeypatch, tmp_path, service, case, SERVICE_NAMES)

    result = service.run_question(case.question, make_runtime(case.experiment, tmp_path), log_context=case.log_context)

    golden(f"service/{name}", {
        "result": normalise(result.to_dict(), replacements),
        "calls": log.calls,
        **capture.payload(),
    })
