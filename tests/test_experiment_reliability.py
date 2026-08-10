import json
import builtins
from types import SimpleNamespace

import pytest

from lakegen.experiment_config import ExperimentConfig
from lakegen.retrieval import RetrievalConfig
from lakegen.service import run_question
from lakegen.ui.state import LakeGenSession
from src.cli import _ask_input, _ask_yes_no
from lakegen.tracing import (
    HumanGate,
    HumanInterventionRecorder,
    build_llm_phase_records,
    summarize_final_ranking,
    normalize_hint,
)
from lakegen.runner import ExperimentRunner


def _runtime(config, tmp_path):
    (tmp_path / "table.csv").write_text("value\n42\n", encoding="utf-8")
    return SimpleNamespace(
        model_name=config.model,
        solr_core=config.core,
        csv_dir=tmp_path,
        portal_name="NYC",
        retrieval=RetrievalConfig(),
        experiment=config,
    )


def _mock_successful_tail(monkeypatch, *, generated=None):
    generated = generated or SimpleNamespace(
        tokens=4,
        clean_code="print(42)",
        code_raw="print(42)",
        rejected_reason="",
        error=None,
        raw_result="42",
    )
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["table.csv"])
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute",
        lambda *_args, **_kwargs: generated,
    )
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize", lambda *_args: ("The answer is 42.", 5)
    )
    monkeypatch.setattr("lakegen.service.persist_manifest", lambda *_args: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)


@pytest.mark.parametrize("architecture", ["unified", "divided"])
def test_csv_architecture_matches_manifest_and_trace(
    architecture, monkeypatch, tmp_path
):
    config = ExperimentConfig(
        discovery_architecture=architecture,
        interaction_mode="autonomous",
    )
    runtime = _runtime(config, tmp_path)
    _mock_successful_tail(monkeypatch)
    if architecture == "unified":
        monkeypatch.setattr(
            "lakegen.service.phase12_agent",
            lambda **_kwargs: (
                ["table.csv"], ["value"], {}, "selected", "trace", 3
            ),
        )
    else:
        monkeypatch.setattr(
            "lakegen.service.phase1_generate_keywords",
            lambda **_kwargs: (["value"], "raw", 1, "keywords"),
        )
        monkeypatch.setattr(
            "lakegen.service.phase2_select_tables",
            lambda **_kwargs: (
                ["table.csv"], ["table.csv"], {}, "selected", "trace", 2
            ),
        )
    logged = {}
    monkeypatch.setattr(
        "lakegen.service.save_experiment_log", lambda **kwargs: logged.update(kwargs)
    )

    result = run_question("Question?", runtime)

    assert result.status == "completed"
    assert logged["architecture"] == architecture
    assert result.manifest["resolved_config"]["discovery_architecture"] == architecture
    run_trace = logged["extra_fields"]["RUN_TRACE_JSON"]
    assert run_trace["architecture"] == architecture
    trace_config = run_trace["configuration"]
    assert trace_config["discovery_architecture"] == architecture


def test_llm_phase_records_are_non_overlapping_for_unified_and_divided():
    unified = build_llm_phase_records(
        total_tokens={"discovery": 10, "code": 20, "result": 30},
        phase_invocations={"discovery": 1, "code": 1, "result": 1},
    )
    divided = build_llm_phase_records(
        total_tokens={"discovery": 12, "code": 20, "result": 30},
        phase_invocations={"discovery": 2, "code": 1, "result": 1},
    )

    assert [record["phase"] for record in unified] == ["discovery", "code", "result"]
    assert [record["phase"] for record in divided] == ["discovery", "code", "result"]
    assert unified[0]["phase_invocation_count"] == 1
    assert divided[0]["phase_invocation_count"] == 2
    assert divided[0]["provider_call_count"] is None
    assert sum(record["total_tokens"] for record in divided) == 62
    assert len({record["phase"] for record in divided}) == 3


def test_service_telemetry_includes_discovery_and_code_retries(monkeypatch, tmp_path):
    config = ExperimentConfig(
        discovery_architecture="divided",
        interaction_mode="autonomous",
    )
    runtime = _runtime(config, tmp_path)
    _mock_successful_tail(monkeypatch)
    monkeypatch.setattr(
        "lakegen.service.phase1_generate_keywords",
        lambda **_kwargs: (["value"], "raw", 2, "keywords"),
    )
    phase2_attempt = {"count": 0}

    def phase2(**_kwargs):
        phase2_attempt["count"] += 1
        if phase2_attempt["count"] == 1:
            return [], ["table.csv"], {}, "REJECT_KEYWORDS: retry", "trace", 3
        return ["table.csv"], ["table.csv"], {}, "selected", "trace", 5

    monkeypatch.setattr("lakegen.service.phase2_select_tables", phase2)
    code_attempt = {"count": 0}

    def phase3(*_args, **_kwargs):
        code_attempt["count"] += 1
        if code_attempt["count"] == 1:
            return SimpleNamespace(
                tokens=7,
                clean_code="bad()",
                code_raw="bad()",
                rejected_reason="",
                error="boom",
                raw_result=None,
            )
        return SimpleNamespace(
            tokens=11,
            clean_code="print(42)",
            code_raw="print(42)",
            rejected_reason="",
            error=None,
            raw_result="42",
        )

    monkeypatch.setattr("lakegen.service.phase3_generate_and_execute", phase3)
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)

    result = run_question("Question?", runtime)
    records = {record["phase"]: record for record in result.llm_calls}

    assert records["discovery"]["phase_invocation_count"] == 4
    assert records["discovery"]["total_tokens"] == 12
    assert records["code"]["phase_invocation_count"] == 2
    assert records["code"]["total_tokens"] == 18
    assert records["result"]["phase_invocation_count"] == 1
    assert records["result"]["total_tokens"] == 5
    assert sum(record["total_tokens"] for record in result.llm_calls) == 35


def test_human_interventions_are_typed_run_local_and_do_not_store_free_text():
    first = HumanInterventionRecorder()
    second = HumanInterventionRecorder()
    first.record_approval(
        phase="discovery",
        gate=HumanGate.KEYWORD_APPROVAL,
        approved=True,
        elapsed_seconds=1.23,
    )
    first.record_approval(
        phase="discovery",
        gate=HumanGate.DATASET_APPROVAL,
        approved=False,
        elapsed_seconds=0.5,
    )
    first.record_hint(
        phase="discovery",
        gate=HumanGate.KEYWORD_HINT,
        provided=True,
        elapsed_seconds=0.25,
    )

    events = first.to_list()
    serialized = json.dumps(events)
    assert second.to_list() == []
    assert all("phase" in event and "gate" in event for event in events)
    assert events[0]["approved"] is True
    assert events[1]["approved"] is False
    assert events[2]["provided"] is True
    assert "code_approval" not in serialized
    assert "hint text" not in serialized
    assert all("value" not in event and "text" not in event for event in events)


def test_cli_records_approval_rejection_and_hint_without_hint_text(monkeypatch):
    answers = iter(["n", "sensitive free-form hint"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))
    recorder = HumanInterventionRecorder()

    approved = _ask_yes_no(
        "Approve?",
        recorder=recorder,
        phase="discovery",
        gate=HumanGate.DATASET_APPROVAL,
    )
    hint = _ask_input(
        "Hint",
        recorder=recorder,
        phase="discovery",
        gate=HumanGate.DATASET_HINT,
    )

    assert approved is False
    assert hint == "sensitive free-form hint"
    events = recorder.to_list()
    assert events[0]["gate"] == "dataset_approval"
    assert events[0]["approved"] is False
    assert events[1]["gate"] == "dataset_hint"
    assert events[1]["provided"] is True
    assert "sensitive free-form hint" not in json.dumps(events)


def test_chainlit_sessions_do_not_share_intervention_recorders():
    first = LakeGenSession()
    second = LakeGenSession()
    first.intervention_recorder.record_approval(
        phase="code",
        gate=HumanGate.FORCE_EXECUTION_CONFIRMATION,
        approved=True,
        elapsed_seconds=0.1,
    )

    assert len(first.intervention_recorder.to_list()) == 1
    assert second.intervention_recorder.to_list() == []


@pytest.mark.parametrize(
    ("raw", "normalized", "provided"),
    [
        ("use parks", "use parks", True),
        ("", "", False),
        ("   ", "", False),
        ("skip", "", False),
        ("none", "", False),
        ("no", "", False),
        ("SKIP", "", False),
        ("NoNe", "", False),
        ("NO", "", False),
    ],
)
def test_hint_normalization_drives_cli_telemetry(
    raw, normalized, provided, monkeypatch
):
    monkeypatch.setattr(builtins, "input", lambda _prompt: raw)
    recorder = HumanInterventionRecorder()

    value = _ask_input(
        "Hint",
        recorder=recorder,
        phase="discovery",
        gate=HumanGate.DATASET_HINT,
    )

    assert normalize_hint(raw) == normalized
    assert value == normalized
    assert recorder.to_list()[0]["provided"] is provided


def test_runner_passes_explicit_question_id_to_executor(tmp_path):
    config = ExperimentConfig(interaction_mode="autonomous")
    captured = {}

    def executor(question, runtime, *, question_id, log_context):
        captured.update(question_id=question_id, log_context=log_context)
        return SimpleNamespace()

    ExperimentRunner(config).run(
        "Same?",
        question_id="explicit-2",
        log_context={"SOURCE_ID": "fallback-1"},
        runtime_factory=lambda **_kwargs: object(),
        executor=executor,
    )

    assert captured == {
        "question_id": "explicit-2",
        "log_context": {"SOURCE_ID": "fallback-1"},
    }


@pytest.mark.parametrize(
    ("question_id", "context", "expected"),
    [
        ("explicit", {"SOURCE_ID": "fallback"}, "explicit"),
        (None, {"SOURCE_ID": "fallback"}, "fallback"),
    ],
)
def test_service_manifest_question_id_precedence(
    question_id, context, expected, monkeypatch, tmp_path
):
    config = ExperimentConfig(interaction_mode="autonomous")
    runtime = _runtime(config, tmp_path)
    _mock_successful_tail(monkeypatch)
    monkeypatch.setattr(
        "lakegen.service.phase12_agent",
        lambda **_kwargs: (["table.csv"], ["value"], {}, "selected", "trace", 3),
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)

    result = run_question(
        "Same?", runtime, question_id=question_id, log_context=context
    )

    assert result.manifest["question_id"] == expected


def test_question_without_id_is_deterministic_and_distinct_explicit_ids(
    monkeypatch, tmp_path
):
    config = ExperimentConfig(interaction_mode="autonomous")
    runtime = _runtime(config, tmp_path)
    _mock_successful_tail(monkeypatch)
    monkeypatch.setattr(
        "lakegen.service.phase12_agent",
        lambda **_kwargs: (["table.csv"], ["value"], {}, "selected", "trace", 3),
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)

    generated_a = run_question("Same?", runtime).manifest["question_id"]
    generated_b = run_question("Same?", runtime).manifest["question_id"]
    explicit_a = run_question("Same?", runtime, question_id="a").manifest["question_id"]
    explicit_b = run_question("Same?", runtime, question_id="b").manifest["question_id"]

    assert generated_a == generated_b
    assert explicit_a == "a"
    assert explicit_b == "b"


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_ranking_uses_only_final_attempt_and_marks_final_selection(mode):
    runs = [
        {
            "retrieval_attempt": 1,
            "mode": "keyword",
            "hits": [
                {"resource_id": "duplicate", "rank": 1, "score": 0.9},
                {"resource_id": "old", "rank": 2, "score": 0.5},
            ],
        },
        {
            "retrieval_attempt": 2,
            "mode": mode,
            "hits": [
                {"resource_id": "duplicate", "rank": 1, "score": 0.8},
                {"resource_id": "chosen", "rank": 2, "score": 0.7},
            ],
        },
    ]

    ranking = summarize_final_ranking(runs, ["chosen.parquet"])

    assert [item["resource_id"] for item in ranking] == ["duplicate", "chosen"]
    assert all(item["attempt"] == 2 and item["mode"] == mode for item in ranking)
    assert [item["selected"] for item in ranking] == [False, True]
