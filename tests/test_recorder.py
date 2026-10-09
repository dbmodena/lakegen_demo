"""The recorder holds what the stages report and the result they build."""

from __future__ import annotations

from types import SimpleNamespace

from lakegen.config.experiment import ExperimentConfig
from lakegen.observability.recorder import Recorder, record_semantic_plan_telemetry
from lakegen.pipeline.context import RunContext


def _ctx(**overrides) -> RunContext:
    config = ExperimentConfig(experiment_id="x", core="nyc", model="openai.gpt-oss-120b", **overrides)
    runtime = SimpleNamespace(model_name=config.model, solr_core="nyc", csv_dir="/tmp/t", portal_name="NYC",
                              retrieval=config.retrieval.to_runtime(), experiment=config)
    return RunContext.from_runtime(runtime)


def test_a_fresh_recorder_describes_the_run_and_counts_nothing_yet():
    recorder = Recorder("How many parks?", _ctx(), deps=None)
    assert recorder.result.question == "How many parks?" and recorder.result.status == "running"
    assert recorder.phase_invocation_counts == {"discovery": 0, "code": 0, "result": 0}
    assert recorder.discovery_phase_tokens == {"p1": 0, "p2": 0}
    telemetry = recorder.context_telemetry
    assert telemetry["discovery_architecture"] == "unified" and telemetry["agent_count"] == 1
    assert telemetry["configured_tool_access"] == "agentic" and telemetry["llm_invocations"] == 0
    assert telemetry["retrieval_mode"] == "keyword" and telemetry["orchestrator_retrieval_calls"] == {}
    assert Recorder("q", _ctx(discovery_architecture="divided"), deps=None).context_telemetry["agent_count"] == 2
    assert not recorder.seed_instruction_provided
    recorder.record_seed_instruction()
    assert recorder.seed_instruction_provided


def test_the_semantic_plan_audit_lands_on_the_result():
    recorder = Recorder("q", _ctx(), deps=None)
    recorder.semantic_plan({
        "semantic_plan_present": True, "semantic_plan_status": "verified", "semantic_plan_locked": True,
        "validation_diagnostics": [], "evidence_count": 2, "coder_started_after_verified_plan": True,
    })
    result = recorder.result
    assert result.semantic_plan_present and result.semantic_plan_status == "verified" and result.semantic_plan_locked
    assert result.evidence_count == 2 and result.coder_started_after_verified_plan
    assert result.coder_blocked_before_start_count == 0
    recorder.semantic_plan({"status": "invalid", "coder_started_after_verified_plan": False})
    assert result.coder_blocked_before_start_count == 1 and result.semantic_plan_final_status == "invalid"
    # The function the tests import from the service is the recorder's.
    from lakegen.pipeline.service import _record_semantic_plan_telemetry

    assert _record_semantic_plan_telemetry is record_semantic_plan_telemetry
