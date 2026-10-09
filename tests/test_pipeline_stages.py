"""The stages decide what follows a coding attempt; the engine only obeys."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lakegen.config.experiment import ExperimentConfig
from lakegen.domain.outcomes import CodeOutcome, CodeStatus
from lakegen.observability.recorder import Recorder
from lakegen.pipeline.budgets import Attempt, DedupeGuard
from lakegen.pipeline.context import RunContext
from lakegen.pipeline.deps import PipelineDeps
from lakegen.pipeline.stages import AnswerStage, CodingStage, Run, SemanticJudge, Verdict
from lakegen.pipeline.state import RunState


def _run(tmp_path: Path, *, phase4=None) -> Run:
    config = ExperimentConfig(experiment_id="x", core="nyc", model="openai.gpt-oss-120b")
    runtime = SimpleNamespace(model_name=config.model, solr_core="nyc", csv_dir=tmp_path, portal_name="NYC",
                              retrieval=config.retrieval.to_runtime(), experiment=config)
    ctx = RunContext.from_runtime(runtime)
    deps = PipelineDeps(**{name: (phase4 if name == "phase4_synthesize" else lambda *a, **k: None)
                           for name in PipelineDeps.__dataclass_fields__})
    recorder = Recorder("q", ctx, deps)
    evaluator = SimpleNamespace(enabled=False)
    return Run(question="q", ctx=ctx, deps=deps, benchmark=SimpleNamespace(), recorder=recorder, state=RunState(),
               guard=DedupeGuard(), llm=None, solr=None, prompt_manager=None, all_files=["a.csv"], run_dir=tmp_path,
               reproducibility=SimpleNamespace(effective_seed=0), attempt_evaluator=evaluator)


def _outcome(status: CodeStatus, **overrides) -> CodeOutcome:
    base = dict(generated=None, code="c", tokens=1, raw_result=None, error="boom", coder_runs=1, retryable=False)
    base.update(overrides)
    return CodeOutcome(status, **base)


def test_a_failed_attempt_retries_only_while_retryable_and_attempts_remain(tmp_path):
    run = _run(tmp_path)
    stage = CodingStage(run, SemanticJudge(run))
    assert stage._failed(_outcome(CodeStatus.EXECUTION_ERROR, retryable=True), Attempt(0, 3))
    assert not stage._failed(_outcome(CodeStatus.EXECUTION_ERROR, retryable=True), Attempt(2, 3))
    assert not stage._failed(_outcome(CodeStatus.EXECUTION_ERROR, retryable=False), Attempt(0, 3))
    assert stage._failed(_outcome(CodeStatus.EXECUTION_ERROR, coder_runs=0), Attempt(2, 3))  # nothing ran: try again
    assert run.result.pipeline_stages["code_execution"] == "failed" and run.state.error == "boom"
    assert [e["type"] for e in run.result.errors] == ["execution_error"] * 4
    stage._failed(_outcome(CodeStatus.EXECUTION_ERROR, execution_error={"category": "missing_column"}), Attempt(0, 3))
    assert run.result.errors[-1]["type"] == "missing_column" and run.result.errors[-1]["details"] == {"category": "missing_column"}


def test_after_a_break_only_a_rejection_or_a_block_sends_discovery_back(tmp_path):
    run = _run(tmp_path)
    stage = CodingStage(run, SemanticJudge(run))
    run.result.pipeline_stages["code_execution"] = "failed"
    assert stage._after_break(Attempt(0, 3)) is Verdict.STOP
    run.result.pipeline_stages["code_execution"] = "tables_rejected"
    assert stage._after_break(Attempt(0, 3)) is Verdict.REDISCOVER
    assert stage._after_break(Attempt(2, 3)) is Verdict.STOP
    run.result.pipeline_stages["code_execution"] = "blocked_by_selection"
    assert stage._after_break(Attempt(1, 3)) is Verdict.REDISCOVER


def test_the_answer_stage_maps_the_validation_to_the_status(tmp_path):
    run = _run(tmp_path, phase4=lambda question, raw, llm, pm: ("There are 42 parks.", 7))
    AnswerStage(run).run_answer("42")
    result = run.result
    assert result.status == "completed" and result.answer == "There are 42 parks." and result.tokens["p4"] == 7
    assert result.pipeline_stages["code_execution"] == "succeeded" and result.pipeline_stages["final_answer"] == "valid"
    assert result.code_evaluation["final_execution_success"] and run.recorder.phase_invocation_counts["result"] == 1

    refusing = _run(tmp_path, phase4=lambda question, raw, llm, pm: ("I cannot answer this question.", 3))
    AnswerStage(refusing).run_answer("42")
    assert refusing.result.status in {"rejected", "failed"} and refusing.result.error
