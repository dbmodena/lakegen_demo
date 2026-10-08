"""The engine: one question through discovery, coding and the answer.

``Pipeline.steps`` is a generator over the workflow of a ``RunContext``
with the phases of a ``PipelineDeps``: up to three discovery rounds, each
followed by the coding stage (direct attempts, or the automatic
coder-context sweep), until a result is final. It yields the signals and,
in a human-gated run, the requests of ``lakegen.pipeline.events``; a driver
answers the requests and the generator returns the ``QueryResult``.
``Pipeline.run`` is the autonomous driver: it drains the steps and never
expects a request. The stages live in ``stages``; the engine decides only
the order, the human's answers and when a round repeats something already
rejected.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Generator, Mapping
from typing import Any

from lakegen.code_attempts import CodeAttemptEvaluator
from lakegen.core import paths
from lakegen.core.config import BASE_DIR
from lakegen.core.resources import capture_retrieval_runs, make_retrieval_run_observer
from lakegen.domain.evaluation import EvaluationContext
from lakegen.experiment_config import DiscoveryArchitecture
from lakegen.keyword_memory import (
    format_question_retrieval_memory,
    load_keyword_memory,
    load_question_retrieval_memory,
    question_memory_key,
)
from lakegen.manifest import ExperimentManifest, create_manifest
from lakegen.observability.recorder import Recorder, record_semantic_plan_telemetry
from lakegen.pipeline.budgets import (
    Attempt,
    DedupeGuard,
    rejected_selection_signature,
    selection_plan_signature,
)
from lakegen.pipeline.context import RunContext
from lakegen.pipeline.deps import PipelineDeps
from lakegen.pipeline.events import (
    Abort,
    Answer,
    Approve,
    Reject,
    Request,
    ReviewTables,
    Signal,
    StageFinished,
    StageStarted,
)
from lakegen.pipeline.stages import (
    CORRECTION_HEADER,
    MAX_CODE_ATTEMPTS,
    MAX_TABLE_ATTEMPTS,
    CoderSweep,
    CodingStage,
    DiscoveryStage,
    Run,
    SemanticJudge,
    Verdict,
    selection_retry_feedback,
)
from lakegen.pipeline.state import RunState
from lakegen.reproducibility import initialize_reproducibility
from lakegen.service_models import QueryResult

logger = logging.getLogger(__name__)

# The helpers the tests import from the service under their old names.
_record_semantic_plan_telemetry = record_semantic_plan_telemetry
_rejected_selection_signature = rejected_selection_signature
_selection_plan_signature = selection_plan_signature
_selection_retry_feedback = selection_retry_feedback

__all__ = ["MAX_CODE_ATTEMPTS", "MAX_TABLE_ATTEMPTS", "Pipeline", "run_pipeline"]


def _open_keyword_memory(state: RunState, ctx: RunContext, question: str) -> None:
    """Load the durable keyword memory of the lake and mode into the run's state.

    Unified lexical memory belongs to the whole question, not one selection
    round: zero-result knowledge survives coder and selection retries.
    Historical searches are audit data and do not consume the per-question
    retry budget; only failed combinations affect future runs.
    """
    state.keyword_memory_path = paths.keyword_memory_path()
    state.keyword_memory_scope = "|".join((
        ctx.lake,
        ctx.retrieval.mode.value,
        ctx.retrieval.representation_version,
        ",".join(ctx.retrieval.lexical_query_fields or ()),
        "q.op=AND",
    ))
    _persistent_keyword_history, state.failed_keyword_combinations = load_keyword_memory(
        state.keyword_memory_path, state.keyword_memory_scope
    )
    state.retrieval_question_key = question_memory_key(question)
    state.retrieval_memory_summary = format_question_retrieval_memory(
        load_question_retrieval_memory(
            state.keyword_memory_path, state.keyword_memory_scope, state.retrieval_question_key
        )
    )


class Pipeline:
    def __init__(self, ctx: RunContext, deps: PipelineDeps) -> None:
        self.ctx = ctx
        self.deps = deps

    def run(
        self,
        question: str,
        *,
        question_id: str | int | None = None,
        log_context: Mapping[str, Any] | None = None,
        manifest: ExperimentManifest | None = None,
    ) -> QueryResult:
        """The autonomous run: every step, no questions asked."""
        steps = self.steps(question, question_id=question_id, log_context=log_context, manifest=manifest)
        try:
            event = next(steps)
            while True:
                if not isinstance(event, StageStarted | StageFinished):
                    raise RuntimeError(f"an autonomous run cannot answer {event!r}")
                event = steps.send(None)
        except StopIteration as stop:
            return stop.value

    def steps(  # noqa: C901, PLR0915
        self,
        question: str,
        *,
        question_id: str | int | None = None,
        log_context: Mapping[str, Any] | None = None,
        manifest: ExperimentManifest | None = None,
    ) -> Generator[Signal | Request, Answer | None, QueryResult]:
        """The run, step by step; see the module docstring."""
        ctx, deps = self.ctx, self.deps
        recorder = Recorder(question, ctx, deps)
        result = recorder.result
        benchmark = EvaluationContext.from_log_context(log_context)
        attempt_evaluator = CodeAttemptEvaluator(
            expected_result_type=benchmark.expected_result_type,
            reference_result=benchmark.reference_result,
            expected_description=benchmark.expected_description,
            evaluation_contract=dict(benchmark.evaluation_contract),
        )
        result.code_evaluation = attempt_evaluator.initial_evaluation()
        experiment = ctx.experiment
        reproducibility = initialize_reproducibility(experiment.seed)
        manifest = manifest or create_manifest(
            experiment,
            base_dir=BASE_DIR,
            question=question,
            question_id=question_id if question_id is not None else benchmark.source_id,
        )
        result.manifest = manifest.model_dump(mode="json")
        result.configuration = dict(manifest.resolved_config)
        deps.persist_manifest(manifest, paths.manifests_dir())
        llm, _token_counter = deps.get_llm(ctx.model_name)
        solr = deps.get_solr(ctx.lake)
        prompt_manager = deps.get_prompt_manager()
        all_files = deps.get_all_table_files(ctx.tables_dir)
        if not all_files:
            raise RuntimeError(f"No CSV or Parquet tables found in {ctx.tables_dir}")

        state = RunState()
        _open_keyword_memory(state, ctx, question)
        run = Run(
            question=question, ctx=ctx, deps=deps, benchmark=benchmark, recorder=recorder,
            state=state, guard=DedupeGuard(), llm=llm, solr=solr, prompt_manager=prompt_manager,
            all_files=all_files, run_dir=paths.coding_dir() / uuid.uuid4().hex,
            reproducibility=reproducibility, attempt_evaluator=attempt_evaluator,
        )
        judge = SemanticJudge(run)
        discovery = DiscoveryStage(run)
        coding = CoderSweep(run, judge) if experiment.automatic_test_coder else CodingStage(run, judge)
        gated = ctx.surface.human_gated

        retrieval_log_context = {**benchmark.to_log_fields(), "EXPERIMENT_ID": experiment.experiment_id}
        with capture_retrieval_runs(retrieval_log_context) as retrieval_runs:
            run.retrieval_runs = retrieval_runs
            run.retrieval_observer = make_retrieval_run_observer(retrieval_log_context, retrieval_runs)
            try:
                index = 0  # automatic rounds consume the budget; a person's re-rounds do not
                while index < MAX_TABLE_ATTEMPTS:
                    table_attempt = Attempt(index, MAX_TABLE_ATTEMPTS)
                    selection_record: dict[str, Any] = {
                        "attempt": table_attempt.number,
                        "selected_datasets": [],
                        "keywords": [],
                        "outcome": "discovery_started",
                        "rejection_feedback": "",
                    }
                    state.selection_attempts.append(selection_record)
                    yield StageStarted("discovery", table_attempt.number)
                    outcome, keywords_rejected = yield from discovery.run_round(table_attempt, selection_record)
                    yield StageFinished("discovery", table_attempt.number, "keywords_rejected" if keywords_rejected else "selected")
                    if keywords_rejected:
                        index += 1
                        continue
                    if gated:
                        answer = yield ReviewTables(
                            keywords=list(state.keywords), selected=list(state.selected), reasoning=state.reasoning,
                            tokens=outcome.tokens if experiment.use_unified_agent else outcome.phase_tokens[1],
                            after_rejection=state.after_coder_rejection,
                        )
                        state.after_coder_rejection = False
                        if isinstance(answer, Abort):
                            result.status = "cancelled"
                            result.error = answer.reason
                            return result
                        if isinstance(answer, Reject):
                            if experiment.discovery_architecture == DiscoveryArchitecture.DIVIDED:
                                state.attempted_keywords.extend(state.keywords)
                                state.attempted_keywords = list(dict.fromkeys(state.attempted_keywords))
                            state.hint = answer.hint
                            continue
                        if not isinstance(answer, Approve):
                            raise TypeError(f"a table review is answered with Approve, Reject or Abort, not {answer!r}")
                    verdict = _repeated_rejection(run, table_attempt, selection_record)
                    if verdict is None:
                        plan_signature = selection_plan_signature(state.selected, state.selection_state.selection_plan)
                        verdict = _repeated_plan(run, table_attempt, selection_record, plan_signature)
                    if verdict is None:
                        yield StageStarted("code", table_attempt.number)
                        verdict = yield from coding.run_coding(table_attempt, selection_record, plan_signature)
                        yield StageFinished("code", table_attempt.number, verdict.value)
                    if verdict is Verdict.RETURN:
                        return result
                    if verdict is Verdict.STOP:
                        break
                    index += 1
                result.status = (
                    "blocked_by_selection"
                    if result.pipeline_stages["code_execution"] == "blocked_by_selection"
                    else "failed"
                )
                result.error = state.error or "LakeGen could not produce an executable answer."
                return result
            except Exception as exc:
                result.status = "failed"
                result.error = f"{type(exc).__name__}: {exc}"
                result.errors.append({"phase": "pipeline", "message": result.error})
                return result
            finally:
                recorder.finish(
                    retrieval_runs=retrieval_runs,
                    retrieval_log_context=retrieval_log_context,
                    selected=state.selected, keywords=state.keywords, reasoning=state.reasoning,
                    trace=state.trace, error=state.error,
                    selection_state=state.selection_state, selection_states=state.selection_states,
                    selection_attempts=state.selection_attempts, last_generated=state.last_generated,
                    manifest=manifest, reproducibility=reproducibility, benchmark=benchmark,
                )


def run_pipeline(
    question: str,
    ctx: RunContext,
    deps: PipelineDeps,
    *,
    question_id: str | int | None = None,
    log_context: Mapping[str, Any] | None = None,
    manifest: ExperimentManifest | None = None,
) -> QueryResult:
    """Run the autonomous LakeGen workflow for one question."""
    return Pipeline(ctx, deps).run(question, question_id=question_id, log_context=log_context, manifest=manifest)


def _repeated_rejection(run: Run, attempt, selection_record: dict[str, Any]) -> Verdict | None:
    """The round reproduced a selection the coder already rejected: hint it away, or give up."""
    state, result = run.state, run.result
    prior = run.guard.prior_rejection(state.selected)
    if prior is None:
        return None
    selection_record["outcome"] = (
        "rejected_selection_excluded" if not attempt.last else "tables_rejected_no_alternative"
    )
    state.error = "No verified alternative selection remains. " + str(prior)
    state.hint = (
        state.error
        + " Select at least one different table. Use these missing "
        f"requirements as retrieval concepts: {state.hint}"
    )
    if not attempt.last:
        return Verdict.REDISCOVER
    result.status = "rejected"
    result.pipeline_stages["code_execution"] = "tables_rejected"
    return Verdict.STOP


def _repeated_plan(run: Run, attempt, selection_record: dict[str, Any], plan_signature: str) -> Verdict | None:
    """The round reproduced a plan the selection-contract gate refused: correct it, or give up."""
    state, result = run.state, run.result
    feedback = run.guard.repeated_plan(state.selected, plan_signature)
    if feedback is None:
        return None
    selection_record["outcome"] = "unchanged_rejected_plan"
    selection_record["rejection_feedback"] = feedback
    state.error = "Discovery repeated a plan already rejected by the selection contract gate."
    state.hint = CORRECTION_HEADER + json.dumps(feedback, ensure_ascii=False, sort_keys=True)
    if not attempt.last:
        return Verdict.REDISCOVER
    result.pipeline_stages["code_execution"] = "blocked_by_selection"
    return Verdict.STOP
