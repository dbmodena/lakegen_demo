"""The engine: one question through discovery, coding and the answer.

``Pipeline.steps`` is a generator over the workflow of a ``RunContext``
with the phases of a ``PipelineDeps``: up to three discovery rounds, each
followed by the coding stage (direct attempts, or the automatic
coder-context sweep), until a result is final. It yields the signals and,
in a human-gated run, the requests of ``lakegen.pipeline.run.events``; a driver
answers the requests and the generator returns the ``QueryResult``.
``Pipeline.run`` is the autonomous driver: it drains the steps and never
expects a request. The stages live in ``stages``; the engine decides only
the order, the human's answers and when a round repeats something already
rejected.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator, Mapping
from typing import Any

from lakegen.config import paths
from lakegen.config.experiment import DiscoveryArchitecture
from lakegen.config.paths import BASE_DIR
from lakegen.datastructures.errors import WorkflowCancelled, WorkflowTimedOut
from lakegen.datastructures.evaluation import EvaluationContext
from lakegen.datastructures.results import QueryResult
from lakegen.pipeline.coding.attempts import CodeAttemptEvaluator
from lakegen.pipeline.observability.manifest import ExperimentManifest, create_manifest
from lakegen.pipeline.observability.recorder import Recorder, record_semantic_plan_telemetry
from lakegen.pipeline.observability.reproducibility import initialize_reproducibility
from lakegen.pipeline.replay import Replay, ReplayedDiscovery
from lakegen.pipeline.run.budgets import (
    Attempt,
    DedupeGuard,
    rejected_selection_signature,
    selection_plan_signature,
)
from lakegen.pipeline.run.context import RunContext
from lakegen.pipeline.run.events import (
    Abort,
    Answer,
    Approve,
    Reject,
    Request,
    ReviewTables,
    RunStarted,
    Signal,
    StageFinished,
    StageStarted,
)
from lakegen.pipeline.run.state import RunState
from lakegen.pipeline.stages import (
    CORRECTION_HEADER,
    MAX_CODE_ATTEMPTS,
    MAX_TABLE_ATTEMPTS,
    AnswerStage,
    CoderSweep,
    CodingStage,
    DiscoveryStage,
    Run,
    SemanticJudge,
    Verdict,
    selection_retry_feedback,
)
from lakegen.pipeline.wiring.deps import PipelineDeps
from lakegen.pipeline.wiring.hooks import StageHooks
from lakegen.retrieval.run_log import capture_retrieval_runs

logger = logging.getLogger(__name__)

# The helpers the tests import from the service under their old names.
_record_semantic_plan_telemetry = record_semantic_plan_telemetry
_rejected_selection_signature = rejected_selection_signature
_selection_plan_signature = selection_plan_signature
_selection_retry_feedback = selection_retry_feedback

__all__ = ["MAX_CODE_ATTEMPTS", "MAX_TABLE_ATTEMPTS", "Pipeline", "run_pipeline"]


class Pipeline:
    """The engine of one ``RunContext`` with the phases of a ``PipelineDeps``.

    Attributes:
        ctx: The settings of the runs.
        deps: The phases and resources the runs call.
        last_discovery_tokens: The last run's discovery tokens by phase
            (keywords, selection), for a driver's summary.

    Args:
        ctx: The settings of the runs.
        deps: The phases and resources the runs call.
    """

    def __init__(self, ctx: RunContext, deps: PipelineDeps) -> None:
        self.ctx = ctx
        self.deps = deps
        self.last_discovery_tokens: tuple[int, int] = (0, 0)

    def run(
        self,
        question: str,
        *,
        question_id: str | int | None = None,
        log_context: Mapping[str, Any] | None = None,
        manifest: ExperimentManifest | None = None,
        replay: Replay | None = None,
    ) -> QueryResult:
        """The autonomous run: every step, no questions asked.

        Args:
            question: The question.
            question_id: The question's id, if it has one.
            log_context: The benchmark fields of the question, if any.
            manifest: The run's manifest; one is created when None.
            replay: What an earlier run recorded that this run starts from.

        Returns:
            The run's result.

        Raises:
            RuntimeError: When the steps ask for an answer, which an autonomous run cannot give.
        """
        steps = self.steps(
            question,
            question_id=question_id,
            log_context=log_context,
            manifest=manifest,
            replay=replay,
        )
        try:
            event = next(steps)
            while True:
                if not isinstance(event, RunStarted | StageStarted | StageFinished):
                    raise RuntimeError(f"an autonomous run cannot answer {event!r}")
                event = steps.send(None)
        except StopIteration as stop:
            return stop.value

    def steps(
        self,
        question: str,
        *,
        question_id: str | int | None = None,
        log_context: Mapping[str, Any] | None = None,
        manifest: ExperimentManifest | None = None,
        interventions: Any = None,
        hooks: StageHooks | None = None,
        replay: Replay | None = None,
    ) -> Generator[Signal | Request, Answer | None, QueryResult]:
        """The run, step by step; see the module docstring.

        A failure inside the rounds ends the run as failed and is returned in
        the result; a stop or a timeout is recorded and raised again. Either
        way the recorder writes the run's record and logs.

        Args:
            question: The question.
            question_id: The question's id, if it has one.
            log_context: The benchmark fields of the question, if any.
            manifest: The run's manifest; one is created when None.
            interventions: The driver's ``HumanInterventionRecorder``; what it
                records goes into the result.
            hooks: The driver's streams and stop check, passed to the phases
                while they run.
            replay: What an earlier run recorded that this run starts from.

        Yields:
            The run's signals and, in a human-gated run, its requests.

        Returns:
            The run's result.

        Raises:
            RuntimeError: When the lake's tables directory holds no table.
            WorkflowCancelled: When the person stopped the run, once the result records it.
            WorkflowTimedOut: When a gate timed out, once the result records it.
            GeneratorExit: When the driver closed the run between two steps, once
                the result records it as cancelled.
        """
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
        yield RunStarted(tables=len(all_files))

        state = RunState()
        if replay is not None and replay.keywords is not None:
            state.last_keyword_round = replay.keywords.round()
            state.reuse_keywords = True
        run = Run(
            question=question,
            ctx=ctx,
            deps=deps,
            benchmark=benchmark,
            recorder=recorder,
            state=state,
            guard=DedupeGuard(),
            llm=llm,
            solr=solr,
            prompt_manager=prompt_manager,
            all_files=all_files,
            run_dir=paths.coding_dir() / manifest.run_id,
            reproducibility=reproducibility,
            attempt_evaluator=attempt_evaluator,
            hooks=hooks or StageHooks(),
        )
        judge = SemanticJudge(run)
        discovery = (
            ReplayedDiscovery(run, replay.selection)
            if replay is not None and replay.selection is not None
            else DiscoveryStage(run)
        )
        coding = (
            CoderSweep(run, judge) if experiment.automatic_test_coder else CodingStage(run, judge)
        )

        retrieval_log_context = {
            **benchmark.to_log_fields(),
            "EXPERIMENT_ID": experiment.experiment_id,
        }
        with capture_retrieval_runs(retrieval_log_context) as retrieval_runs:
            run.retrieval_runs = retrieval_runs
            try:
                return (yield from _rounds(run, discovery, coding))
            except (WorkflowCancelled, WorkflowTimedOut) as exc:
                # The person stopped the run, or a gate timed out: the record
                # says so, and the surface decides what to show.
                result.status = "cancelled" if isinstance(exc, WorkflowCancelled) else "timed_out"
                result.error = f"{type(exc).__name__}: {exc}"
                raise
            except GeneratorExit:
                # The driver closed the run between two steps.
                result.status = "cancelled"
                result.error = result.error or "Workflow stopped by user."
                raise
            except Exception as exc:
                result.status = "failed"
                result.error = f"{type(exc).__name__}: {exc}"
                result.errors.append({"phase": "pipeline", "message": result.error})
                return result
            finally:
                if interventions is not None:
                    result.human_interventions = interventions.to_list()
                tokens = recorder.discovery_phase_tokens
                self.last_discovery_tokens = (tokens["p1"], tokens["p2"])
                recorder.finish(
                    retrieval_runs=retrieval_runs,
                    retrieval_log_context=retrieval_log_context,
                    selected=state.selected,
                    keywords=state.keywords,
                    reasoning=state.reasoning,
                    trace=state.trace,
                    error=state.error,
                    selection_state=state.selection_state,
                    selection_states=state.selection_states,
                    selection_attempts=state.selection_attempts,
                    last_generated=state.last_generated,
                    manifest=manifest,
                    reproducibility=reproducibility,
                    benchmark=benchmark,
                )


def _rounds(
    run: Run,
    discovery: DiscoveryStage | ReplayedDiscovery,
    coding: CodingStage | CoderSweep,
) -> Generator[Signal | Request, Answer | None, QueryResult]:
    """The discovery rounds, each followed by coding, until a result is final or the budget is spent.

    Automatic rounds consume the budget; a person's re-rounds do not. A run
    that spends it ends failed, or blocked by the selection, and on a
    surface that wants an answer anyway one is written from the failure.

    Args:
        run: The run in progress.
        discovery: The discovery stage, or the replay of an earlier run's selection.
        coding: The coding stage: direct attempts, or the coder-context sweep.

    Yields:
        The rounds' signals and requests.

    Returns:
        The run's result.
    """
    state, result = run.state, run.result
    index = 0  # automatic rounds consume the budget; a person's re-rounds do not
    while index < MAX_TABLE_ATTEMPTS:
        verdict = yield from _round(run, discovery, coding, Attempt(index, MAX_TABLE_ATTEMPTS))
        if verdict is Verdict.RETURN:
            return result
        if verdict is Verdict.STOP:
            break
        if verdict is Verdict.REDISCOVER:
            index += 1
    result.status = (
        "blocked_by_selection"
        if result.pipeline_stages["code_execution"] == "blocked_by_selection"
        else "failed"
    )
    result.error = state.error or "LakeGen could not produce an executable answer."
    if run.ctx.surface.answer_on_failure and state.last_generated is not None:
        # The step-by-step surfaces still want an answer: synthesise
        # one from the failure, the status staying failed.
        yield StageStarted("answer")
        AnswerStage(run).run_fallback(
            f"Execution failed after {MAX_CODE_ATTEMPTS} attempts. Last error: {state.error}"
        )
        yield StageFinished(
            "answer",
            message=result.answer,
            details={"answer": result.answer, "tokens": result.tokens["p4"]},
        )
    return result


def _round(
    run: Run,
    discovery: DiscoveryStage | ReplayedDiscovery,
    coding: CodingStage | CoderSweep,
    table_attempt: Attempt,
) -> Generator[Signal | Request, Answer | None, Verdict | None]:
    """One discovery round, its review in a gated run, and the coding of its selection.

    Args:
        run: The run in progress.
        discovery: The discovery stage, or the replay of an earlier run's selection.
        coding: The coding stage: direct attempts, or the coder-context sweep.
        table_attempt: The round's place in the discovery budget.

    Yields:
        The round's signals and requests.

    Returns:
        What the engine does next; None when the person sent the round back,
        which runs again without consuming the budget.

    Raises:
        TypeError: When a table review is answered with something other than
            Approve, Reject or Abort.
    """
    state, result, experiment = run.state, run.result, run.experiment
    selection_record: dict[str, Any] = {
        "attempt": table_attempt.number,
        "selected_datasets": [],
        "keywords": [],
        "outcome": "discovery_started",
        "rejection_feedback": "",
    }
    state.selection_attempts.append(selection_record)
    round_hint = state.hint
    state.architect_feedback = ""
    yield StageStarted("discovery", table_attempt.number)
    outcome, keywords_rejected = yield from discovery.run_round(table_attempt, selection_record)
    yield _discovery_finished(state, outcome, keywords_rejected, table_attempt, round_hint)
    state.forced_selection = False
    if keywords_rejected:
        if state.keyword_search_exhausted and not outcome.candidates:
            # The writer already rewrote its keywords as often as the budget allows.
            state.error = (
                "The keyword search found no table: every proposal matched nothing "
                "or was refused, and its retries are spent."
            )
            return Verdict.STOP
        return Verdict.REDISCOVER
    if run.ctx.surface.human_gated:
        answer = yield ReviewTables(
            keywords=list(state.keywords),
            selected=list(state.selected),
            reasoning=state.reasoning,
            tokens=outcome.tokens if experiment.use_unified_agent else outcome.phase_tokens[1],
            after_rejection=state.after_coder_rejection,
            architect_feedback=state.architect_feedback,
        )
        state.after_coder_rejection = False
        if isinstance(answer, Abort):
            result.status = "cancelled"
            result.error = answer.reason
            return Verdict.RETURN
        if isinstance(answer, Reject):
            _tables_sent_back(run, answer)
            return None
        if not isinstance(answer, Approve):
            raise TypeError(
                f"a table review is answered with Approve, Reject or Abort, not {answer!r}"
            )
    verdict = _repeated_rejection(run, table_attempt, selection_record)
    if verdict is None:
        plan_signature = selection_plan_signature(
            state.selected, state.selection_state.selection_plan
        )
        verdict = _repeated_plan(run, table_attempt, selection_record, plan_signature)
    if verdict is None:
        yield StageStarted("coding", table_attempt.number)
        verdict = yield from coding.run_coding(table_attempt, selection_record, plan_signature)
        yield StageFinished("coding", table_attempt.number, verdict.value)
    return verdict


def _discovery_finished(
    state: RunState,
    outcome: Any,
    keywords_rejected: bool,
    table_attempt: Attempt,
    round_hint: str,
) -> StageFinished:
    """The signal that a discovery round ended, with what it found.

    Args:
        state: The run's state, as the round left it.
        outcome: The round's ``DiscoveryOutcome``.
        keywords_rejected: Whether the round asked for new keywords.
        table_attempt: The round's place in the discovery budget.
        round_hint: The hint the round started with.

    Returns:
        The ``StageFinished`` signal of the round.
    """
    return StageFinished(
        "discovery",
        table_attempt.number,
        "keywords_rejected"
        if keywords_rejected
        else "forced_selection"
        if state.forced_selection
        else "selected",
        message=state.reasoning,
        details={
            "keywords": list(state.keywords),
            "selected": list(state.selected),
            "candidates": list(outcome.candidates),
            "metadata": dict(state.solr_meta),
            "reasoning": state.reasoning,
            "trace": state.trace,
            "selection_trace": outcome.selection_trace,
            "selection_plan": dict(state.selection_state.selection_plan),
            "tokens": outcome.tokens,
            "phase_tokens": outcome.phase_tokens,
            "hint": round_hint,
        },
    )


def _tables_sent_back(run: Run, answer: Reject) -> None:
    """The person rejected the round's tables: set up the next round as they asked.

    In the divided architecture the next round writes new keywords, the
    current ones avoided, or reuses them; either way it gets the round's
    architect feedback and the person's hint.

    Args:
        run: The run in progress.
        answer: The person's rejection.
    """
    state = run.state
    if run.experiment.discovery_architecture == DiscoveryArchitecture.DIVIDED:
        if answer.regenerate_keywords:
            state.attempted_keywords.extend(state.keywords)
            state.attempted_keywords = list(dict.fromkeys(state.attempted_keywords))
        else:
            state.reuse_keywords = True
    state.hint = next_round_hint(state.architect_feedback, answer.hint)


def run_pipeline(
    question: str,
    ctx: RunContext,
    deps: PipelineDeps,
    *,
    question_id: str | int | None = None,
    log_context: Mapping[str, Any] | None = None,
    manifest: ExperimentManifest | None = None,
    replay: Replay | None = None,
) -> QueryResult:
    """Run the autonomous LakeGen workflow for one question."""
    return Pipeline(ctx, deps).run(
        question,
        question_id=question_id,
        log_context=log_context,
        manifest=manifest,
        replay=replay,
    )


def next_round_hint(architect_feedback: str, user_hint: str) -> str:
    """A rejected round's architect feedback, plus what the person added."""
    if architect_feedback and user_hint:
        return f"{architect_feedback} User feedback: {user_hint}"
    return architect_feedback or user_hint


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
        state.error + " Select at least one different table. Use these missing "
        f"requirements as retrieval concepts: {state.hint}"
    )
    if not attempt.last:
        return Verdict.REDISCOVER
    result.status = "rejected"
    result.pipeline_stages["code_execution"] = "tables_rejected"
    return Verdict.STOP


def _repeated_plan(
    run: Run, attempt, selection_record: dict[str, Any], plan_signature: str
) -> Verdict | None:
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
