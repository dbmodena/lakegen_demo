"""The stages of a run: discovery, coding (direct or as a sweep), the answer.

Each stage is a small object over the ``Run`` (the resources of the run,
its state, its recorder) with one ``run`` method. Discovery returns whether
the round asked for new keywords; coding returns a ``Verdict`` that tells
the engine whether the result is final, whether discovery should run again,
or whether the run is over. The stages report what they did to the
recorder and leave what the next round needs in the state; the engine
decides nothing but the order.
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from lakegen.config.experiment import (
    CoderContextLevel,
    ExperimentConfig,
)
from lakegen.datastructures.evaluation import EvaluationContext
from lakegen.datastructures.outcomes import CodeOutcome, CodeStatus, DiscoveryOutcome
from lakegen.pipeline.coding.attempts import CodeAttemptEvaluator
from lakegen.pipeline.coding.sweep import run_coder_context_sweep
from lakegen.pipeline.discovery.keywords import KeywordSearch, strict_and_count
from lakegen.pipeline.discovery.state import DiscoveryState
from lakegen.pipeline.evaluation.output_validation import AnswerDisposition, validate_answer
from lakegen.pipeline.evidence.candidates import rejection_feedback_hint
from lakegen.pipeline.observability.recorder import Recorder
from lakegen.pipeline.run.budgets import Attempt, DedupeGuard
from lakegen.pipeline.run.context import RunContext
from lakegen.pipeline.run.events import (
    Answer,
    Approve,
    CoderRejected,
    Force,
    Rediscover,
    Regenerate,
    Request,
    ReviewKeywords,
    StageFinished,
    StageStarted,
)
from lakegen.pipeline.run.state import RunState
from lakegen.pipeline.strategies import build as build_strategy
from lakegen.pipeline.wiring.adapters import (
    code_outcome,
    sweep_outcome,
)
from lakegen.pipeline.wiring.deps import PipelineDeps
from lakegen.pipeline.wiring.hooks import StageHooks

MAX_CODE_ATTEMPTS = 3
MAX_TABLE_ATTEMPTS = 3

CORRECTION_HEADER = "STRUCTURED DISCOVERY CORRECTION (follow every item before confirming):\n"


class Verdict(StrEnum):
    """What the engine does after a coding stage."""

    RETURN = "return"
    """The result is final."""
    REDISCOVER = "rediscover"
    """The selection was rejected or blocked; discovery runs again."""
    STOP = "stop"
    """No more attempts; the run ends as it stands."""


@dataclass
class Run:
    """The run in progress, as the stages see it.

    Attributes:
        question: The question.
        ctx: The settings of the run.
        deps: The phases and resources the run calls.
        benchmark: The question's benchmark context.
        recorder: Holds the result and the telemetry the stages report.
        state: What the stages share and the rounds carry forward.
        guard: The selections and plans already rejected.
        llm: The run's model.
        solr: The lake's Solr client.
        prompt_manager: The prompt manager.
        all_files: The lake's table files.
        run_dir: The directory the coder works in.
        reproducibility: The run's ``ReproducibilityContext``.
        attempt_evaluator: Evaluates the coder's attempts against the benchmark.
        retrieval_runs: The retrieval runs of the question, as they happen.
        hooks: The driver's streams and stop check.
    """

    question: str
    ctx: RunContext
    deps: PipelineDeps
    benchmark: EvaluationContext
    recorder: Recorder
    state: RunState
    guard: DedupeGuard
    llm: Any
    solr: Any
    prompt_manager: Any
    all_files: list[str]
    run_dir: Path
    reproducibility: Any
    attempt_evaluator: CodeAttemptEvaluator
    retrieval_runs: list[dict[str, Any]] = field(default_factory=list)
    hooks: StageHooks = field(default_factory=StageHooks)

    @property
    def experiment(self) -> ExperimentConfig:
        """The run's experiment config."""
        return self.ctx.experiment

    @property
    def result(self):
        """The run's result, as the recorder holds it."""
        return self.recorder.result

    @property
    def code_evaluation_enabled(self) -> bool:
        """Whether the coder's attempts are evaluated against the benchmark."""
        return self.attempt_evaluator.enabled


def selection_retry_feedback(
    tables: list[str], audit: Mapping[str, Any] | None, error: str
) -> dict[str, Any]:
    """Compile one mode-neutral correction hint from the contract gate audit."""
    audit = audit or {}
    diagnostics = list(audit.get("validation_diagnostics") or audit.get("diagnostics") or [])
    action_by_category = {
        "unknown_column": "inspect the runtime schema and bind the exact column name",
        "missing_columns": "inspect the runtime schema and choose the required columns",
        "ambiguous_column": "inspect the candidate columns and resolve the binding",
        "invalid_join": "inspect both schemas and provide explicit compatible join keys",
        "unknown_join_column": "inspect both schemas and correct the join-column binding",
        "table_unreadable": "select a readable table that covers the same requirement",
        "unknown_table": "select only tables returned by the current retrieval",
    }
    corrections: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        item = diagnostic if isinstance(diagnostic, Mapping) else {"evidence": diagnostic}
        category = str(item.get("category") or "contract_not_grounded")
        correction = {
            "category": category,
            "action": action_by_category.get(
                category,
                "inspect the relevant runtime columns or values and revise the plan",
            ),
        }
        for key in ("table", "column", "columns", "evidence"):
            if item.get(key) not in (None, "", [], {}):
                correction[key] = item[key]
        corrections.append(correction)
    if not corrections:
        corrections.append(
            {
                "category": str(audit.get("status") or "contract_not_grounded"),
                "action": (
                    "inspect the selected runtime schemas and values, then provide exact "
                    "column bindings and explicit join keys"
                ),
                "evidence": error,
            }
        )
    keep_tables = not any(
        item["category"] in {"unknown_table", "table_unreadable"} for item in corrections
    )
    return {
        "retry_reason": "selection_contract_invalid",
        "keep_current_tables": keep_tables,
        "previous_selection": list(tables),
        "required_corrections": corrections,
        "instruction": (
            "Resolve every required correction before confirming. Do not repeat the "
            "previous plan unchanged."
        ),
    }


class SemanticJudge:
    """Adjudicates a pending code evaluation with the semantic judge, once per run.

    Attributes:
        run: The run in progress.
        llm: The judge's model, resolved on first use.
        requirements: The requirements the judge derived on its first call, reused after.

    Args:
        run: The run in progress.
    """

    def __init__(self, run: Run) -> None:
        self.run = run
        self.llm: Any = None
        self.requirements: list[dict[str, Any]] | None = None

    def adjudicate(self, evaluation: dict[str, Any], generated: Any) -> dict[str, Any]:
        """Settle an evaluation pending semantic review with the judge's verdict.

        An evaluation not pending review is returned as it is; with the judge
        disabled it becomes indeterminate. The judge's model is the run's when
        they are the same, and the requirements of its first judgment are
        reused for every later one.

        Args:
            evaluation: The deterministic evaluation of the code's result; updated in place.
            generated: The coder's result object.

        Returns:
            The evaluation, updated.
        """
        run, experiment = self.run, self.run.experiment
        if evaluation.get("evaluation_disposition") != "pending_semantic_review":
            return evaluation
        if not experiment.semantic_code_judge_enabled:
            evaluation.update(
                {
                    "evaluation_disposition": "indeterminate",
                    "supported_correct": False,
                    "semantic_judge_used": False,
                    "semantic_judge_model": experiment.semantic_code_judge_model,
                }
            )
            return evaluation
        if self.llm is None:
            self.llm = (
                run.llm
                if experiment.semantic_code_judge_model == run.ctx.model_name
                else run.deps.get_llm(experiment.semantic_code_judge_model)[0]
            )
        judgment, judge_tokens = run.deps.judge_semantic_code_result(
            question=run.question,
            expected_description=run.benchmark.expected_description,
            reference_result=run.benchmark.reference_result,
            selected_tables=run.state.selected,
            selected_metadata=run.state.solr_meta,
            generated_code=generated.clean_code or generated.code_raw,
            generated_result=getattr(generated, "structured_result", None),
            deterministic_evaluation=evaluation,
            llm=self.llm,
            prompt_manager=run.prompt_manager,
            requirements_override=self.requirements,
        )
        returned_requirements = judgment.get("requirements")
        if self.requirements is None and isinstance(returned_requirements, list):
            self.requirements = [
                dict(item) for item in returned_requirements if isinstance(item, Mapping)
            ]
        evaluation.update(_judgment_fields(judgment, judge_tokens, evaluation, experiment))
        return evaluation


def _judgment_fields(
    judgment: Mapping[str, Any],
    judge_tokens: int,
    evaluation: Mapping[str, Any],
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    disposition = judgment["disposition"]
    judge_error = str(judgment.get("judge_error") or "")
    assessments = judgment.get("requirement_assessments", [])
    if not isinstance(assessments, list):
        assessments = []
    assessment_counts = {
        status: sum(
            isinstance(item, Mapping) and item.get("status") == status for item in assessments
        )
        for status in ("verified", "failed", "unknown")
    }
    objections = judgment.get("critique", {}).get("objections", [])
    if not isinstance(objections, list):
        objections = []
    blocking_objection_count = sum(
        isinstance(item, Mapping)
        and str(item.get("severity") or "").casefold() in {"critical", "major"}
        for item in objections
    )
    requested_disposition = str(judgment.get("requested_disposition") or disposition)
    requirement_count = len(assessments)
    canonical_disposition = {
        "alternative_correct": "correct",
        "incorrect": "incorrect",
        "indeterminate": "completed_with_warnings",
    }.get(disposition, "completed_with_warnings")
    if judge_error:
        canonical_disposition = "completed_with_warnings"
    return {
        "evaluation_disposition": canonical_disposition,
        "semantic_correctness": disposition,
        "supported_correct": disposition == "alternative_correct",
        "semantic_pass_at_1": (
            bool(evaluation.get("semantic_pass_at_1"))
            or (
                disposition == "alternative_correct"
                and int(evaluation.get("attempt_count") or 0) == 1
            )
        ),
        "semantic_success_within_3": (
            bool(evaluation.get("semantic_success_within_3"))
            or (
                disposition == "alternative_correct"
                and 0 < int(evaluation.get("attempt_count") or 0) <= 3
            )
        ),
        "semantic_judge_used": True,
        "semantic_judge_model": experiment.semantic_code_judge_model,
        "semantic_judge_tokens": judge_tokens,
        "semantic_judge_parse_success": not bool(judge_error),
        "semantic_judge_status": "parse_failed" if judge_error else "completed",
        "semantic_judge_pipeline_version": "evidence_v2",
        "semantic_judge_requested_disposition": requested_disposition,
        "semantic_judge_downgraded": requested_disposition != disposition,
        "semantic_judge_requirement_count": requirement_count,
        "semantic_judge_verified_requirement_count": assessment_counts["verified"],
        "semantic_judge_failed_requirement_count": assessment_counts["failed"],
        "semantic_judge_unknown_requirement_count": assessment_counts["unknown"],
        "semantic_judge_evidence_coverage": (
            assessment_counts["verified"] / requirement_count if requirement_count else 0.0
        ),
        "semantic_judge_blocking_objection_count": blocking_objection_count,
        "semantic_judge_hardcoding_risk": str(
            judgment.get("code_analysis", {}).get("hardcoding_risk") or "unknown"
        ),
        "semantic_judgment": judgment,
    }


# --- discovery ---------------------------------------------------------------


class DiscoveryStage:
    """One discovery round: the configured strategy's agents, then what the round leaves behind.

    Attributes:
        run: The run in progress.
        strategy: The discovery strategy of the experiment's architecture and tool access.

    Args:
        run: The run in progress.
    """

    def __init__(self, run: Run) -> None:
        self.run = run
        experiment = run.experiment
        self.strategy = build_strategy(experiment.discovery_architecture)

    def run_round(
        self,
        attempt: Attempt,
        selection_record: dict[str, Any],
    ) -> Generator[Request, Answer, tuple[DiscoveryOutcome, bool]]:
        """Run the round and record it: its outcome, and whether it asked for new keywords.

        A strategy that writes keywords first gets them from ``keywords()``,
        reviewed in a gated run before the selection; the selection itself
        is reviewed by the engine.

        Args:
            attempt: The round's place in the discovery budget.
            selection_record: The round's record, filled in here.

        Yields:
            The keyword reviews of a gated run.

        Returns:
            The round's outcome, and whether it asked for new keywords.
        """
        run, state, strategy = self.run, self.run.state, self.strategy
        started = time.monotonic()
        keyword_round = (yield from self.keywords()) if strategy.writes_keywords_first else None
        if strategy.writes_keywords_first:
            selection_record["keyword_searches"] = list(state.keyword_searches)
        outcome = strategy.discover(self, keyword_round)
        state.selected = outcome.selected
        state.keywords = outcome.keywords
        state.solr_meta = outcome.metadata
        state.reasoning = outcome.reasoning
        state.trace = outcome.trace
        keywords_rejected = strategy.after_round(self, outcome, attempt)
        run.recorder.discovery_round(
            outcome,
            state=state,
            selection_record=selection_record,
            keywords_rejected=keywords_rejected,
            attempt=attempt,
            elapsed=round(time.monotonic() - started, 6),
            retrieval_runs=run.retrieval_runs,
        )
        return outcome, keywords_rejected

    def keywords(self) -> Generator[Request, Answer, tuple[Any, ...]]:
        """The divided architecture's keyword round; in a gated run, written until approved.

        A person who asked for a re-selection with the same keywords gets the
        last approved round back, its tokens already counted.

        Yields:
            In a gated run, a ``ReviewKeywords`` for each round written.

        Returns:
            The approved keyword round: keywords, raw text, tokens, reasoning.

        Raises:
            TypeError: When a keyword review is answered with something other
                than Approve or Regenerate.
        """
        state = self.run.state
        state.keyword_searches = []
        state.keyword_search_exhausted = False
        if state.reuse_keywords and state.last_keyword_round is not None:
            state.reuse_keywords = False
            keywords, raw, _tokens, reasoning = state.last_keyword_round
            return (keywords, raw, 0, reasoning)
        while True:
            keyword_round = self.keyword_round()
            state.last_keyword_round = keyword_round
            if not self.run.ctx.surface.human_gated:
                return keyword_round
            answer = yield ReviewKeywords(
                list(keyword_round[0]),
                keyword_round[2],
                hint=state.hint,
                raw=str(keyword_round[1]),
                reasoning=str(keyword_round[3]),
            )
            if isinstance(answer, Approve):
                return keyword_round
            if not isinstance(answer, Regenerate):
                raise TypeError(
                    f"a keyword review is answered with Approve or Regenerate, not {answer!r}"
                )
            self.run.recorder.discarded_keywords(keyword_round[2])
            state.attempted_keywords.extend(keyword_round[0])
            state.attempted_keywords = list(dict.fromkeys(state.attempted_keywords))
            state.hint = answer.hint

    def seeded_state(self) -> DiscoveryState:
        """A fresh round state, seeded with the tables earlier rounds excluded and carried.

        A coder rejection starts a genuinely new retrieval turn, so each
        attempt gets a fresh state rather than one that would repeat the last
        round's searches and selection.

        Returns:
            The seeded round state.
        """
        state = self.run.state
        seeded = DiscoveryState()
        seeded.excluded_tables = set(state.excluded_tables)
        seeded.carried_tables = list(state.carried_tables)
        seeded.carried_metadata = dict(state.carried_metadata)
        return seeded

    def keyword_round(self) -> tuple[Any, ...]:
        """The writer's keywords, rewritten while a strict-AND search of them matches nothing.

        In keyword and hybrid mode each proposal goes through ``KeywordSearch``
        before retrieval runs: a refused or empty one goes back to the writer,
        in the same conversation, with what the search learned, at most
        ``max_zero_result_retries`` times. The word sets proved empty stay
        banned for the rest of the run. The round's tokens and reasoning are
        those of every proposal.

        Returns:
            The round: the last proposal's keywords and raw text, and the
            tokens and reasoning of every proposal.
        """
        run, state, ctx = self.run, self.run.state, self.run.ctx
        count = strict_and_count(run.solr, ctx.retrieval)
        search = (
            KeywordSearch(
                count, state.zero_result_bans, 1 + run.experiment.discovery.max_zero_result_retries
            )
            if count is not None
            else None
        )
        # The first message stays the same through the round's conversation.
        known_bans = [sorted(ban) for ban in state.zero_result_bans]
        follow_ups: list[tuple[str, str]] = []
        tokens, reasoning = 0, ""
        while True:
            keywords, raw, turn_tokens, turn_reasoning = run.deps.write_keywords(
                query=run.question,
                llm=run.llm,
                pm=run.prompt_manager,
                hint=state.hint,
                portal_name=ctx.portal_name,
                avoid_keywords=state.attempted_keywords,
                value_search=ctx.retrieval.mode.value_keywords,
                verbatim_entities=ctx.retrieval.mode.verbatim_entities,
                zero_result_sets=known_bans,
                follow_ups=list(follow_ups),
                **run.hooks.placeholder_kwargs(cancel=False),
            )
            run.recorder.phase_invocation_counts["discovery"] += 1
            tokens += turn_tokens
            reasoning = f"{reasoning}\n\n{turn_reasoning}" if reasoning else turn_reasoning
            feedback = search.review(keywords) if search is not None else None
            if feedback is None:
                break
            print(f"[phase1] {feedback}", flush=True)
            follow_ups.append((raw, feedback))
            reasoning += f"\n\n--- Keyword search ---\n{feedback}"
            run.recorder.context_telemetry["llm_invocations"] += 1
        if search is not None:
            state.keyword_searches.extend(search.log)
            state.keyword_search_exhausted = search.exhausted
        return keywords, raw, tokens, reasoning

    def apply_rejection(self, outcome: DiscoveryOutcome, attempt: Attempt) -> bool:
        """The architect rejected the round: carry its judgment, hint the next round.

        A table it proved unusable is excluded, one it found useful carries
        forward with its metadata and, for the unified agent, its cached
        inspection. On the last attempt a selection is forced from this
        round's candidates, minus the tables just rejected.

        Args:
            outcome: The rejected round's outcome.
            attempt: The round's place in the discovery budget.

        Returns:
            Whether another round follows.
        """
        state = self.run.state
        state.attempted_keywords.extend(state.keywords)
        state.attempted_keywords = list(dict.fromkeys(state.attempted_keywords))
        state.carry(outcome.keep_tables, outcome.skip_tables)
        state.hint = rejection_feedback_hint(state.reasoning, outcome.keep_tables)
        state.architect_feedback = state.hint
        if attempt.last:
            surviving = [t for t in outcome.candidates if t.casefold() not in state.excluded_tables]
            state.selected = (surviving or outcome.candidates or self.run.all_files)[:3]
            state.forced_selection = True
            return False
        return True


# --- coding ------------------------------------------------------------------


class AnswerStage:
    """The result became an answer: synthesise it and validate it.

    Attributes:
        run: The run in progress.

    Args:
        run: The run in progress.
    """

    def __init__(self, run: Run) -> None:
        self.run = run

    def run_answer(self, raw_result: str) -> None:
        """Synthesise the answer from the code's result and set the run's status from its validation.

        Args:
            raw_result: The code's raw result.
        """
        run, result = self.run, self.run.result
        result.pipeline_stages["code_execution"] = "succeeded"
        result.code_evaluation["final_execution_success"] = True
        synthesis_started = time.monotonic()
        answer, synthesis_tokens = run.deps.phase4_synthesize(
            run.question, raw_result, run.llm, run.prompt_manager
        )
        run.recorder.phase_invocation_counts["result"] += 1
        result.answer = answer
        result.tokens["p4"] = synthesis_tokens
        result.phase_metrics["result"] = {
            "latency_seconds": round(time.monotonic() - synthesis_started, 6),
            "retries": 0,
        }
        validation = validate_answer(raw_result, answer)
        result.answer_disposition = validation.disposition.value
        result.pipeline_stages["final_answer"] = validation.disposition.value
        if validation.disposition == AnswerDisposition.VALID:
            result.status = "completed"
            result.error = ""
        elif validation.disposition == AnswerDisposition.REJECTED:
            result.status = "rejected"
            result.error = validation.reason
        else:
            result.status = "failed"
            result.error = validation.reason

    def run_fallback(self, raw_result: str) -> None:
        """No code succeeded: synthesise an answer from the failure, the status unchanged."""
        run, result = self.run, self.run.result
        synthesis_started = time.monotonic()
        answer, synthesis_tokens = run.deps.phase4_synthesize(
            run.question, raw_result, run.llm, run.prompt_manager
        )
        run.recorder.phase_invocation_counts["result"] += 1
        result.answer = answer
        result.tokens["p4"] = synthesis_tokens
        result.phase_metrics["result"] = {
            "latency_seconds": round(time.monotonic() - synthesis_started, 6),
            "retries": 0,
        }


class _Coding:
    """What the direct coder and the sweep share: the gate and the rejection.

    Attributes:
        run: The run in progress.
        judge: The run's semantic judge.
        answer: The answer stage, run when the code succeeds.

    Args:
        run: The run in progress.
        judge: The run's semantic judge.
    """

    def __init__(self, run: Run, judge: SemanticJudge) -> None:
        self.run = run
        self.judge = judge
        self.answer = AnswerStage(run)

    def previously_visible(self) -> list[str]:
        state = self.run.state
        return [
            table
            for table in state.selection_state.all_candidates[
                : state.selection_state.visible_candidate_count
            ]
            if table not in state.selected
        ][:5]

    def blocked(
        self, outcome: CodeOutcome, selection_record: dict[str, Any], plan_signature: str
    ) -> None:
        """The selection-contract gate stopped the coder: ask discovery for a corrected plan."""
        run, state, result = self.run, self.run.state, self.run.result
        selection_record["outcome"] = "selection_contract_blocked"
        result.pipeline_stages["code_execution"] = "blocked_by_selection"
        result.code_evaluation.update(
            {
                "evaluation_disposition": "blocked",
                "error_category": "selection_contract_gate",
            }
        )
        state.error = outcome.error
        audit = outcome.audit or {}
        if audit.get("status") == "missing" and state.last_coder_rejection:
            selection_record["recovery_from_rejection"] = dict(state.last_coder_rejection)
            state.error = (
                "Selection recovery produced no contract. Original "
                f"coder rejection: {state.last_coder_rejection['reason']} "
                f"Current gate: {outcome.error}"
            )
        feedback = selection_retry_feedback(state.selected, audit, state.error)
        run.guard.reject_plan(plan_signature, feedback)
        selection_record["rejection_feedback"] = feedback
        state.hint = CORRECTION_HEADER + json.dumps(feedback, ensure_ascii=False, sort_keys=True)

    def rejected(
        self, outcome: CodeOutcome, selection_record: dict[str, Any], *, who: str, level: str | None
    ) -> None:
        """The coder rejected the tables: remember the selection, hint the next round.

        The coder's own per-table judgment feeds the same cross-round memory
        as an architect rejection: a table it proved unusable is banned
        outright, one it found partially useful carries forward. The direct
        coder and the sweep both pass through here.

        Args:
            outcome: The coder's outcome.
            selection_record: The round's record.
            who: Who rejected the tables, as the next round's hint names it.
            level: The coder-context level of the sweep variant; None for the direct coder.
        """
        run, state, result = self.run, self.run.state, self.run.result
        state.last_coder_rejection = {
            "reason": outcome.error,
            "details": outcome.rejection_details,
            "tables": list(state.selected),
        }
        run.guard.reject_selection(state.selected, outcome.rejection_details, outcome.error)
        record: dict[str, Any] = {
            "phase": "code",
            "type": "tables_rejected",
            "message": outcome.error,
        }
        if level is not None:
            record["coder_context_level"] = level
        result.errors.append(record)
        result.pipeline_stages["code_execution"] = "tables_rejected"
        state.hint = (
            f"{who} rejected the previous table combination. "
            "First reassess these previously visible alternatives if retrieval returns them again: "
            f"{self.previously_visible()}. Select a different combination. Start a genuinely new "
            f"search only if none covers the missing requirement. Feedback: {outcome.error}"
        )
        state.error = outcome.error
        state.after_coder_rejection = True
        state.carry(outcome.keep_tables, outcome.skip_tables)


class CoderSweep(_Coding):
    """The automatic coder-context sweep: every context level on the same tables.

    Full context is the only reliable table-rejection gate, so it runs
    first; a rejected selection returns to discovery without the calls the
    reduced-context variants would spend.
    """

    def _sweep(self, levels: list[CoderContextLevel]) -> dict[str, Any]:
        run, state, ctx, experiment = self.run, self.run.state, self.run.ctx, self.run.experiment
        return run_coder_context_sweep(
            question=run.question,
            selected=state.selected,
            solr_meta=state.solr_meta,
            reasoning=state.reasoning,
            llm=run.llm,
            prompt_manager=run.prompt_manager,
            csv_dir=ctx.tables_dir,
            run_dir=run.run_dir,
            seed=run.reproducibility.effective_seed,
            record_seed_instruction=run.recorder.record_seed_instruction,
            expected_result_type=run.benchmark.expected_result_type,
            evaluation_enabled=run.code_evaluation_enabled,
            evaluator=run.attempt_evaluator,
            adjudicate=self.judge.adjudicate,
            generate_and_execute=run.deps.write_and_run_code,
            result=run.result,
            phase_invocation_counts=run.recorder.phase_invocation_counts,
            max_attempts=MAX_CODE_ATTEMPTS,
            context_levels=levels,
            selection_plan=dict(state.selection_state.selection_plan),
            source_field_names=run.benchmark.field_names,
            require_semantic_plan=experiment.require_semantic_plan,
        )

    def run_coding(
        self,
        attempt: Attempt,
        selection_record: dict[str, Any],
        plan_signature: str,
    ) -> Generator[Request | StageStarted, Answer, Verdict]:
        """Code the selection at full context, then at the reduced ones, and answer.

        A full-context run that the gate blocked or that rejected the tables
        sends the round back to discovery (or stops on the last round); the
        reduced-context variants run only on accepted tables.

        Args:
            attempt: The round's place in the discovery budget.
            selection_record: The round's record.
            plan_signature: The round's plan signature, for the gate's memory.

        Yields:
            The answer stage's signals.

        Returns:
            What the engine does next.
        """
        run, state, result, experiment = (
            self.run,
            self.run.state,
            self.run.result,
            self.run.experiment,
        )
        variants = self._sweep([CoderContextLevel.FULL])
        primary = variants[CoderContextLevel.FULL.value]
        sweep = sweep_outcome(primary, require_semantic_plan=experiment.require_semantic_plan)
        run.recorder.semantic_plan(sweep.audit)
        coder_audit = sweep.audit or {}
        selection_record["coder_brief_status"] = str(
            coder_audit.get("effective_coder_brief_status")
            or coder_audit.get("status")
            or "missing"
        )
        selection_record["coder_started"] = bool(
            coder_audit.get("coder_started_after_verified_plan")
        )
        selection_record["coder_outcome"] = sweep.raw_status
        if sweep.status is CodeStatus.BLOCKED_BY_SELECTION:
            self.blocked(sweep, selection_record, plan_signature)
            return Verdict.STOP if attempt.last else Verdict.REDISCOVER
        if sweep.status is CodeStatus.TABLES_REJECTED:
            self.rejected(
                sweep,
                selection_record,
                who="The full-context code generator",
                level=CoderContextLevel.FULL.value,
            )
            selection_record["outcome"] = "tables_rejected"
            selection_record["rejection_feedback"] = sweep.error
            return Verdict.STOP if attempt.last else Verdict.REDISCOVER

        variants.update(self._sweep([CoderContextLevel.SCHEMA_ONLY, CoderContextLevel.MINIMAL]))
        result.coder_context_experiment = {
            "shared_retrieval": True,
            "shared_tables": list(state.selected),
            "shared_keywords": list(state.keywords),
            "shared_reasoning": None,
            "primary_level": CoderContextLevel.FULL.value,
            "variants": variants,
        }
        selection_record["outcome"] = "accepted"
        result.code = sweep.code
        result.raw_result = primary["raw_result"]
        result.code_evaluation = primary["code_evaluation"]
        if sweep.status is CodeStatus.SUCCESS:
            yield StageStarted("answer")
            self.answer.run_answer(result.raw_result)
            yield StageFinished(
                "answer",
                message=result.answer,
                details={"answer": result.answer, "tokens": result.tokens["p4"]},
            )
        else:
            result.pipeline_stages["code_execution"] = sweep.raw_status
            result.status = "rejected" if sweep.raw_status == "rejected" else "failed"
            result.error = sweep.error
        return Verdict.RETURN


class CodingStage(_Coding):
    """Up to three coder attempts on the selection; the result or a reason to rediscover."""

    def run_coding(
        self,
        attempt: Attempt,
        selection_record: dict[str, Any],
        plan_signature: str,
    ) -> Generator[Request | StageStarted, Answer, Verdict]:
        """Code the selection in up to ``MAX_CODE_ATTEMPTS`` attempts, each given the last one's code and error.

        Args:
            attempt: The round's place in the discovery budget.
            selection_record: The round's record.
            plan_signature: The round's plan signature, for the gate's memory.

        Yields:
            The attempts' signals and, in a gated run, the coder's rejections.

        Returns:
            What the engine does next.

        Raises:
            TypeError: When a coder rejection is answered with something other
                than Force or Rediscover.
        """
        run, state, result, experiment = (
            self.run,
            self.run.state,
            self.run.result,
            self.run.experiment,
        )
        previous_code = ""
        total_code_runs = 0
        force = False
        index = 0
        while index < MAX_CODE_ATTEMPTS:
            code_attempt = Attempt(index, MAX_CODE_ATTEMPTS)
            yield StageStarted("code", code_attempt.number)
            code_started = time.monotonic()
            generated = run.deps.write_and_run_code(
                run.question,
                state.selected,
                state.selected,
                state.solr_meta,
                state.reasoning,
                run.llm,
                run.prompt_manager,
                run.ctx.tables_dir,
                retries=code_attempt.index,
                error_msg=state.error,
                previous_code=previous_code,
                run_dir=run.run_dir,
                seed=run.reproducibility.effective_seed,
                seed_instruction_recorder=run.recorder.record_seed_instruction,
                coder_context_level=experiment.coder_context_level,
                evaluation_result_type=None,
                max_run_calls=max(1, MAX_CODE_ATTEMPTS - total_code_runs),
                selection_plan=dict(state.selection_state.selection_plan),
                source_field_names=run.benchmark.field_names,
                require_semantic_plan=experiment.require_semantic_plan,
                allow_semantic_revision=(experiment.revision_experiment_mode.value == "current"),
                brief_advisory=(experiment.coder_brief_mode.value == "advisory"),
                **({"force_execution": True} if force else {}),
                **run.hooks.placeholder_kwargs(cancel=True),
            )
            state.last_generated = generated
            error_msg = state.error
            outcome = code_outcome(
                generated, require_semantic_plan=experiment.require_semantic_plan
            )
            details = {
                "code_raw": generated.code_raw,
                "clean_code": generated.clean_code,
                "tokens": outcome.tokens,
                "error": generated.error,
                "error_msg": error_msg,
                "raw_result": outcome.raw_result,
            }
            run.recorder.code_attempt(
                outcome, attempt=code_attempt, elapsed=time.monotonic() - code_started
            )
            total_code_runs += outcome.coder_runs
            previous_code = outcome.code
            result.code = previous_code

            if outcome.status is CodeStatus.BLOCKED_BY_SELECTION:
                self.blocked(outcome, selection_record, plan_signature)
                yield StageFinished(
                    "code",
                    code_attempt.number,
                    "blocked_by_selection",
                    state.error,
                    details=details,
                )
                return self._after_break(attempt)
            if run.code_evaluation_enabled:
                attempts = result.code_evaluation["attempts"]
                attempts.extend(
                    run.attempt_evaluator.evaluate_generated_attempts(generated, len(attempts) + 1)
                )
                result.code_evaluation = run.attempt_evaluator.summarize(attempts)
                result.code_evaluation["generation_attempt_count"] = code_attempt.number
            if outcome.status is CodeStatus.TABLES_REJECTED:
                yield StageFinished(
                    "code", code_attempt.number, "tables_rejected", outcome.error, details=details
                )
                if run.ctx.surface.human_gated:
                    answer = yield CoderRejected(outcome.error)
                    if isinstance(answer, Force):
                        force = True  # the same attempt again, the coder told to run anyway
                        continue
                    if not isinstance(answer, Rediscover):
                        raise TypeError(
                            f"a coder rejection is answered with Force or Rediscover, not {answer!r}"
                        )
                self.rejected(outcome, selection_record, who="The code generator", level=None)
                return self._after_break(attempt)
            if outcome.status is CodeStatus.SUCCESS:
                verdict = yield from self._succeeded(outcome, generated, code_attempt, details)
                if verdict is not None:
                    return verdict
                index += 1
                continue
            more = self._failed(outcome, code_attempt)
            # The person sees the coder's own words; the structured repair
            # context stays with the next attempt.
            bare_error = str(
                getattr(generated, "error", "") or "Code execution returned no output."
            )
            yield StageFinished(
                "code", code_attempt.number, "failed", bare_error, more=more, details=details
            )
            if more:
                index += 1
                continue
            return self._after_break(attempt)
        return Verdict.STOP

    def _succeeded(
        self,
        outcome: CodeOutcome,
        generated: Any,
        code_attempt: Attempt,
        details: dict[str, Any],
    ) -> Generator[StageStarted | StageFinished, Answer, Verdict | None]:
        """The code ran: an empty result tries again (None), anything else is final."""
        run, state, result = self.run, self.run.state, self.run.result
        result.raw_result = outcome.raw_result
        raw_validation = validate_answer(outcome.raw_result)
        if raw_validation.disposition == AnswerDisposition.EMPTY:
            result.pipeline_stages["code_execution"] = "empty"
            state.error = raw_validation.reason
            yield StageFinished(
                "code",
                code_attempt.number,
                "empty",
                state.error,
                more=not code_attempt.last,
                details=details,
            )
            return None
        if raw_validation.disposition == AnswerDisposition.REJECTED:
            result.pipeline_stages["code_execution"] = "rejected"
            result.pipeline_stages["final_answer"] = "rejected"
            result.answer_disposition = raw_validation.disposition.value
            result.status = "rejected"
            result.error = raw_validation.reason
            yield StageFinished(
                "code", code_attempt.number, "rejected", state.error, details=details
            )
            return Verdict.RETURN
        if run.code_evaluation_enabled:
            result.code_evaluation = self.judge.adjudicate(result.code_evaluation, generated)
        yield StageFinished(
            "code", code_attempt.number, "succeeded", str(outcome.raw_result), details=details
        )
        yield StageStarted("answer")
        self.answer.run_answer(outcome.raw_result)
        yield StageFinished(
            "answer",
            message=result.answer,
            details={"answer": result.answer, "tokens": result.tokens["p4"]},
        )
        return Verdict.RETURN

    def _failed(self, outcome: CodeOutcome, code_attempt: Attempt) -> bool:
        """The code failed: record it; True when another attempt should follow.

        A turn that executed nothing tries again regardless; one that ran
        retries only while its failure is retryable and attempts remain.

        Args:
            outcome: The failed attempt's outcome.
            code_attempt: The attempt's place in the coding budget.

        Returns:
            Whether another attempt should follow.
        """
        state, result = self.run.state, self.run.result
        state.error = outcome.error
        error_record: dict[str, Any] = {
            "phase": "code",
            "type": "execution_error",
            "message": state.error,
        }
        if outcome.execution_error:
            error_record["details"] = outcome.execution_error
            error_record["type"] = outcome.execution_error.get("category", "execution_error")
        result.errors.append(error_record)
        result.pipeline_stages["code_execution"] = "failed"
        if not outcome.coder_runs:
            return True
        return outcome.retryable and not code_attempt.last

    def _after_break(self, attempt: Attempt) -> Verdict:
        """A coder execution failure is not evidence that discovery was wrong.

        The verified selection is kept; only a table rejection or an
        incomplete selection contract starts another discovery round.

        Args:
            attempt: The round's place in the discovery budget.

        Returns:
            STOP, or REDISCOVER after a rejection or a block when rounds remain.
        """
        if self.run.result.pipeline_stages["code_execution"] not in {
            "tables_rejected",
            "blocked_by_selection",
        }:
            return Verdict.STOP
        return Verdict.STOP if attempt.last else Verdict.REDISCOVER
