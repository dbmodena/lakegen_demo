"""Shared orchestration boundary for API, batch, CLI and UI adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from lakegen.config.experiment import ExperimentConfig, InteractionMode
from lakegen.pipeline.replay import Replay
from lakegen.pipeline.service import QueryResult, make_runtime_settings, run_question


@dataclass(frozen=True)
class ApprovalRequest:
    """A gate's question to the approval adapter.

    Attributes:
        gate: The gate's name.
        payload: What the gate shows.
    """

    gate: str
    payload: Mapping[str, Any]


class ApprovalAdapter(Protocol):
    """Answers the gates of a run."""

    def approve(self, request: ApprovalRequest) -> bool:
        """Whether the gate's request is approved."""
        ...


class AutonomousApprovalAdapter:
    """Approves every gate: the autonomous policy."""

    def approve(self, request: ApprovalRequest) -> bool:
        """Approve the request."""
        return True


class ExperimentRunner:
    """Own resolved configuration and execute the currently shared core.

    The non-interactive pipeline is fully adapted. Interactive surfaces retain
    their existing gate loops for now, but consume this same configuration and
    approval contract so they can migrate one gate at a time without drift.

    Attributes:
        config: The resolved experiment config.

    Args:
        config: The resolved experiment config.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(
        self,
        question: str,
        *,
        question_id: str | int | None = None,
        log_context: Mapping[str, Any] | None = None,
        approval_adapter: ApprovalAdapter | None = None,
        runtime_factory=make_runtime_settings,
        executor=run_question,
        replay: Replay | None = None,
    ) -> QueryResult:
        """Run one question autonomously under the runner's config.

        Args:
            question: The question.
            question_id: The question's id, if it has one.
            log_context: The benchmark fields of the question, if any.
            approval_adapter: Answers the gates; the autonomous policy when None.
            runtime_factory: Builds the runtime settings from the config.
            executor: Runs the question with them.
            replay: What an earlier run recorded that this run starts from.

        Returns:
            The run's result.

        Raises:
            ValueError: When the config's interaction mode is not autonomous.
            RuntimeError: When the approval adapter rejects the autonomous policy.
        """
        if self.config.interaction_mode != InteractionMode.AUTONOMOUS:
            raise ValueError(
                "ExperimentRunner currently supports execution only in autonomous mode; "
                "CLI and Chainlit preserve their existing human-gated adapters"
            )
        adapter = approval_adapter or AutonomousApprovalAdapter()
        # Resolve the policy without prompting. This also makes accidental input()
        # calls observable in tests through the adapter boundary.
        if not adapter.approve(ApprovalRequest("autonomous_policy", {})):
            raise RuntimeError("autonomous approval policy rejected the run")
        runtime = runtime_factory(
            core=self.config.core,
            model=self.config.model,
            use_unified_agent=self.config.use_unified_agent,
            retrieval_mode=self.config.retrieval.mode,
            top_k=self.config.retrieval.top_k,
            alpha=self.config.retrieval.alpha,
            candidate_multiplier=self.config.retrieval.candidate_multiplier,
            interaction_mode=self.config.interaction_mode,
            experiment_id=self.config.experiment_id,
            seed=self.config.seed,
            config=self.config,
        )
        return executor(
            question,
            runtime,
            question_id=question_id,
            log_context=log_context,
            **({"replay": replay} if replay is not None else {}),
        )
