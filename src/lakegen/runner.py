"""Shared orchestration boundary for API, batch, CLI and UI adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from lakegen.experiment_config import ExperimentConfig, InteractionMode
from lakegen.service import QueryResult, make_runtime_settings, run_question


@dataclass(frozen=True)
class ApprovalRequest:
    gate: str
    payload: Mapping[str, Any]


class ApprovalAdapter(Protocol):
    def approve(self, request: ApprovalRequest) -> bool: ...


class AutonomousApprovalAdapter:
    def approve(self, request: ApprovalRequest) -> bool:
        return True


class ExperimentRunner:
    """Own resolved configuration and execute the currently shared core.

    The non-interactive pipeline is fully adapted. Interactive surfaces retain
    their existing gate loops for now, but consume this same configuration and
    approval contract so they can migrate one gate at a time without drift.
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
    ) -> QueryResult:
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
        )
