"""The discovery strategies: one per architecture and tool access, chosen by the config.

A ``DiscoveryStrategy`` runs one discovery round of its architecture on the
``DiscoveryStage`` that owns the run's state, and settles what the round
leaves behind (``after_round``: the architect's rejection in the agentic
strategies, the orchestrator's telemetry and the empty-context retry in the
orchestrated ones). A strategy that writes keywords first says so, and the
stage writes and reviews them before ``discover`` runs.

``build(architecture, tool_access)`` is the one place the two config values
turn into code; an unregistered pair fails with the registered ones listed.
To add a strategy: a value in ``DiscoveryArchitecture`` or ``ToolAccess``, a
class with the three members and ``@register(architecture, tool_access)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from lakegen.config.experiment import DiscoveryArchitecture, ToolAccess
from lakegen.domain.outcomes import DiscoveryOutcome, SelectorKind
from lakegen.retrieval.intent import intent_entities
from lakegen.workflow.discovery.memory import (
    format_question_retrieval_memory,
    load_question_retrieval_memory,
    persist_keyword_memory,
    persist_question_retrieval_memory,
)
from lakegen.workflow.discovery.orchestrated import (
    OrchestratedContextPreparationError,
    OrchestratedSelectorError,
    RetrievalRequestProtocolError,
)
from lakegen.workflow.pipeline.model.budgets import Attempt
from lakegen.workflow.pipeline.wiring.adapters import (
    divided_agentic_outcome,
    divided_orchestrated_outcome,
    unified_agentic_outcome,
    unified_orchestrated_outcome,
)

if TYPE_CHECKING:
    from lakegen.workflow.pipeline.stages import DiscoveryStage

KeywordRound = tuple[Any, ...]
"""What ``phase1_generate_keywords`` returns: keywords, raw text, tokens, reasoning."""


@runtime_checkable
class DiscoveryStrategy(Protocol):
    architecture: DiscoveryArchitecture
    tool_access: ToolAccess
    writes_keywords_first: bool
    """Whether the stage writes (and, gated, reviews) keywords before ``discover``."""

    def discover(self, stage: DiscoveryStage, keyword_round: KeywordRound | None) -> DiscoveryOutcome:
        """Run the round's agents; ``keyword_round`` is the approved keywords when the strategy asked for them."""
        ...

    def after_round(self, stage: DiscoveryStage, outcome: DiscoveryOutcome, attempt: Attempt) -> bool:
        """Settle the round on the state; returns whether it asked for new keywords."""
        ...


Key = tuple[DiscoveryArchitecture, ToolAccess]
Factory = Callable[[], DiscoveryStrategy]
_REGISTRY: dict[Key, Factory] = {}


def register(architecture: DiscoveryArchitecture, tool_access: ToolAccess) -> Callable[[Factory], Factory]:
    def mark(factory: Factory) -> Factory:
        _REGISTRY[(DiscoveryArchitecture(architecture), ToolAccess(tool_access))] = factory
        return factory

    return mark


def registered_keys() -> tuple[Key, ...]:
    return tuple(_REGISTRY)


def build(architecture: DiscoveryArchitecture | str, tool_access: ToolAccess | str) -> DiscoveryStrategy:
    """The strategy of the pair; an unregistered pair names the registered ones."""
    try:
        factory = _REGISTRY[(DiscoveryArchitecture(architecture), ToolAccess(tool_access))]
    except (KeyError, ValueError):
        known = ", ".join(sorted(f"{a.value}/{t.value}" for a, t in _REGISTRY))
        raise ValueError(
            f"unsupported discovery strategy: {architecture}/{tool_access}; registered: {known}"
        ) from None
    return factory()


# --- agentic: the agents search the lake with tools ---------------------------

class _Agentic:
    tool_access = ToolAccess.AGENTIC

    def after_round(self, stage: DiscoveryStage, outcome: DiscoveryOutcome, attempt: Attempt) -> bool:
        return stage.apply_rejection(outcome, attempt) if outcome.rejected else False


@register(DiscoveryArchitecture.UNIFIED, ToolAccess.AGENTIC)
class UnifiedAgentic(_Agentic):
    """One agent searches, inspects and selects (``phase12_agent``)."""

    architecture = DiscoveryArchitecture.UNIFIED
    writes_keywords_first = False

    def discover(self, stage: DiscoveryStage, keyword_round: KeywordRound | None = None) -> DiscoveryOutcome:
        run, state, ctx, experiment = stage.run, stage.run.state, stage.run.ctx, stage.run.experiment
        state.selection_state = stage.seeded_state()
        state.selection_states.append(state.selection_state)
        raw = run.deps.phase12_agent(
            query=run.question,
            llm=run.llm,
            pm=run.prompt_manager,
            all_files=run.all_files,
            solr_client=run.solr,
            csv_dir=ctx.tables_dir,
            hint=state.hint,
            retrieval_memory=state.retrieval_memory_summary,
            portal_name=ctx.portal_name,
            retrieval_config=ctx.retrieval,
            state=state.selection_state,
            retrieval_observer=run.retrieval_observer,
            require_semantic_plan=experiment.require_semantic_plan,
            discovery_config=experiment.discovery,
            **run.hooks.agent_kwargs(),
        )
        stage.remember_agent_round(state.selection_state)
        run.recorder.phase_invocation_counts["discovery"] += 1
        run.recorder.context_telemetry["llm_invocations"] += 1
        return unified_agentic_outcome(raw, state.selection_state)


@register(DiscoveryArchitecture.DIVIDED, ToolAccess.AGENTIC)
class DividedAgentic(_Agentic):
    """A keyword writer, then a judge with tools over the retrieved candidates (``phase2_select_tables``)."""

    architecture = DiscoveryArchitecture.DIVIDED
    writes_keywords_first = True

    def discover(self, stage: DiscoveryStage, keyword_round: KeywordRound | None) -> DiscoveryOutcome:
        assert keyword_round is not None
        run, state, ctx, experiment = stage.run, stage.run.state, stage.run.ctx, stage.run.experiment
        selection_round = run.deps.phase2_select_tables(
            query=run.question, llm=run.llm, pm=run.prompt_manager,
            all_files=run.all_files, keywords=keyword_round[0], solr_client=run.solr,
            csv_dir=ctx.tables_dir, hint=state.hint,
            portal_name=ctx.portal_name,
            retrieval_config=ctx.retrieval,
            selection_state=state.selection_state,
            entities=intent_entities(keyword_round[1]),
            carried_tables=state.carried_tables,
            carried_metadata=state.carried_metadata,
            excluded_tables=state.excluded_tables,
            discovery=experiment.discovery,
            **run.hooks.agent_kwargs(),
        )
        outcome = divided_agentic_outcome(keyword_round, selection_round, state.selection_state)
        selector_llm_calls = 0 if outcome.selector is SelectorKind.JEV else 1
        run.recorder.phase_invocation_counts["discovery"] += selector_llm_calls
        run.recorder.context_telemetry["llm_invocations"] += 1 + selector_llm_calls
        return outcome


# --- orchestrated: the orchestrator retrieves, the agents only read ------------

class _Orchestrated:
    tool_access = ToolAccess.ORCHESTRATED_CONTEXT

    def after_round(self, stage: DiscoveryStage, outcome: DiscoveryOutcome, attempt: Attempt) -> bool:
        stage.orchestrated_round(outcome, outcome.rejected)
        return outcome.rejected


@register(DiscoveryArchitecture.UNIFIED, ToolAccess.ORCHESTRATED_CONTEXT)
class UnifiedOrchestrated(_Orchestrated):
    """One agent asks the orchestrator for context and selects from it (``run_unified_orchestrated_discovery``)."""

    architecture = DiscoveryArchitecture.UNIFIED
    writes_keywords_first = False

    def discover(self, stage: DiscoveryStage, keyword_round: KeywordRound | None = None) -> DiscoveryOutcome:
        run, state, ctx, experiment = stage.run, stage.run.state, stage.run.ctx, stage.run.experiment
        telemetry, counts = run.recorder.context_telemetry, run.recorder.phase_invocation_counts
        memory = experiment.orchestrated_memory_enabled
        try:
            discovery_result = run.deps.run_unified_orchestrated_discovery(
                query=run.question, llm=run.llm, solr_client=run.solr,
                all_files=run.all_files, retrieval_config=ctx.retrieval,
                table_dir=ctx.tables_dir,
                portal_name=ctx.portal_name,
                hint=state.hint,
                retrieval_memory=state.retrieval_memory_summary if memory else "",
                failed_keyword_combinations=state.failed_keyword_combinations if memory else [],
                max_zero_result_retries=experiment.discovery.max_zero_result_retries if memory else 0,
                memory_enabled=memory,
                **run.hooks.agent_kwargs(),
            )
        except RetrievalRequestProtocolError as exc:
            telemetry["request_protocol_error"] = f"{type(exc).__name__}: {exc}"
            telemetry["llm_invocations"] += 1
            counts["discovery"] += 1
            raise
        except OrchestratedContextPreparationError as exc:
            telemetry["preparation_error"] = f"{type(exc).__name__}: {exc}"
            telemetry["llm_invocations"] += 1
            counts["discovery"] += 1
            stage.count_orchestrator_call()
            raise
        except OrchestratedSelectorError as exc:
            telemetry["selector_error"] = f"{type(exc).__name__}: {exc}"
            telemetry["llm_invocations"] += 2
            counts["discovery"] += 2
            stage.count_orchestrator_call()
            raise
        outcome = unified_orchestrated_outcome(discovery_result)
        counts["discovery"] += discovery_result.llm_invocations
        telemetry["llm_invocations"] += discovery_result.llm_invocations
        if memory and state.keyword_memory_path is not None:
            self._remember(stage, discovery_result)
        return outcome

    @staticmethod
    def _remember(stage: DiscoveryStage, discovery_result: Any) -> None:
        """The word sets the agent proved dead and its retrieval evidence go to the lake's memory file."""
        state = stage.run.state
        returned_bans = [frozenset(item) for item in discovery_result.failed_keyword_combinations if item]
        if returned_bans:
            state.failed_keyword_combinations = returned_bans
        persist_keyword_memory(
            state.keyword_memory_path, state.keyword_memory_scope, [], state.failed_keyword_combinations,
        )
        persist_question_retrieval_memory(
            state.keyword_memory_path, state.keyword_memory_scope, state.retrieval_question_key,
            discovery_result.retrieval_memory_events,
        )
        state.retrieval_memory_summary = format_question_retrieval_memory(
            load_question_retrieval_memory(
                state.keyword_memory_path, state.keyword_memory_scope, state.retrieval_question_key,
            )
        )


@register(DiscoveryArchitecture.DIVIDED, ToolAccess.ORCHESTRATED_CONTEXT)
class DividedOrchestrated(_Orchestrated):
    """A keyword writer, the orchestrator's prepared context, a tool-free selector over it."""

    architecture = DiscoveryArchitecture.DIVIDED
    writes_keywords_first = True

    def discover(self, stage: DiscoveryStage, keyword_round: KeywordRound | None) -> DiscoveryOutcome:
        assert keyword_round is not None
        run, state, ctx, experiment = stage.run, stage.run.state, stage.run.ctx, stage.run.experiment
        telemetry, counts = run.recorder.context_telemetry, run.recorder.phase_invocation_counts
        try:
            prepared, prepared_metadata = run.deps.prepare_discovery_context(
                query=run.question, keywords=keyword_round[0], solr_client=run.solr,
                all_files=run.all_files, retrieval_config=ctx.retrieval,
                table_dir=ctx.tables_dir,
                entities=intent_entities(keyword_round[1]),
            )
        except Exception as preparation_error:
            exc = OrchestratedContextPreparationError(str(preparation_error))
            telemetry["preparation_error"] = f"{type(exc).__name__}: {exc}"
            telemetry["llm_invocations"] += 1
            stage.count_orchestrator_call()
            raise exc from preparation_error
        selection = None
        if prepared.candidates:
            try:
                selection = run.deps.select_from_prepared_context(
                    query=run.question, llm=run.llm, context=prepared,
                    all_files=run.all_files, architecture=experiment.discovery_architecture,
                    hint=state.hint, discovery=experiment.discovery,
                    table_dir=ctx.tables_dir,
                    **run.hooks.agent_kwargs(),
                )
            except OrchestratedSelectorError as exc:
                telemetry["selector_error"] = f"{type(exc).__name__}: {exc}"
                telemetry["llm_invocations"] += 2
                counts["discovery"] += 1
                raise
        outcome = divided_orchestrated_outcome(keyword_round, prepared, prepared_metadata, selection)
        selector_llm_calls = 0 if outcome.selector in (SelectorKind.JEV, SelectorKind.NONE) else 1
        counts["discovery"] += selector_llm_calls
        telemetry["llm_invocations"] += 1 + selector_llm_calls
        return outcome
