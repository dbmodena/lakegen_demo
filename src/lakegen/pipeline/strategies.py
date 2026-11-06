"""The discovery strategies: one per architecture, chosen by the config.

A ``DiscoveryStrategy`` runs one discovery round of its architecture on the
``DiscoveryStage`` that owns the run's state, and settles what the round
leaves behind (``after_round``: the architect's rejection). A strategy that
writes keywords first says so, and the stage writes and reviews them before
``discover`` runs.

``build(architecture)`` is the one place the config value turns into code; an
unregistered architecture fails with the registered ones listed. To add a
strategy: a value in ``DiscoveryArchitecture``, a class with the two members
and ``@register(architecture)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from lakegen.config.experiment import DiscoveryArchitecture
from lakegen.datastructures.outcomes import DiscoveryOutcome, SelectorKind
from lakegen.pipeline.run.budgets import Attempt
from lakegen.pipeline.wiring.adapters import (
    divided_agentic_outcome,
    unified_agentic_outcome,
)
from lakegen.retrieval.intent import intent_entities

if TYPE_CHECKING:
    from lakegen.pipeline.stages import DiscoveryStage

KeywordRound = tuple[Any, ...]
"""What ``write_keywords`` returns: keywords, raw text, tokens, reasoning."""


@runtime_checkable
class DiscoveryStrategy(Protocol):
    """One discovery round of an architecture.

    Attributes:
        architecture: The discovery architecture the strategy implements.
        writes_keywords_first: Whether the stage writes (and, gated, reviews)
            keywords before ``discover``.
    """

    architecture: DiscoveryArchitecture
    writes_keywords_first: bool

    def discover(
        self, stage: DiscoveryStage, keyword_round: KeywordRound | None
    ) -> DiscoveryOutcome:
        """Run the round's agents; ``keyword_round`` is the approved keywords when the strategy asked for them."""
        ...

    def after_round(
        self, stage: DiscoveryStage, outcome: DiscoveryOutcome, attempt: Attempt
    ) -> bool:
        """Settle the round on the state; returns whether it asked for new keywords."""
        ...


Factory = Callable[[], DiscoveryStrategy]
_REGISTRY: dict[DiscoveryArchitecture, Factory] = {}


def register(architecture: DiscoveryArchitecture) -> Callable[[Factory], Factory]:
    """A class decorator that registers the strategy of ``architecture`` for ``build``."""

    def mark(factory: Factory) -> Factory:
        _REGISTRY[DiscoveryArchitecture(architecture)] = factory
        return factory

    return mark


def registered_keys() -> tuple[DiscoveryArchitecture, ...]:
    """The registered architectures."""
    return tuple(_REGISTRY)


def build(architecture: DiscoveryArchitecture | str) -> DiscoveryStrategy:
    """The strategy of the architecture; an unregistered one names the registered ones."""
    try:
        factory = _REGISTRY[DiscoveryArchitecture(architecture)]
    except (KeyError, ValueError):
        known = ", ".join(sorted(item.value for item in _REGISTRY))
        raise ValueError(
            f"unsupported discovery architecture: {architecture}; registered: {known}"
        ) from None
    return factory()


# --- agentic: the agents search the lake with tools ---------------------------


class _Agentic:
    def after_round(
        self, stage: DiscoveryStage, outcome: DiscoveryOutcome, attempt: Attempt
    ) -> bool:
        return stage.apply_rejection(outcome, attempt) if outcome.rejected else False


@register(DiscoveryArchitecture.UNIFIED)
class UnifiedAgentic(_Agentic):
    """One agent searches, inspects and selects (``discover_tables``)."""

    architecture = DiscoveryArchitecture.UNIFIED
    writes_keywords_first = False

    def discover(
        self, stage: DiscoveryStage, keyword_round: KeywordRound | None = None
    ) -> DiscoveryOutcome:
        """Run the agent on a fresh state seeded with what earlier rounds learned.

        Args:
            stage: The discovery stage that owns the run's state.
            keyword_round: Unused: the agent writes its own searches.

        Returns:
            The round's outcome.
        """
        run, state, ctx = stage.run, stage.run.state, stage.run.ctx
        state.selection_state = stage.seeded_state()
        state.selection_states.append(state.selection_state)
        raw = run.deps.discover_tables(
            query=run.question,
            llm=run.llm,
            pm=run.prompt_manager,
            all_files=run.all_files,
            solr_client=run.solr,
            csv_dir=ctx.tables_dir,
            hint=state.hint,
            portal_name=ctx.portal_name,
            retrieval_config=ctx.retrieval,
            state=state.selection_state,
            lake=ctx.lake,
            **run.hooks.selector_kwargs(),
        )
        run.recorder.phase_invocation_counts["discovery"] += 1
        run.recorder.context_telemetry["llm_invocations"] += 1
        return unified_agentic_outcome(raw, state.selection_state)


@register(DiscoveryArchitecture.DIVIDED)
class DividedAgentic(_Agentic):
    """A keyword writer, then the MemGPT selector with tools over the retrieved candidates (``select_tables``)."""

    architecture = DiscoveryArchitecture.DIVIDED
    writes_keywords_first = True

    def discover(
        self, stage: DiscoveryStage, keyword_round: KeywordRound | None
    ) -> DiscoveryOutcome:
        """Run the selector over what the approved keywords retrieve.

        Args:
            stage: The discovery stage that owns the run's state.
            keyword_round: The approved keyword round.

        Returns:
            The round's outcome.

        Raises:
            RuntimeError: When the stage passed no keyword round.
        """
        if keyword_round is None:
            raise RuntimeError(
                "the divided strategy writes keywords first, but no keyword round was passed"
            )
        run, state, ctx, experiment = (
            stage.run,
            stage.run.state,
            stage.run.ctx,
            stage.run.experiment,
        )
        selection_round = run.deps.select_tables(
            query=run.question,
            llm=run.llm,
            pm=run.prompt_manager,
            all_files=run.all_files,
            keywords=keyword_round[0],
            solr_client=run.solr,
            csv_dir=ctx.tables_dir,
            hint=state.hint,
            portal_name=ctx.portal_name,
            lake=ctx.lake,
            retrieval_config=ctx.retrieval,
            selection_state=state.selection_state,
            entities=intent_entities(keyword_round[1]),
            carried_tables=state.carried_tables,
            carried_metadata=state.carried_metadata,
            excluded_tables=state.excluded_tables,
            discovery=experiment.discovery,
            **run.hooks.selector_kwargs(),
        )
        outcome = divided_agentic_outcome(keyword_round, selection_round, state.selection_state)
        selector_llm_calls = 0 if outcome.selector is SelectorKind.NONE else 1
        run.recorder.phase_invocation_counts["discovery"] += selector_llm_calls
        run.recorder.context_telemetry["llm_invocations"] += 1 + selector_llm_calls
        return outcome
