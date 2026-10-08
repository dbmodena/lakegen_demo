"""Every architecture and tool-access pair is a registered strategy; the stage runs through it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lakegen.domain.outcomes import DiscoveryOutcome, DiscoveryStatus
from lakegen.experiment_config import DiscoveryArchitecture, ToolAccess
from lakegen.pipeline.budgets import Attempt
from lakegen.pipeline.strategies import (
    DiscoveryStrategy,
    DividedAgentic,
    DividedOrchestrated,
    UnifiedAgentic,
    UnifiedOrchestrated,
    build,
    registered_keys,
)


def test_every_pair_is_registered_and_an_unknown_one_names_the_known():
    assert set(registered_keys()) == {(a, t) for a in DiscoveryArchitecture for t in ToolAccess}
    with pytest.raises(ValueError, match="unsupported discovery strategy: nope/agentic; registered: divided/agentic, divided/orchestrated_context"):
        build("nope", "agentic")


@pytest.mark.parametrize(
    ("architecture", "tool_access", "kind", "keywords_first"),
    [
        ("unified", "agentic", UnifiedAgentic, False),
        ("divided", "agentic", DividedAgentic, True),
        ("unified", "orchestrated_context", UnifiedOrchestrated, False),
        ("divided", "orchestrated_context", DividedOrchestrated, True),
    ],
)
def test_the_divided_strategies_write_keywords_first(architecture, tool_access, kind, keywords_first):
    strategy = build(architecture, tool_access)
    assert isinstance(strategy, kind) and isinstance(strategy, DiscoveryStrategy)
    assert strategy.writes_keywords_first is keywords_first
    assert (strategy.architecture, strategy.tool_access) == (DiscoveryArchitecture(architecture), ToolAccess(tool_access))


def _outcome(status: DiscoveryStatus) -> DiscoveryOutcome:
    return DiscoveryOutcome(status=status, selected=[], candidates=[], keywords=[], metadata={}, reasoning="", trace="", tokens=0)


def _stage(rejection_continues: bool) -> SimpleNamespace:
    calls = []
    return SimpleNamespace(
        calls=calls,
        apply_rejection=lambda outcome, attempt: calls.append(("apply_rejection", attempt.number)) or rejection_continues,
        orchestrated_round=lambda outcome, rejected: calls.append(("orchestrated_round", rejected)),
    )


def test_an_agentic_strategy_applies_the_architects_rejection_only_when_it_rejected():
    strategy, attempt = build("divided", "agentic"), Attempt(index=1, limit=3)
    stage = _stage(rejection_continues=True)
    assert strategy.after_round(stage, _outcome(DiscoveryStatus.SELECTED), attempt) is False
    assert stage.calls == []
    assert strategy.after_round(stage, _outcome(DiscoveryStatus.REJECT_KEYWORDS), attempt) is True
    assert stage.calls == [("apply_rejection", 2)]
    assert build("unified", "agentic").after_round(_stage(rejection_continues=False), _outcome(DiscoveryStatus.REJECT_KEYWORDS), attempt) is False


def test_an_orchestrated_strategy_records_every_round_and_reports_a_rejection():
    strategy, attempt = build("unified", "orchestrated_context"), Attempt(index=0, limit=3)
    stage = _stage(rejection_continues=True)
    assert strategy.after_round(stage, _outcome(DiscoveryStatus.SELECTED), attempt) is False
    assert strategy.after_round(stage, _outcome(DiscoveryStatus.REJECT_KEYWORDS), attempt) is True
    assert stage.calls == [("orchestrated_round", False), ("orchestrated_round", True)]
