"""Every architecture is a registered strategy; the stage runs through it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lakegen.config.experiment import DiscoveryArchitecture
from lakegen.datastructures.outcomes import DiscoveryOutcome, DiscoveryStatus
from lakegen.pipeline.run.budgets import Attempt
from lakegen.pipeline.strategies import (
    DiscoveryStrategy,
    DividedAgentic,
    UnifiedAgentic,
    build,
    registered_keys,
)


def test_every_architecture_is_registered_and_an_unknown_one_names_the_known():
    assert set(registered_keys()) == set(DiscoveryArchitecture)
    with pytest.raises(
        ValueError,
        match="unsupported discovery architecture: nope; registered: divided, unified",
    ):
        build("nope")


@pytest.mark.parametrize(
    ("architecture", "kind", "keywords_first"),
    [
        ("unified", UnifiedAgentic, False),
        ("divided", DividedAgentic, True),
    ],
)
def test_the_divided_strategy_writes_keywords_first(architecture, kind, keywords_first):
    strategy = build(architecture)
    assert isinstance(strategy, kind) and isinstance(strategy, DiscoveryStrategy)
    assert strategy.writes_keywords_first is keywords_first
    assert strategy.architecture == DiscoveryArchitecture(architecture)


def _outcome(status: DiscoveryStatus) -> DiscoveryOutcome:
    return DiscoveryOutcome(
        status=status,
        selected=[],
        candidates=[],
        keywords=[],
        metadata={},
        reasoning="",
        trace="",
        tokens=0,
    )


def _stage(rejection_continues: bool) -> SimpleNamespace:
    calls = []
    return SimpleNamespace(
        calls=calls,
        apply_rejection=lambda outcome, attempt: (
            calls.append(("apply_rejection", attempt.number)) or rejection_continues
        ),
    )


def test_an_agentic_strategy_applies_the_architects_rejection_only_when_it_rejected():
    strategy, attempt = build("divided"), Attempt(index=1, limit=3)
    stage = _stage(rejection_continues=True)
    assert strategy.after_round(stage, _outcome(DiscoveryStatus.SELECTED), attempt) is False
    assert stage.calls == []
    assert strategy.after_round(stage, _outcome(DiscoveryStatus.REJECT_KEYWORDS), attempt) is True
    assert stage.calls == [("apply_rejection", 2)]
    assert (
        build("unified").after_round(
            _stage(rejection_continues=False), _outcome(DiscoveryStatus.REJECT_KEYWORDS), attempt
        )
        is False
    )
