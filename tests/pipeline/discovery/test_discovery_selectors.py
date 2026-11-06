"""Every running table selector is registered; the call sites choose through the registry."""

from __future__ import annotations

import pytest

from lakegen.config.experiment import DiscoveryConfig, TableSelector
from lakegen.pipeline.discovery.selectors import Selector, build, choose, registered_kinds


def test_every_selector_but_the_removed_ones_is_registered():
    assert set(registered_kinds()) == set(TableSelector) - {TableSelector.LLM, TableSelector.JEV}
    with pytest.raises(ValueError, match="unsupported table selector: nope; registered: memgpt"):
        build("nope", DiscoveryConfig())


def test_choose_builds_the_configured_selector_and_refuses_the_removed_ones():
    selector = choose(None)
    assert isinstance(selector, Selector) and selector.kind is TableSelector.MEMGPT
    assert selector.title == "MemGPT table selector"
    assert choose(DiscoveryConfig()).kind is TableSelector.MEMGPT
    with pytest.raises(ValueError, match="replaced on 2026-10-21"):
        choose(DiscoveryConfig(selector="llm"))
    with pytest.raises(ValueError, match="removed on 2026-10-29"):
        choose(DiscoveryConfig(selector="jev"))
