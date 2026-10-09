"""Every non-LLM table selector is registered; the call sites choose through the registry."""

from __future__ import annotations

import pytest

from lakegen.config.experiment import DiscoveryConfig, TableSelector
from lakegen.workflow.discovery.selectors import (
    Selection,
    SelectionRequest,
    Selector,
    SelectorUnavailable,
    build,
    choose,
    registered_kinds,
)
from lakegen.workflow.discovery.selectors import (
    jev as jev_selectors,
)
from lakegen.workflow.discovery.selectors.jev_service import JevSelection, JevSelectorError


def test_every_selector_but_the_architectures_own_llm_is_registered():
    assert set(registered_kinds()) == set(TableSelector) - {TableSelector.LLM}
    with pytest.raises(ValueError, match="unsupported table selector: nope; registered: jev"):
        build("nope", DiscoveryConfig())


def test_choose_leaves_the_llm_selector_to_the_architecture():
    assert choose(None) is None
    assert choose(DiscoveryConfig(selector="llm")) is None
    selector = choose(DiscoveryConfig(selector="jev"))
    assert isinstance(selector, Selector) and selector.kind is TableSelector.JEV
    assert selector.title == "Jev table selector"


def _request() -> SelectionRequest:
    return SelectionRequest("q", ["a.parquet", "b.parquet"], {"a.parquet": {"title": "A"}}, None, DiscoveryConfig(selector="jev"))


def test_the_jev_selector_returns_a_selection_with_a_plan(monkeypatch):
    seen = {}

    def fake(query, candidates, metadata, table_dir, discovery, post=None):
        seen.update(query=query, candidates=candidates, metadata=metadata, table_dir=table_dir)
        return JevSelection(["a.parquet"], "a fits", "--- Jev Table Selector ---\ntrace", {"a.parquet": 0.9}, 12)

    monkeypatch.setattr(jev_selectors, "select_with_jev", fake)
    selection = build(TableSelector.JEV, DiscoveryConfig(selector="jev")).select(_request())
    assert seen == {"query": "q", "candidates": ["a.parquet", "b.parquet"], "metadata": {"a.parquet": {"title": "A"}}, "table_dir": None}
    assert isinstance(selection, Selection)
    assert (selection.selected, selection.reasoning, selection.tokens, selection.source) == (["a.parquet"], "a fits", 0, "jev_selector")
    assert set(selection.plan["table_roles"]) == {"a.parquet"} and selection.plan["recovered_from_existing_discovery_context"]
    assert selection.trace.startswith("--- Jev Table Selector ---")


def test_the_jev_selector_reports_a_failed_service_as_unavailable(monkeypatch):
    def failing(*args, **kwargs):
        raise JevSelectorError("HTTP 502")

    monkeypatch.setattr(jev_selectors, "select_with_jev", failing)
    with pytest.raises(SelectorUnavailable, match="HTTP 502"):
        build(TableSelector.JEV, DiscoveryConfig(selector="jev")).select(_request())


def test_a_rejection_carries_no_plan(monkeypatch):
    monkeypatch.setattr(
        jev_selectors, "select_with_jev",
        lambda *a, **k: JevSelection([], "REJECT_KEYWORDS: nothing fits", "--- Jev Table Selector ---\n", {}, 0),
    )
    selection = build(TableSelector.JEV, DiscoveryConfig(selector="jev")).select(_request())
    assert selection.selected == [] and selection.plan == {} and selection.advisories == ()
