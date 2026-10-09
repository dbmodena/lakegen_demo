"""The discovery agent's budget is configuration, not hardcoded constants."""

import pandas as pd
import pytest

from lakegen.config.experiment import (
    DiscoveryConfig,
    ExperimentConfig,
    dotted_overrides,
)
from lakegen.discovery.tools import P12State, Phase12ToolsManager
from lakegen.discovery.tools import search as search_tools
from lakegen.retrieval import RetrievalConfig, RetrievalHit


def _hit(resource_id, rank):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id, "columns": []},
        score=1.0 / rank,
        rank=rank,
        lexical_rank=rank,
    )


def _manager(monkeypatch, tmp_path, *, discovery=None, top_k=10, tables=12):
    names = [f"t{index:02d}" for index in range(1, tables + 1)]
    for name in names:
        pd.DataFrame({"value": [1]}).to_parquet(tmp_path / f"{name}.parquet")

    class FakeService:
        def retrieve(self, **kwargs):
            return [_hit(name, index) for index, name in enumerate(names, 1)]

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    return Phase12ToolsManager(
        P12State(),
        object(),
        [f"{name}.parquet" for name in names],
        tmp_path,
        question="tables",
        retrieval_config=RetrievalConfig(top_k=top_k),
        discovery_config=discovery,
    )


def test_defaults_reproduce_the_previously_hardcoded_budget():
    config = DiscoveryConfig()
    assert (config.initial_candidates, config.expansion_size) == (10, 5)
    assert (config.max_expansions, config.max_search_attempts) == (1, 1)
    assert (config.initial_shortlist_size, config.max_inspected_candidates) == (3, 6)
    assert config.max_inspections_per_file == 2
    assert config.search_after_inspection is False


def test_visible_window_is_configurable(monkeypatch, tmp_path):
    manager = _manager(
        monkeypatch, tmp_path, discovery=DiscoveryConfig(initial_candidates=4), top_k=10
    )
    manager.search_tables("tables")

    assert manager.state.visible_candidate_count == 4
    # top_k still governs how many are retained, so the rest stay hidden.
    assert len(manager.state.all_candidates) == 10


def _hidden_pool(manager):
    return len(manager.state.all_candidates) - manager.state.visible_candidate_count


def test_a_narrower_window_gives_expansion_something_to_reveal(monkeypatch, tmp_path):
    """Expansion draws on retained-minus-visible, which the defaults leave empty."""
    strict = _manager(monkeypatch, tmp_path, discovery=DiscoveryConfig(), top_k=10)
    strict.search_tables("tables")
    assert _hidden_pool(strict) == 0
    strict.inspect_columns(strict.state.all_candidates[0])
    assert strict.expand_candidates("more") == "No additional candidates are available."

    loose = _manager(
        monkeypatch, tmp_path, discovery=DiscoveryConfig(initial_candidates=5), top_k=10
    )
    loose.search_tables("tables")
    assert _hidden_pool(loose) == 5
    loose.inspect_columns(loose.state.all_candidates[0])
    # There is now a pool to search, so expansion gets as far as matching against it
    # instead of refusing outright.
    assert loose.expand_candidates("more") != "No additional candidates are available."


def test_re_inspection_limit_comes_from_configuration(monkeypatch, tmp_path):
    manager = _manager(
        monkeypatch, tmp_path, discovery=DiscoveryConfig(max_inspections_per_file=3)
    )
    manager.search_tables("tables")
    target = manager.state.all_candidates[0]

    first = manager.inspect_columns(target)
    assert "Schema for" in first
    assert "attempt 2/3" in manager.inspect_columns(target)
    assert "attempt 3/3" in manager.inspect_columns(target)
    refused = manager.inspect_columns(target)
    assert "3 times" in refused and "Schema for" not in refused


def test_a_stricter_re_inspection_limit_refuses_sooner(monkeypatch, tmp_path):
    manager = _manager(
        monkeypatch, tmp_path, discovery=DiscoveryConfig(max_inspections_per_file=1)
    )
    manager.search_tables("tables")
    target = manager.state.all_candidates[0]

    assert "Schema for" in manager.inspect_columns(target)
    assert "1 times" in manager.inspect_columns(target)


def test_search_budget_is_configurable(monkeypatch, tmp_path):
    manager = _manager(
        monkeypatch, tmp_path, discovery=DiscoveryConfig(max_search_attempts=2)
    )
    assert "Attempt: 1" in manager.search_tables("first concept")
    assert "Attempt: 2" in manager.search_tables("second concept")
    assert "Search limit reached (2 attempt(s))" in manager.search_tables("third")


def test_inspecting_forecloses_further_search_unless_configured(monkeypatch, tmp_path):
    """Inspection is how the agent learns a candidate is wrong."""
    strict = _manager(
        monkeypatch, tmp_path, discovery=DiscoveryConfig(max_search_attempts=2)
    )
    strict.search_tables("first concept")
    strict.inspect_columns(strict.state.all_candidates[0])
    assert "Search refinement blocked" in strict.search_tables("second concept")

    loose = _manager(
        monkeypatch,
        tmp_path,
        discovery=DiscoveryConfig(max_search_attempts=2, search_after_inspection=True),
    )
    loose.search_tables("first concept")
    loose.inspect_columns(loose.state.all_candidates[0])
    assert "Attempt: 2" in loose.search_tables("second concept")


def test_expansion_budget_is_configurable(monkeypatch, tmp_path):
    manager = _manager(
        monkeypatch,
        tmp_path,
        discovery=DiscoveryConfig(initial_candidates=4, max_expansions=0),
    )
    manager.search_tables("tables")
    manager.inspect_columns(manager.state.all_candidates[0])
    assert "Expansion limit reached" in manager.expand_candidates("more")


def test_config_is_reachable_from_files_and_dotted_overrides():
    nested = dotted_overrides({
        "discovery.max_expansions": 2,
        "discovery.initial_candidates": 6,
    })
    assert nested == {"discovery": {"max_expansions": 2, "initial_candidates": 6}}

    config = ExperimentConfig(**nested)
    assert config.discovery.max_expansions == 2
    assert config.discovery.initial_candidates == 6
    # Survives the manifest round-trip that records what a run actually used.
    assert ExperimentConfig(**config.model_dump()).discovery == config.discovery


def test_an_inspection_budget_that_cannot_be_satisfied_is_rejected():
    with pytest.raises(ValueError, match="initial_shortlist_size"):
        DiscoveryConfig(initial_shortlist_size=7, max_inspected_candidates=6)


def test_unknown_discovery_keys_are_rejected():
    with pytest.raises(ValueError):
        DiscoveryConfig(max_serch_attempts=3)  # typo must not pass silently
