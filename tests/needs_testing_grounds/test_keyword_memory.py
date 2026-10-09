"""End-to-end functional checks for the keyword_memory_enabled augmentation
in Phase12ToolsManager._search -- no live Solr/LLM, fake retriever/client,
same pattern as test_discovery_config.py's `_manager` helper."""

import pandas as pd
import pytest

from lakegen.discovery import tools as tools_p12
from lakegen.discovery.tools import P12State, Phase12ToolsManager
from lakegen.config.experiment import DiscoveryConfig
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode


def _hit(resource_id, rank):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id, "columns": []},
        score=1.0 / rank,
        rank=rank,
        lexical_rank=rank,
    )


class FakeSolr:
    """Only `.select` is exercised (the global rarity report)."""

    def __init__(self):
        self.select_calls: list[tuple] = []

    def select(self, tokens, *, q_op="AND", rows=10, **_kw):
        self.select_calls.append((tuple(tokens), q_op, rows))
        return {"response": {"numFound": 3}}


def _manager(monkeypatch, tmp_path, *, keywords_to_hits, discovery=None, top_k=10, tables=12):
    """`keywords_to_hits(keywords) -> list[str] of resource_ids` decides what
    a given search call returns, so a test can script zero-hit vs. normal
    vs. truncated-at-cap responses per call."""
    names = [f"t{index:02d}" for index in range(1, tables + 1)]
    for name in names:
        pd.DataFrame({"value": [1]}).to_parquet(tmp_path / f"{name}.parquet")

    call_log: list[list[str]] = []

    class FakeService:
        def retrieve(self, **kwargs):
            kw = list(kwargs.get("keywords") or [])
            call_log.append(kw)
            found = keywords_to_hits(kw)
            return [_hit(name, index) for index, name in enumerate(found, 1)]

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    solr = FakeSolr()
    manager = Phase12ToolsManager(
        P12State(),
        solr,
        [f"{name}.parquet" for name in names],
        tmp_path,
        question="a question about tables",
        retrieval_config=RetrievalConfig(top_k=top_k, mode=RetrievalMode.KEYWORD),
        discovery_config=discovery or DiscoveryConfig(
            keyword_memory_enabled=True, max_search_attempts=10,
        ),
    )
    return manager, call_log, solr


def test_zero_hit_search_is_banned_and_blocks_a_superset(monkeypatch, tmp_path):
    manager, call_log, solr = _manager(
        monkeypatch, tmp_path, keywords_to_hits=lambda kw: [],
    )
    response = manager.search_tables("nonexistent stuff")
    assert "No tables found" in response
    assert "now blocked" in response
    assert "appears in 3" in response  # global rarity report used FakeSolr
    assert solr.select_calls  # rarity report actually queried Solr
    assert len(manager.state.zero_hit_bans) == 1
    # Pre-existing, unrelated behavior: a zero-hit AND search with >1 keyword
    # triggers one OR-fallback retry within the same _search() call.
    calls_after_first_search = len(call_log)
    assert calls_after_first_search == 2

    # A superset of the banned combo is blocked WITHOUT another retrieve() call.
    blocked = manager.search_tables("nonexistent stuff extra")
    assert "Search blocked" in blocked
    assert len(call_log) == calls_after_first_search  # no new retrieve() at all


def test_nonzero_hit_repeat_is_relaxed_not_hard_blocked(monkeypatch, tmp_path):
    manager, call_log, _solr = _manager(
        monkeypatch, tmp_path, keywords_to_hits=lambda kw: ["t01", "t02"],
    )
    first = manager.search_tables("things")
    assert "t01" in first
    assert len(call_log) == 1

    repeat = manager.search_tables("things")
    assert "not blocked" in repeat
    assert "Search blocked" not in repeat
    assert "t01" in repeat  # still shows what it returned
    assert len(call_log) == 1  # relaxed retry served from cache, no re-query


def test_truncated_search_gets_local_coverage_report(monkeypatch, tmp_path):
    # fetch_k = max(discovery.fetch_floor, retrieval_config.top_k); pin both
    # to 5 so "returns >= fetch_k hits" is easy to trigger deterministically.
    manager, _call_log, _solr = _manager(
        monkeypatch, tmp_path,
        keywords_to_hits=lambda kw: [f"t{i:02d}" for i in range(1, 6)],
        top_k=5,
        discovery=DiscoveryConfig(
            keyword_memory_enabled=True, max_search_attempts=10,
            initial_candidates=5, expansion_size=0, max_expansions=0,
        ),
    )
    response = manager.search_tables("things")
    assert "Local word-coverage check" in response


def test_flag_off_reproduces_todays_blanket_repeat_block(monkeypatch, tmp_path):
    manager, call_log, solr = _manager(
        monkeypatch, tmp_path, keywords_to_hits=lambda kw: [],
        discovery=DiscoveryConfig(keyword_memory_enabled=False, max_search_attempts=10),
    )
    response = manager.search_tables("nonexistent stuff")
    assert "No tables found" in response
    assert "now blocked" not in response
    assert not solr.select_calls  # no rarity lookup when the flag is off
    # Pre-existing, unrelated behavior: a zero-hit AND search with >1 keyword
    # triggers one OR-fallback retry within the same _search() call.
    calls_after_first_search = len(call_log)
    assert calls_after_first_search == 2

    repeat = manager.search_tables("nonexistent stuff")
    assert "Search skipped: identical concepts" in repeat
    assert len(call_log) == calls_after_first_search  # cache hit, no new retrieve()


def test_semantic_mode_is_inert_even_with_flag_on(monkeypatch, tmp_path):
    names = ["t01", "t02"]
    for name in names:
        pd.DataFrame({"value": [1]}).to_parquet(tmp_path / f"{name}.parquet")

    class FakeService:
        def retrieve(self, **kwargs):
            return []

    monkeypatch.setattr(
        tools_p12, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    solr = FakeSolr()
    manager = Phase12ToolsManager(
        P12State(), solr, [f"{n}.parquet" for n in names], tmp_path,
        question="a question", retrieval_config=RetrievalConfig(mode=RetrievalMode.SEMANTIC),
        discovery_config=DiscoveryConfig(keyword_memory_enabled=True, max_search_attempts=10),
    )
    response = manager.search_tables("anything")
    assert "now blocked" not in response  # no ban language leaks into SEMANTIC mode
    assert not solr.select_calls
    assert manager.state.zero_hit_bans == []
