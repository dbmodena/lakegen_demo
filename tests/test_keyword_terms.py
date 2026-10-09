"""The zero-result keyword banlist must key on what Solr ANDs together, not on
how the agent happened to group, punctuate or order its concepts."""

import json

import pytest

from lakegen.config.experiment import DiscoveryConfig
from lakegen.discovery.memory import load_keyword_memory
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode
from lakegen.tools import unified as tools_p12_v1
from lakegen.tools.shared.terms import keyword_terms
from lakegen.tools.unified import search as search_tools

# agent_tools_2 was the experimental v2 module; its improvements were promoted
# into agent_tools/tools_p12.py and the duplicate package was removed, so only
# the promoted module is left to exercise here.


def _hit(resource_id, rank):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id, "columns": []},
        score=1.0 / rank,
        rank=rank,
        lexical_rank=rank,
    )


def test_keyword_terms_ignores_grouping_order_and_case():
    expected = frozenset({"catchment", "network", "contribution"})

    assert keyword_terms(["catchment", "network contribution"]) == expected
    assert keyword_terms(["Catchment network", "CONTRIBUTION"]) == expected
    assert keyword_terms(["network contribution catchment"]) == expected


@pytest.mark.parametrize(
    "variant",
    ["ash-free", "ash‑free", "ash‐free", "ash–free", "ash free", "Ash_Free"],
)
def test_keyword_terms_treats_delimiters_as_solr_does(variant):
    # WordDelimiterGraphFilter splits on every non-alphanumeric character, so a
    # non-breaking hyphen (U+2011) is not a different word from a plain space.
    assert keyword_terms([variant]) == frozenset({"ash", "free"})


def test_keyword_terms_strips_punctuation_accents_and_possessives():
    assert keyword_terms(["Berwick St. James"]) == keyword_terms(["Berwick St James"])
    assert keyword_terms(["Café"]) == frozenset({"cafe"})
    assert keyword_terms(["Smith’s", "Smith's"]) == frozenset({"smith"})
    assert keyword_terms(["O'Brien"]) == frozenset({"o", "brien"})


def test_keyword_terms_splits_on_case_and_digit_boundaries():
    assert keyword_terms(["PostCode"]) == frozenset({"post", "code"})
    assert keyword_terms(["LSOA11CD"]) == frozenset({"lsoa", "11", "cd"})
    assert keyword_terms(["December2019v2"]) == frozenset({"december", "2019", "v", "2"})
    # An all-caps acronym has no lower-to-upper transition to split on.
    assert keyword_terms(["XMLHttp"]) == frozenset({"xmlhttp"})


def test_keyword_terms_of_nothing_is_empty():
    assert keyword_terms([]) == frozenset()
    assert keyword_terms(["", "  ", "--", "."]) == frozenset()


@pytest.mark.parametrize("module", [tools_p12_v1])
def test_manager_bans_punctuation_variants_of_a_failed_query(
    module, monkeypatch, tmp_path
):
    class FakeService:
        def retrieve(self, **_kwargs):
            return []

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    state = module.P12State()
    manager = module.Phase12ToolsManager(
        state, object(), [], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    manager.search_keyword_concepts(["ash‑free dry mass"])
    blocked = manager.search_keyword_concepts(["ash-free", "dry mass", "coarse"])

    assert blocked.startswith("Search rejected before retrieval")
    assert "{ash, dry, free, mass}" in blocked


@pytest.mark.parametrize("module", [tools_p12_v1])
def test_zero_hit_and_query_is_banned_and_never_retried_as_or(
    module, monkeypatch, tmp_path
):
    """A strict AND that matches nothing is banned outright: no OR retry may
    rescue it, and any regrouping of the same terms is rejected."""
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append((list(kwargs["keywords"]), kwargs["q_op"]))
            return [_hit("table", 1)] if kwargs["q_op"] == "OR" else []

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    state = module.P12State()
    manager = module.Phase12ToolsManager(
        state, object(), ["table.parquet"], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    first = manager.search_keyword_concepts(["catchment", "network contribution"])

    assert "added to the zero-result banlist" in first
    assert calls == [(["catchment", "network", "contribution"], "AND")]
    assert state.failed_keyword_combinations == [
        frozenset({"catchment", "network", "contribution"})
    ]
    for regrouped in (
        ["catchment network", "contribution"],
        ["network contribution", "catchment", "area"],
        ["catchment network contribution"],
    ):
        assert manager.search_keyword_concepts(regrouped).startswith(
            "Search rejected before retrieval"
        )
    assert len(calls) == 1


def test_strict_and_hits_do_not_ban(monkeypatch, tmp_path):
    class FakeService:
        def retrieve(self, **_kwargs):
            return [_hit("table", 1)]

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    state = tools_p12_v1.P12State()
    manager = tools_p12_v1.Phase12ToolsManager(
        state, object(), ["table.parquet"], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    manager.search_keyword_concepts(["catchment", "network contribution"])

    assert state.failed_keyword_combinations == []


def test_persisted_bans_recorded_under_old_normalisation_still_apply(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"version": 1, "scopes": {"uk": {
        "keyword_history": [],
        "failed_keyword_combinations": [
            ["ash‑free", "dry"],
            ["ash", "dry", "free", "mass"],
        ],
    }}}), encoding="utf-8")

    _history, failed = load_keyword_memory(path, "uk")

    # Both spell the same constraint; the larger one is now redundant.
    assert failed == [frozenset({"ash", "free", "dry"})]


@pytest.mark.parametrize("module", [tools_p12_v1])
@pytest.mark.parametrize(
    "concepts",
    [
        ["hospital", "street"],
        ["Hospital Street"],
        ["hospital", "STREET", "street", "Hospital"],
    ],
)
def test_concepts_reach_retrieval_as_deduplicated_single_words(
    module, concepts, monkeypatch, tmp_path
):
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(list(kwargs["keywords"]))
            return []

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    state = module.P12State()
    manager = module.Phase12ToolsManager(
        state, object(), [], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
    )

    manager.search_keyword_concepts(concepts)

    assert [word.casefold() for word in calls[0]] == ["hospital", "street"]
    assert all(" " not in word for word in calls[0])
    assert state.keyword_history == [calls[0]]


@pytest.mark.parametrize("module", [tools_p12_v1])
def test_superset_of_a_failed_word_set_is_rejected_and_a_failing_subset_evicts_it(
    module, monkeypatch, tmp_path
):
    """The minimal-antichain contract, stated in the words the agent sees."""
    calls = []

    class FakeService:
        def retrieve(self, **kwargs):
            calls.append(list(kwargs["keywords"]))
            return []

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    state = module.P12State()
    manager = module.Phase12ToolsManager(
        state, object(), [], tmp_path,
        retrieval_config=RetrievalConfig(mode=RetrievalMode.KEYWORD),
        discovery_config=DiscoveryConfig(max_search_attempts=1, max_zero_result_retries=5),
    )

    manager.search_keyword_concepts(["hospital", "street"])
    assert [sorted(b) for b in state.failed_keyword_combinations] == [["hospital", "street"]]

    rejected = manager.search_keyword_concepts(["hospital", "street", "emergency"])
    assert "known zero-result keyword subset {hospital, street}" in rejected
    assert len(calls) == 1

    manager.search_keyword_concepts(["hospital"])
    assert [sorted(b) for b in state.failed_keyword_combinations] == [["hospital"]]


def _manager_with_lexical_count(module, monkeypatch, tmp_path, mode, lexical_count, calls):
    class FakeService:
        last_lexical_hit_count = lexical_count

        def retrieve(self, **kwargs):
            calls.append(list(kwargs["keywords"]))
            return [_hit("table", 1)]

    monkeypatch.setattr(
        search_tools, "get_table_retrieval_service", lambda *_a, **_k: FakeService()
    )
    state = module.P12State()
    manager = module.Phase12ToolsManager(
        state, object(), ["table.parquet"], tmp_path,
        retrieval_config=RetrievalConfig(mode=mode),
    )
    return state, manager


@pytest.mark.parametrize("module", [tools_p12_v1])
def test_hybrid_lexical_branch_matching_nothing_is_reported_and_banned(
    module, monkeypatch, tmp_path
):
    """Hybrid lists semantic candidates even when no table has every word; the
    agent must be told, and the word set must be banned like any zero-hit AND."""
    calls = []
    state, manager = _manager_with_lexical_count(
        module, monkeypatch, tmp_path, RetrievalMode.HYBRID, 0, calls
    )

    response = manager.search_keyword_concepts(["Belfast Lough", "grid"])

    assert "No table contains all of these words" in response
    assert "candidates below do not match every word" in response
    assert "Zero-result banlist: [['belfast', 'grid', 'lough']]" in response
    assert "table.parquet" in response
    assert state.failed_keyword_combinations == [frozenset({"belfast", "lough", "grid"})]
    assert manager.search_keyword_concepts(["belfast", "lough", "grid", "area"]).startswith(
        "Search rejected before retrieval"
    )
    assert len(calls) == 1


@pytest.mark.parametrize("module", [tools_p12_v1])
@pytest.mark.parametrize(
    ("mode", "lexical_count"),
    [
        (RetrievalMode.HYBRID, 3),       # the lexical branch matched tables
        (RetrievalMode.HYBRID, None),    # the lexical branch did not run
        (RetrievalMode.KEYWORD, 0),      # only hybrid reports a branch count
    ],
)
def test_lexical_notice_and_ban_need_a_hybrid_branch_that_matched_nothing(
    module, mode, lexical_count, monkeypatch, tmp_path
):
    state, manager = _manager_with_lexical_count(
        module, monkeypatch, tmp_path, mode, lexical_count, []
    )

    response = manager.search_keyword_concepts(["hospital", "street"])

    assert "No table contains all of these words" not in response
    assert response.startswith("Attempt: 1\nSearched: concepts ['hospital', 'street']\n\nCandidates")
    assert state.failed_keyword_combinations == []
