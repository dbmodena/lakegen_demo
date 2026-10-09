"""A zero-result query is banned as the smallest word sets that already match
nothing, and what one round learned is still known in the next."""

import pytest

from lakegen.interfaces.chainlit.state import LakeGenSession
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode
from lakegen.tools import unified as tools_p12_v1
from lakegen.tools.shared.terms import keyword_terms, minimal_failing_subsets
from lakegen.tools.unified import search as search_tools

# agent_tools_2 was the experimental v2 module; its improvements were promoted
# into agent_tools/tools_p12.py and the duplicate package was removed, so only
# the promoted module is left to exercise here.
BOTH = pytest.mark.parametrize("module", [tools_p12_v1])


def _oracle(tables):
    """count(words) over a small universe of tables, each a set of words."""
    calls = []

    def count(words):
        calls.append(frozenset(words))
        return sum(all(word in table for word in words) for table in tables)

    return count, calls


# --- the algorithm ----------------------------------------------------------


def test_every_pair_that_never_co_occurs_is_found_and_no_superset_is_probed():
    tables = [{"belfast", "flood"}, {"lough", "marine"}, {"catchment", "river"}]
    count, calls = _oracle(tables)

    failing, alone = minimal_failing_subsets({"belfast", "lough", "catchment"}, count)

    assert set(failing) == {
        frozenset({"belfast", "lough"}),
        frozenset({"belfast", "catchment"}),
        frozenset({"lough", "catchment"}),
    }
    assert alone == {"belfast": 1, "catchment": 1, "lough": 1}
    assert max(len(words) for words in calls) == 2  # the triple was never asked


def test_a_single_culprit_pair_hides_every_bigger_query_containing_it():
    tables = [{"a", "c", "d"}, {"b", "c", "d"}, {"a", "d"}, {"b", "c"}]
    count, calls = _oracle(tables)

    failing, _ = minimal_failing_subsets({"a", "b", "c", "d"}, count)

    assert failing == [frozenset({"a", "b"})]
    assert not any(frozenset({"a", "b"}) < probed for probed in calls)


def test_a_word_that_matches_nothing_alone_is_the_whole_explanation():
    count, calls = _oracle([{"belfast", "lough", "flood"}, {"marine"}])

    failing, alone = minimal_failing_subsets({"belfast", "ncs", "lough"}, count)

    assert failing == [frozenset({"ncs"})]
    assert alone["ncs"] == 0
    assert all("ncs" not in probed or len(probed) == 1 for probed in calls)


def test_words_that_each_match_but_never_all_together_have_no_failing_subset():
    count, _ = _oracle([{"a", "b"}, {"b", "c"}, {"a", "c"}])

    failing, alone = minimal_failing_subsets({"a", "b", "c"}, count)

    assert failing == []
    assert alone == {"a": 2, "b": 2, "c": 2}


def test_a_single_word_has_nothing_to_shrink():
    count, calls = _oracle([])

    assert minimal_failing_subsets({"belfast"}, count) == ([], {})
    assert calls == []


def test_an_exhausted_probe_budget_still_returns_only_verified_sets():
    tables = [{"a", "x"}, {"b", "y"}, {"c", "z"}, {"d", "w"}]
    count, calls = _oracle(tables)

    failing, alone = minimal_failing_subsets({"a", "b", "c", "d"}, count, max_probes=6)

    assert len(calls) == 6
    assert set(alone) == {"a", "b", "c", "d"}
    assert all(count(sorted(subset)) == 0 for subset in failing)


# --- the manager bans what was learned --------------------------------------


def _hit(resource_id, rank=1):
    return RetrievalHit(
        document={"resource_id": resource_id, "title": resource_id, "columns": []},
        score=1.0 / rank, rank=rank, lexical_rank=rank,
    )


class _OracleService:
    """Retrieval and match counts over the same tiny universe of tables."""

    def __init__(self, tables, *, hybrid_semantic=None, broken=False):
        self.tables = tables
        self.hybrid_semantic = hybrid_semantic
        self.broken = broken
        self.retrieves, self.counted = [], []
        if hybrid_semantic is not None:
            self.last_lexical_hit_count = None

    def lexical_match_count(self, words):
        self.counted.append(frozenset(words))
        if self.broken:
            raise RuntimeError("solr unavailable")
        return sum(all(word in table for word in words) for table in self.tables)

    def retrieve(self, **kwargs):
        words = keyword_terms(kwargs["keywords"])
        self.retrieves.append(words)
        matched = [i for i, table in enumerate(self.tables) if words <= table]
        if self.hybrid_semantic is not None:
            self.last_lexical_hit_count = len(matched)
            return [_hit(self.hybrid_semantic)]
        return [_hit(f"table{i}", rank) for rank, i in enumerate(matched, 1)]


def _manager(module, monkeypatch, tmp_path, service, *, mode=RetrievalMode.KEYWORD,
             files=None, state=None):
    monkeypatch.setattr(search_tools, "get_table_retrieval_service", lambda *_a, **_k: service)
    state = state or module.P12State()
    files = files if files is not None else [f"table{i}.parquet" for i in range(len(service.tables))]
    return state, module.Phase12ToolsManager(
        state, object(), files, tmp_path, retrieval_config=RetrievalConfig(mode=mode),
    )


TABLES = [{"belfast", "flood"}, {"lough", "marine"}, {"catchment", "river"},
          {"network", "contribution", "score"}]


@BOTH
def test_a_zero_hit_query_is_banned_as_its_smallest_failing_sets(module, monkeypatch, tmp_path):
    service = _OracleService(TABLES)
    state, manager = _manager(module, monkeypatch, tmp_path, service)

    response = manager.search_keyword_concepts(["Belfast", "Lough", "catchment", "network"])

    assert "No tables found." in response
    assert "Smallest word sets that already match nothing: {belfast, catchment}, " in response
    assert "Tables containing each word alone: belfast 1, catchment 1, lough 1, network 1." in response
    assert "added to the zero-result banlist" in response
    assert len(state.failed_keyword_combinations) == 6
    assert all(len(banned) == 2 for banned in state.failed_keyword_combinations)
    assert max(len(words) for words in service.counted) == 2

    # Anything containing a culprit pair is now rejected without a retrieval...
    rejected = manager.search_keyword_concepts(["belfast", "network", "area"])
    assert rejected.startswith("Search rejected before retrieval")
    assert len(service.retrieves) == 1
    # ...while a query that avoids every culprit pair still runs.
    assert "table3.parquet" in manager.search_keyword_concepts(["network", "contribution"])


@BOTH
def test_words_that_match_alone_but_never_together_are_banned_whole(module, monkeypatch, tmp_path):
    service = _OracleService([{"a", "b"}, {"b", "c"}, {"a", "c"}])
    state, manager = _manager(module, monkeypatch, tmp_path, service)

    response = manager.search_keyword_concepts(["a", "b", "c"])

    assert "Each word matches tables alone (a 2, b 2, c 2)" in response
    assert state.failed_keyword_combinations == [frozenset({"a", "b", "c"})]


@BOTH
def test_no_probe_is_spent_when_solr_matched_tables_that_are_merely_not_local(
    module, monkeypatch, tmp_path
):
    service = _OracleService([{"a", "b"}])
    state, manager = _manager(module, monkeypatch, tmp_path, service, files=[])

    response = manager.search_keyword_concepts(["a", "b"])

    assert "No tables found. This AND keyword combination was added" in response
    assert service.counted == []
    assert state.failed_keyword_combinations == [frozenset({"a", "b"})]


@BOTH
def test_a_failing_probe_never_fails_the_search_and_bans_the_whole_query(
    module, monkeypatch, tmp_path
):
    service = _OracleService(TABLES, broken=True)
    state, manager = _manager(module, monkeypatch, tmp_path, service)

    response = manager.search_keyword_concepts(["belfast", "lough"])

    assert "No tables found. This AND keyword combination was added" in response
    assert state.failed_keyword_combinations == [frozenset({"belfast", "lough"})]


@BOTH
def test_hybrid_bans_the_smallest_sets_when_only_the_semantic_branch_found_tables(
    module, monkeypatch, tmp_path
):
    service = _OracleService(TABLES, hybrid_semantic="semantic")
    state, manager = _manager(
        module, monkeypatch, tmp_path, service,
        mode=RetrievalMode.HYBRID, files=["semantic.parquet"],
    )

    response = manager.search_keyword_concepts(["belfast", "lough", "catchment"])

    assert "No table contains all of these words." in response
    assert (
        "Smallest word sets that already match nothing: {belfast, catchment}, "
        "{belfast, lough}, {catchment, lough}."
    ) in response
    assert "candidates below do not match every word" in response
    assert "semantic.parquet" in response
    assert len(state.failed_keyword_combinations) == 3
    assert all(len(banned) == 2 for banned in state.failed_keyword_combinations)


@BOTH
def test_a_two_word_hybrid_query_whose_words_match_alone_bans_the_pair(
    module, monkeypatch, tmp_path
):
    service = _OracleService(TABLES, hybrid_semantic="semantic")
    state, manager = _manager(
        module, monkeypatch, tmp_path, service,
        mode=RetrievalMode.HYBRID, files=["semantic.parquet"],
    )

    response = manager.search_keyword_concepts(["belfast", "lough"])

    assert "Each word matches tables alone (belfast 1, lough 1)" in response
    assert "Zero-result banlist: [['belfast', 'lough']]" in response
    assert state.failed_keyword_combinations == [frozenset({"belfast", "lough"})]


@BOTH
def test_a_query_that_matched_tables_is_not_probed_or_banned(module, monkeypatch, tmp_path):
    service = _OracleService(TABLES)
    state, manager = _manager(module, monkeypatch, tmp_path, service)

    manager.search_keyword_concepts(["network", "contribution"])

    assert service.counted == [] and state.failed_keyword_combinations == []


# --- what one round learned is known in the next ----------------------------


def test_a_new_session_knows_no_bans_and_seeding_copies_rather_than_shares():
    session = LakeGenSession()
    assert session.failed_keyword_combinations == []

    first = tools_p12_v1.P12State()
    first.failed_keyword_combinations = [frozenset({"belfast", "lough"})]
    session.remember_keyword_bans(first)
    second = tools_p12_v1.P12State()
    session.seed_keyword_bans(second)
    second.failed_keyword_combinations.append(frozenset({"other"}))

    assert second.failed_keyword_combinations[0] == frozenset({"belfast", "lough"})
    assert session.failed_keyword_combinations == [frozenset({"belfast", "lough"})]


@BOTH
def test_a_later_round_rejects_a_set_an_earlier_round_proved_dead(module, monkeypatch, tmp_path):
    session = LakeGenSession()

    round_one = _OracleService(TABLES)
    state_one, manager_one = _manager(module, monkeypatch, tmp_path, round_one)
    manager_one.search_keyword_concepts(["belfast", "lough"])
    session.remember_keyword_bans(state_one)

    round_two = _OracleService(TABLES)
    state_two = module.P12State()
    session.seed_keyword_bans(state_two)
    _, manager_two = _manager(module, monkeypatch, tmp_path, round_two, state=state_two)
    response = manager_two.search_keyword_concepts(["Belfast", "Lough", "catchment"])

    assert response.startswith("Search rejected before retrieval")
    assert round_two.retrieves == []
