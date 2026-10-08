"""keyword_search="cascade" (DiscoveryConfig): for each per-table request the
model ranks keyword_cascade_size AND word sets by importance. They run one at a
time: a set containing a banned subset is skipped unsearched, a set that finds
nothing is banned, and the first set that finds a local table ends the cascade.
Hybrid mode skips the split entirely and runs one fused search."""

import json
from types import SimpleNamespace

from lakegen.agent_tools import tools_p12
from lakegen.experiment_config import DiscoveryConfig, KeywordSearchStrategy
from lakegen.retrieval import RetrievalConfig, RetrievalHit, RetrievalMode


TABLE_WORDS = {
    "orr_2021_11.parquet": {"organogram", "office", "rail", "road", "november", "2021"},
    "tf_2022_01.parquet": {"organogram", "transport", "focus", "january", "2022"},
}
QUESTION = (
    "What is the total junior FTE across the Office of Rail and Road November 2021 "
    "and Transport Focus January 2022 organograms?"
)


class _FakeRetrievalService:
    """keywords -> hits by exact-word-subset containment, like the real AND search."""

    def __init__(self):
        self.retrieves: list[list[str]] = []

    def retrieve(self, **kwargs):
        words = {str(w).casefold() for w in kwargs["keywords"]}
        self.retrieves.append(sorted(words))
        matched = [name for name, table_words in TABLE_WORDS.items() if words <= table_words]
        return [
            RetrievalHit(
                document={"resource_id": name, "title": name, "columns": []},
                score=1.0 / rank, rank=rank, lexical_rank=rank,
            )
            for rank, name in enumerate(matched, 1)
        ]


class _FakeLLM:
    """Answers the decompose prompt with ``requests`` and each cascade prompt
    with the cascade whose key occurs in that call's request. Keyed on content,
    not call order, since cascades for several requests are proposed in parallel."""

    def __init__(self, requests: list[str], cascades: dict[str, list[list[str]]]):
        self.requests = requests
        self.cascades = cascades
        self.calls: list[str] = []
        self.users: list[str] = []

    def chat(self, messages):
        system, user = messages[0].content, messages[1].content
        self.calls.append(system)
        self.users.append(user)
        if "distinct tables the question needs" in system:
            content = json.dumps({"tables": [{"detail": r, "request": r} for r in self.requests]})
        elif "You write keyword searches" in system:
            request = user.split("\n", 1)[0]
            searches = next(
                (value for key, value in self.cascades.items() if key in request), []
            )
            content = json.dumps({"searches": searches})
        else:
            raise AssertionError(f"unexpected prompt: {system[:60]}")
        return SimpleNamespace(message=SimpleNamespace(content=content))


class _CountingRetrievalService(_FakeRetrievalService):
    """Also answers the per-subset match counts the ban shrinking probes."""

    def lexical_match_count(self, keywords):
        words = {str(w).casefold() for w in keywords}
        return sum(words <= table_words for table_words in TABLE_WORDS.values())


def _manager(
    monkeypatch, tmp_path, *, llm, discovery=None, mode=RetrievalMode.KEYWORD,
    service=None,
):
    service = service or _FakeRetrievalService()
    monkeypatch.setattr(tools_p12, "get_table_retrieval_service", lambda *_a, **_k: service)
    notices: list[str] = []
    state = tools_p12.P12State()
    manager = tools_p12.Phase12ToolsManager(
        state, None, list(TABLE_WORDS), tmp_path,
        question=QUESTION,
        retrieval_config=RetrievalConfig(mode=mode),
            discovery_config=discovery or DiscoveryConfig(),
        notice_callback=notices.append,
        llm=llm,
    )
    return state, manager, service, notices


def test_cascade_of_five_is_the_default():
    config = DiscoveryConfig()
    assert config.keyword_search == KeywordSearchStrategy.CASCADE
    assert config.keyword_cascade_size == 5


def test_cascade_stops_at_the_first_step_that_finds_a_table(monkeypatch, tmp_path):
    llm = _FakeLLM(["organogram Office of Rail and Road November 2021"], {"Rail": [
        ["organogram", "rail", "road", "nov"],
        ["organogram", "rail", "road"],
        ["organogram"],
    ]})
    state, manager, service, notices = _manager(monkeypatch, tmp_path, llm=llm)

    response = manager.search_keyword_concepts(["organogram", "rail"])

    assert service.retrieves == [
        ["nov", "organogram", "rail", "road"],
        ["organogram", "rail", "road"],
    ]  # step 3 never runs
    assert "orr_2021_11.parquet" in response
    assert "tf_2022_01.parquet" not in response
    assert frozenset({"nov", "organogram", "rail", "road"}) in state.failed_keyword_combinations
    assert any("Keyword cascade, table 1" in notice for notice in notices)


def test_each_table_request_runs_its_own_cascade(monkeypatch, tmp_path):
    llm = _FakeLLM(
        ["organogram Office of Rail and Road November 2021",
         "organogram Transport Focus January 2022"],
        {
            "Rail": [["organogram", "rail", "november"]],
            "Focus": [["organogram", "focus", "jan"], ["organogram", "focus"]],
        },
    )
    state, manager, service, _ = _manager(monkeypatch, tmp_path, llm=llm)

    response = manager.search_keyword_concepts(["organogram"])

    assert service.retrieves == [
        ["november", "organogram", "rail"],
        ["focus", "jan", "organogram"],
        ["focus", "organogram"],
    ]
    assert "orr_2021_11.parquet" in response
    assert "tf_2022_01.parquet" in response
    assert "Searched (split by table): table 1:" in response


def test_a_set_another_table_already_ran_is_not_searched_twice(monkeypatch, tmp_path):
    llm = _FakeLLM(
        ["organogram rail 2021", "organogram rail 2020"],
        {
            "2021": [["organogram", "office"]],
            "2020": [["office", "organogram"], ["organogram", "rail"]],
        },
    )
    state, manager, service, notices = _manager(monkeypatch, tmp_path, llm=llm)

    manager.search_keyword_concepts(["organogram"])

    # Table 2's first set is table 1's, reordered: reused, and it found a table.
    assert service.retrieves == [["office", "organogram"]]
    assert any("already searched" in notice for notice in notices)


def test_a_step_containing_a_banned_subset_is_skipped_without_retrieval(monkeypatch, tmp_path):
    llm = _FakeLLM(["organogram rail"], {"rail": [
        ["organogram", "railway"],
        ["organogram", "rail"],
    ]})
    state, manager, service, notices = _manager(monkeypatch, tmp_path, llm=llm)
    state.failed_keyword_combinations = [frozenset({"railway"})]

    manager.search_keyword_concepts(["organogram"])

    assert service.retrieves == [["organogram", "rail"]]
    assert any("skipped, contains banned {railway}" in notice for notice in notices)


def test_all_steps_empty_reports_no_tables_and_bans_each(monkeypatch, tmp_path):
    llm = _FakeLLM(["budget"], {"budget": [["budget", "rail"], ["budget"]]})
    state, manager, service, _ = _manager(monkeypatch, tmp_path, llm=llm)

    response = manager.search_keyword_concepts(["budget"])

    assert len(service.retrieves) == 2
    assert "No tables found for any of the split searches" in response
    assert frozenset({"budget"}) in state.failed_keyword_combinations


def test_a_step_finding_only_banned_tables_is_empty_and_the_cascade_goes_on(
    monkeypatch, tmp_path
):
    llm = _FakeLLM(["organogram"], {"organogram": [["organogram", "rail"], ["organogram"]]})
    state, manager, service, notices = _manager(monkeypatch, tmp_path, llm=llm)
    state.excluded_tables = {"orr_2021_11.parquet"}

    response = manager.search_keyword_concepts(["organogram"])

    assert service.retrieves == [["organogram", "rail"], ["organogram"]]
    assert "tf_2022_01.parquet" in response
    assert "orr_2021_11.parquet" not in response
    assert state.banned_table_keyword_combinations == [frozenset({"organogram", "rail"})]
    assert state.failed_keyword_combinations == []
    assert any("only banned tables (1)" in notice for notice in notices)


def test_a_cascade_finding_only_banned_tables_nudges_for_new_keywords(monkeypatch, tmp_path):
    llm = _FakeLLM(["rail"], {"rail": [["organogram", "rail"], ["rail"]]})
    state, manager, service, _ = _manager(monkeypatch, tmp_path, llm=llm)
    state.excluded_tables = {"orr_2021_11.parquet"}

    response = manager.search_keyword_concepts(["rail"])

    assert "No usable tables" in response
    assert "{organogram, rail}, {rail} are now banned for this question" in response
    assert "No tables found for any of the split searches" not in response
    # {rail} subsumes {organogram, rail}; a later search containing it is refused.
    assert state.banned_table_keyword_combinations == [frozenset({"rail"})]
    assert "found only tables already banned" in manager.search_keyword_concepts(
        ["rail", "road"]
    )
    assert len(service.retrieves) == 2


def test_proposals_drop_aggregation_words_duplicates_and_extras(monkeypatch, tmp_path):
    llm = _FakeLLM([], {"x": [
        ["Total", "Organogram"],
        ["organogram"],  # same terms as the first once "total" is dropped
        ["rail"], ["road"], ["office"], ["2021"], ["november"],
    ]})
    discovery = DiscoveryConfig(keyword_cascade_size=3)
    _, manager, _, _ = _manager(monkeypatch, tmp_path, llm=llm, discovery=discovery)

    assert manager._propose_keyword_cascade("x", []) == [["organogram"], ["rail"], ["road"]]


def test_no_usable_proposal_falls_back_to_the_plain_search(monkeypatch, tmp_path):
    llm = _FakeLLM(["organogram"], {})
    _, manager, service, _ = _manager(monkeypatch, tmp_path, llm=llm)

    response = manager.search_keyword_concepts(["organogram", "rail"])

    assert service.retrieves == [["organogram", "rail"]]
    assert "Candidates in retrieval order" in response


def test_hybrid_runs_one_fused_search_without_splitting(monkeypatch, tmp_path):
    llm = _FakeLLM(["unused"], {"unused": [["unused"]]})
    _, manager, service, _ = _manager(
        monkeypatch, tmp_path, llm=llm, mode=RetrievalMode.HYBRID
    )

    manager.search_keyword_concepts(["organogram", "rail"])

    assert service.retrieves == [["organogram", "rail"]]
    assert llm.calls == []


def test_single_strategy_never_calls_the_llm(monkeypatch, tmp_path):
    llm = _FakeLLM(["unused"], {"unused": [["unused"]]})
    discovery = DiscoveryConfig(keyword_search="single")
    _, manager, service, _ = _manager(monkeypatch, tmp_path, llm=llm, discovery=discovery)

    manager.search_keyword_concepts(["organogram", "rail"])

    assert service.retrieves == [["organogram", "rail"]]
    assert llm.calls == []


def test_a_step_whose_subset_an_earlier_step_banned_never_reaches_the_portal(
    monkeypatch, tmp_path
):
    llm = _FakeLLM(["organogram rail 2021"], {"rail": [
        ["organogram", "railway", "2021"],  # empty: shrinks to the ban {railway}
        ["railway", "office"],              # contains {railway}: skipped
        ["organogram", "office"],           # finds the table: cascade ends
        ["organogram"],
    ]})
    state, manager, service, notices = _manager(
        monkeypatch, tmp_path, llm=llm, service=_CountingRetrievalService()
    )

    response = manager.search_keyword_concepts(["organogram"])

    assert service.retrieves == [["2021", "organogram", "railway"], ["office", "organogram"]]
    assert state.failed_keyword_combinations == [frozenset({"railway"})]
    assert any("skipped, contains banned {railway}" in notice for notice in notices)
    assert "orr_2021_11.parquet" in response


def test_a_step_that_is_itself_banned_is_skipped(monkeypatch, tmp_path):
    llm = _FakeLLM(["organogram rail"], {"rail": [
        ["Rail", "Organogram"],
        ["organogram"],
    ]})
    state, manager, service, _ = _manager(monkeypatch, tmp_path, llm=llm)
    state.failed_keyword_combinations = [frozenset({"organogram", "rail"})]

    manager.search_keyword_concepts(["organogram"])

    assert service.retrieves == [["organogram"]]


def test_one_cascade_spends_one_search_attempt(monkeypatch, tmp_path):
    # Three empty steps are three retrievals but one search_tables call, so
    # the agent keeps its zero-result retries for a different query...
    llm = _FakeLLM(["budget"], {"budget": [
        ["budget", "rail"], ["budget", "road"], ["budget"],
    ]})
    state, manager, service, _ = _manager(monkeypatch, tmp_path, llm=llm)

    manager.search_keyword_concepts(["budget"])

    assert len(service.retrieves) == 3
    assert manager.is_tool_available("search_tables")

    # ...and once a cascade finds a table, searching is over: go select.
    llm.requests = ["organogram rail"]
    llm.cascades = {"rail": [["organogram", "rail"]]}
    manager.search_keyword_concepts(["organogram", "rail"])

    assert not manager.is_tool_available("search_tables")


def test_region_short_forms_survive_and_the_question_is_in_context(monkeypatch, tmp_path):
    llm = _FakeLLM([], {"Belfast": [
        ["network", "contribution", "Belfast", "NI"],
        ["highest", "network", "contribution", "NI"],
        ["network", "contribution"],
    ]})
    _, manager, _, _ = _manager(monkeypatch, tmp_path, llm=llm)

    cascade = manager._propose_keyword_cascade("network contribution Belfast NI 2007", [])

    assert cascade == [
        ["network", "contribution", "belfast", "ni"],
        ["network", "contribution", "ni"],
        ["network", "contribution"],
    ]
    assert f"QUESTION: {QUESTION}" in llm.users[0]
