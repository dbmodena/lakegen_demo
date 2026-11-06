"""The divided architecture's keyword writer: the intent it streams, its reasoning, its tokens, its guards,
and the strict-AND search that sends a proposal matching no table back to it."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lakegen.config.retrieval import FusionMethod
from lakegen.pipeline.discovery.keywords import (
    MAX_REFUSALS,
    KeywordSearch,
    strict_and_count,
    write_keywords,
)
from lakegen.pipeline.discovery.terms import keyword_terms
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.intent import parse_retrieval_intent

INTENT = {
    "status": "resolved",
    "concepts": ["parks", "district"],
    "entities": ["Manhattan"],
    "search_values": [],
    "measures": ["count"],
    "filters": [],
    "time_constraints": [],
    "group_by": ["district"],
    "order_by": [],
    "limit": None,
    "join_requirements": [],
    "missing_evidence": [],
}
ANSWER = "RETRIEVAL_INTENT: " + json.dumps(INTENT)


def _chunk(delta: str = "", thinking: str = "", usage: int | None = None) -> SimpleNamespace:
    kwargs = {"thinking_delta": thinking} if thinking else {}
    raw = {"usage": {"total_tokens": usage}} if usage is not None else {}
    return SimpleNamespace(delta=delta, additional_kwargs=kwargs, raw=raw)


class FakeLLM:
    """Streams scripted chunks (an exception among them is raised there); records the request options.

    ``usage_total`` is what the model's own accounting reports once a stream
    ended; the writer resets it before it starts.
    """

    def __init__(self, *streams, usage_total: int = 0) -> None:
        self.streams = list(streams)
        self.usage_after = usage_total
        self.token_usage_total = 0
        self.requests: list[dict] = []
        self.conversations: list[list] = []

    def reset_token_usage(self) -> None:
        self.token_usage_total = 0

    def stream_chat(self, messages, **kwargs):
        self.requests.append(kwargs)
        self.conversations.append(list(messages))
        stream = self.streams.pop(0)
        if isinstance(stream, Exception):
            raise stream
        for chunk in stream:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk
        self.token_usage_total = self.usage_after


def _run(llm, **options):
    pm = SimpleNamespace(render=lambda *a, **k: "prompt words")
    return write_keywords("How many parks per district?", llm, pm, **options)


def test_the_intent_the_model_streams_becomes_the_keywords_and_its_thinking_the_reasoning():
    llm = FakeLLM(
        [
            _chunk(thinking="Parks per district, so "),
            _chunk(delta=ANSWER[:30]),
            _chunk(delta=ANSWER[30:], usage=42),
        ]
    )
    keywords, raw, tokens, reasoning = _run(llm)
    assert keywords == parse_retrieval_intent(ANSWER).search_terms(False)
    assert raw == ANSWER and tokens == 42 and reasoning == "Parks per district, so"
    assert llm.requests == [{"think": True}]


def test_a_backend_that_rejects_the_thinking_option_gets_a_plain_request():
    llm = FakeLLM(RuntimeError("unknown option think"), [_chunk(delta=ANSWER)])
    keywords, raw, _, _ = _run(llm)
    assert keywords and raw == ANSWER and llm.requests == [{"think": True}, {}]
    # Once something streamed, an error is the caller's to see.
    with pytest.raises(RuntimeError, match="late"):
        _run(FakeLLM([_chunk(delta="RETRIEVAL_INTENT:"), RuntimeError("late")]))


def test_without_streamed_reasoning_the_request_asks_for_none():
    llm = FakeLLM([_chunk(delta=ANSWER)])
    _run(llm, stream_reasoning=False)
    assert llm.requests == [{}]


def test_tagged_reasoning_in_the_answer_is_split_out_and_cell_values_can_be_the_terms():
    intent = {**INTENT, "search_values": ["East River"]}
    answer = "<think>consider rivers</think>RETRIEVAL_INTENT: " + json.dumps(intent)
    keywords, raw, _, reasoning = _run(FakeLLM([_chunk(delta=answer)]), value_search=True)
    assert keywords == ["East River"] and raw == "RETRIEVAL_INTENT: " + json.dumps(intent)
    assert reasoning == "consider rivers"


def test_an_unresolved_or_unparsable_intent_yields_no_keywords():
    unresolved = {**INTENT, "status": "unresolved", "missing_evidence": ["the period"]}
    keywords, _, _, _ = _run(FakeLLM([_chunk(delta="RETRIEVAL_INTENT: " + json.dumps(unresolved))]))
    assert keywords == []
    keywords, raw, _, _ = _run(FakeLLM([_chunk(delta="I cannot decide.")]))
    assert keywords == [] and raw == "I cannot decide."


def test_an_intent_may_name_more_than_two_concepts():
    intent = {**INTENT, "concepts": ["network contribution", "Belfast", "NI"]}
    keywords, _, _, _ = _run(FakeLLM([_chunk(delta="RETRIEVAL_INTENT: " + json.dumps(intent))]))
    assert keywords == ["network contribution", "Belfast", "NI"]


def test_a_reasoning_loop_stops_the_stream_and_is_reported():
    tail = "the same thought again and again, " * 10
    llm = FakeLLM([_chunk(thinking=tail * 6), _chunk(delta=ANSWER)])
    keywords, raw, _, reasoning = _run(llm)
    assert keywords == [] and raw == ""
    assert reasoning.endswith("⚠️ **[Phase 1] The model looped; retrieval intent is unresolved.**")


def test_tokens_come_from_the_stream_then_the_model_then_an_estimate():
    _, _, tokens, _ = _run(FakeLLM([_chunk(delta=ANSWER, usage=10)], usage_total=99))
    assert tokens == 99
    _, _, tokens, _ = _run(FakeLLM([_chunk(delta=ANSWER)]))
    assert tokens == int((2 + 2 + len(ANSWER.split())) * 1.3)


def test_a_cancellation_check_runs_on_every_chunk():
    seen = []

    def cancel():
        seen.append(True)
        if len(seen) == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(
            FakeLLM([_chunk(delta="a"), _chunk(delta="b"), _chunk(delta="c")]), cancel_check=cancel
        )
    assert len(seen) == 2


def test_follow_ups_continue_the_conversation_and_banned_sets_reach_the_first_message():
    rendered = {}

    def render(_name, part, **values):
        rendered[part] = values
        return part

    llm = FakeLLM([_chunk(delta=ANSWER)])
    write_keywords(
        "q",
        llm,
        SimpleNamespace(render=render),
        zero_result_sets=[["2007"], ["belfast", "ni"]],
        follow_ups=[("first answer", "it matched nothing"), ("second answer", "refused")],
    )
    assert rendered["user_prompt"]["zero_result_sets_str"] == "{2007}; {belfast, ni}"
    (conversation,) = llm.conversations
    assert [(m.role.value, m.content) for m in conversation] == [
        ("system", "system_prompt"),
        ("user", "user_prompt"),
        ("assistant", "first answer"),
        ("user", "it matched nothing"),
        ("assistant", "second answer"),
        ("user", "refused"),
    ]


# --- the strict-AND search of the writer's keywords ----------------------------

CATALOGUE = [{"network", "contribution", "belfast", "ni"}, {"parks", "district"}]


class Counter:
    """How many CATALOGUE entries contain every word; records each count."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, keywords) -> int:
        if self.fail:
            raise ConnectionError("solr is down")
        self.calls.append(sorted(keyword_terms(keywords)))
        return sum(keyword_terms(keywords) <= entry for entry in CATALOGUE)


def test_a_proposal_that_matches_a_table_goes_to_retrieval_as_written():
    search = KeywordSearch(Counter(), [], searches=3)
    assert search.review(["network contribution", "belfast"]) is None
    assert search.log == [
        {"keywords": ["network contribution", "belfast"], "outcome": "matched", "tables": 1}
    ]
    assert not search.exhausted and search.bans == []


def test_an_empty_proposal_is_banned_as_its_smallest_failing_sets_and_sent_back():
    bans: list[frozenset[str]] = []
    search = KeywordSearch(Counter(), bans, searches=3)
    feedback = search.review(["network contribution", "belfast 2007"])
    assert bans == [frozenset({"2007"})]
    assert (
        "no catalogue entry contains every word of {2007, belfast, contribution, network}"
        in feedback
    )
    assert "Smallest word sets that already match nothing: {2007}." in feedback
    assert "refused from now on: {2007}" in feedback and feedback.endswith("Searches left: 2.")
    assert search.log[-1]["outcome"] == "empty" and not search.exhausted


def test_a_proposal_containing_a_banned_set_is_refused_unsearched_and_spends_no_search():
    counter = Counter()
    search = KeywordSearch(counter, [frozenset({"2007"})], searches=1)
    feedback = search.review(["belfast 2007"])
    assert feedback == (
        "Refused before searching: {2007, belfast} contains {2007}, a word set no table matches. "
        "Write a new RETRIEVAL_INTENT whose concepts leave out that combination."
    )
    assert counter.calls == [] and search.log == [
        {"keywords": ["belfast 2007"], "outcome": "refused", "banned": ["2007"]}
    ]
    # The one search is still there to spend.
    assert search.review(["belfast"]) is None and search.log[-1]["outcome"] == "matched"


def test_the_round_goes_on_once_its_searches_or_refusals_are_spent():
    search = KeywordSearch(Counter(), [], searches=2)
    assert search.review(["network 1999"]) is not None
    assert search.review(["network 2001"]) is None and search.exhausted
    assert [entry["outcome"] for entry in search.log] == ["empty", "empty"]

    search = KeywordSearch(Counter(), [frozenset({"2007"})], searches=3)
    for _ in range(MAX_REFUSALS):
        assert search.review(["2007"]).startswith("Refused before searching")
    assert search.review(["2007"]) is None and search.exhausted


def test_a_count_that_fails_lets_the_keywords_through_and_no_keywords_skip_the_check():
    search = KeywordSearch(Counter(fail=True), [], searches=3)
    assert search.review(["network"]) is None
    assert (
        search.log == [{"keywords": ["network"], "outcome": "unchecked"}] and not search.exhausted
    )
    assert search.review([]) is None and len(search.log) == 1


def test_only_a_strict_and_keyword_side_is_counted():
    solr = object()
    assert strict_and_count(solr, RetrievalConfig(mode=RetrievalMode.KEYWORD)) is not None
    assert strict_and_count(solr, RetrievalConfig(mode=RetrievalMode.HYBRID)) is not None
    assert strict_and_count(solr, RetrievalConfig(mode=RetrievalMode.SEMANTIC)) is None
    assert (
        strict_and_count(
            solr, RetrievalConfig(mode=RetrievalMode.KEYWORD, keyword_or_fallback=True)
        )
        is None
    )
    no_lexical = RetrievalConfig(
        mode=RetrievalMode.HYBRID, fusion_method=FusionMethod.WEIGHTED, alpha=0.0
    )
    assert strict_and_count(solr, no_lexical) is None
