"""The divided architecture's keyword writer: the intent it streams, its reasoning, its tokens, its guards."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lakegen.discovery.keywords import phase1_generate_keywords
from lakegen.retrieval.intent import parse_retrieval_intent

INTENT = {
    "status": "resolved", "concepts": ["parks", "district"], "entities": ["Manhattan"], "search_values": [],
    "measures": ["count"], "filters": [], "time_constraints": [], "group_by": ["district"], "order_by": [],
    "limit": None, "join_requirements": [], "missing_evidence": [],
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

    def reset_token_usage(self) -> None:
        self.token_usage_total = 0

    def stream_chat(self, messages, **kwargs):
        self.requests.append(kwargs)
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
    return phase1_generate_keywords("How many parks per district?", llm, pm, **options)


def test_the_intent_the_model_streams_becomes_the_keywords_and_its_thinking_the_reasoning():
    llm = FakeLLM([_chunk(thinking="Parks per district, so "), _chunk(delta=ANSWER[:30]), _chunk(delta=ANSWER[30:], usage=42)])
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
        _run(FakeLLM([_chunk(delta="a"), _chunk(delta="b"), _chunk(delta="c")]), cancel_check=cancel)
    assert len(seen) == 2
