"""Every retrieval mode is a registered retriever; the service composes and dispatches through it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.models import RetrievalHit
from lakegen.retrieval.registry import (
    Branches,
    ModeRetriever,
    RetrievalRequest,
    RetrievalResult,
    build,
    registered_modes,
)
from lakegen.retrieval.retrievers import HybridMode, KeywordMode


def test_every_mode_is_registered_and_an_unknown_one_names_the_known():
    assert set(registered_modes()) == set(RetrievalMode)
    with pytest.raises(ValueError, match="unsupported retrieval mode: nope; registered: duckdb_agentic, hybrid"):
        build("nope", Branches(config=RetrievalConfig(), keyword=None))


def _hit(name: str, score: float) -> RetrievalHit:
    return RetrievalHit(document={"resource_id": name}, score=score, rank=0)


def test_the_keyword_mode_fetches_more_cuts_to_top_k_and_numbers_the_ranks():
    calls = []

    def retrieve(keywords, *, top_k, q_op):
        calls.append((keywords, top_k, q_op))
        return [_hit("a", 3.0), _hit("b", 2.0), _hit("c", 1.0)]

    branches = Branches(config=RetrievalConfig(), keyword=SimpleNamespace(retrieve=retrieve))
    result = build(RetrievalMode.KEYWORD, branches).retrieve(
        RetrievalRequest(question="q", keywords=("parks", "district"), top_k=2, lexical_fetch_k=10, q_op="OR")
    )
    assert calls == [(["parks", "district"], 10, "OR")]
    assert [(hit.document["resource_id"], hit.rank, hit.lexical_rank) for hit in result.hits] == [("a", 1, 1), ("b", 2, 2)]
    assert result.lexical_hit_count is None and isinstance(build(RetrievalMode.KEYWORD, branches), KeywordMode)


def test_the_hybrid_mode_reports_the_lexical_count_and_a_missing_branch_is_named():
    hybrid = SimpleNamespace(retrieve_with_lexical_count=lambda question, keywords, *, top_k: ([_hit("a", 1.0)], 7))
    branches = Branches(config=RetrievalConfig(mode=RetrievalMode.HYBRID), keyword=None, hybrid=hybrid)
    retriever = build(RetrievalMode.HYBRID, branches)
    assert isinstance(retriever, HybridMode) and isinstance(retriever, ModeRetriever)
    result = retriever.retrieve(RetrievalRequest(question="q", keywords=("parks",), top_k=5))
    assert result == RetrievalResult([_hit("a", 1.0)], 7)
    with pytest.raises(ValueError, match="semantic retrieval needs the semantic branch"):
        build(RetrievalMode.SEMANTIC, Branches(config=RetrievalConfig(mode=RetrievalMode.SEMANTIC), keyword=None))


def test_mode_traits_name_what_the_service_composes():
    assert RetrievalMode.HYBRID.fuses and RetrievalMode.HYBRID_DUCKDB_SEMANTIC.fuses and not RetrievalMode.KEYWORD.fuses
    assert RetrievalMode.DUCKDB_AGENTIC.uses_duckdb and RetrievalMode.HYBRID_DUCKDB_SEMANTIC.uses_duckdb
    assert not RetrievalMode.PNEUMA.uses_duckdb
