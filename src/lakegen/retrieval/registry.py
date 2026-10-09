"""The retrieval modes: a protocol, a registry, and what a mode composes from.

A mode is a ``ModeRetriever``: it ranks the lake's tables for a
``RetrievalRequest`` and says what its lexical branch matched, when it has
one. ``register(mode)`` marks the class that builds the mode's retriever
from the ``Branches`` the service prepared (Solr, the embedding model, the
local tables, the Pneuma client); ``build(mode, branches)`` is the one place
a mode name turns into code, and an unregistered mode fails with the known
ones listed.

To add a mode: a value in ``RetrievalMode`` (with its traits), a class with
``retrieve(request)`` and one ``@register(RetrievalMode.NEW)`` on it, next
to the retriever it composes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from lakegen.config.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.models import RetrievalHit

if TYPE_CHECKING:
    from lakegen.retrieval.duckdb_agentic import DuckDBAgenticRetriever
    from lakegen.retrieval.duckdb_semantic import DuckDBSemanticHybridRetriever
    from lakegen.retrieval.hybrid import HybridRetriever
    from lakegen.retrieval.keyword import KeywordRetriever
    from lakegen.retrieval.pneuma import PneumaRetriever
    from lakegen.retrieval.semantic import SemanticRetriever


@dataclass(frozen=True)
class RetrievalRequest:
    question: str
    keywords: tuple[str, ...]
    top_k: int
    lexical_fetch_k: int | None = None
    """How many lexical hits to fetch before cutting to ``top_k`` (the keyword mode)."""
    q_op: str = "AND"
    entities: tuple[str, ...] = ()
    """Named entities of the question, for the modes that search contents."""


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[RetrievalHit]
    lexical_hit_count: int | None = None
    """What the lexical branch matched, when the mode has one and reports it."""


@runtime_checkable
class ModeRetriever(Protocol):
    """One retrieval mode, built from the branches: ranks the lake's tables for a request."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


@dataclass
class Branches:
    """What the service prepared for the modes to compose from; None where the mode needs it not."""

    config: RetrievalConfig
    keyword: KeywordRetriever
    semantic: SemanticRetriever | None = None
    hybrid: HybridRetriever | None = None
    pneuma: PneumaRetriever | None = None
    duckdb_agentic: DuckDBAgenticRetriever | None = None
    duckdb_semantic: DuckDBSemanticHybridRetriever | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    """Room for a new mode's own branch without changing this class."""

    def require(self, name: str) -> Any:
        branch = getattr(self, name)
        if branch is None:
            raise ValueError(f"{self.config.mode.value} retrieval needs the {name} branch, which was not prepared")
        return branch


Factory = Callable[[Branches], ModeRetriever]
_REGISTRY: dict[RetrievalMode, Factory] = {}


def register(*modes: RetrievalMode) -> Callable[[Factory], Factory]:
    """Mark the class (or function) that builds the retriever of ``modes`` from the branches."""

    def mark(factory: Factory) -> Factory:
        for mode in modes:
            _REGISTRY[RetrievalMode(mode)] = factory
        return factory

    return mark


def registered_modes() -> tuple[RetrievalMode, ...]:
    return tuple(_REGISTRY)


def build(mode: RetrievalMode | str, branches: Branches) -> ModeRetriever:
    """The retriever of ``mode``; an unregistered mode names the registered ones."""
    try:
        factory = _REGISTRY[RetrievalMode(mode)]
    except (KeyError, ValueError):
        known = ", ".join(sorted(item.value for item in _REGISTRY))
        raise ValueError(f"unsupported retrieval mode: {mode}; registered: {known}") from None
    return factory(branches)
