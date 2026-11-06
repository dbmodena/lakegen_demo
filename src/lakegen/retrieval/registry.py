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
from lakegen.datastructures.retrieval import RetrievalHit

if TYPE_CHECKING:
    from lakegen.retrieval.modes.duckdb_agentic import DuckDBAgenticRetriever
    from lakegen.retrieval.modes.duckdb_semantic import DuckDBSemanticRetriever
    from lakegen.retrieval.modes.hybrid import HybridRetriever
    from lakegen.retrieval.modes.keyword import KeywordRetriever
    from lakegen.retrieval.modes.pneuma import PneumaRetriever
    from lakegen.retrieval.modes.semantic import SemanticRetriever


@dataclass(frozen=True)
class RetrievalRequest:
    """What a mode retrieves for: the question and the search terms derived from it.

    Attributes:
        question: The question in natural language.
        keywords: The search keywords.
        top_k: How many tables to return.
        lexical_fetch_k: How many lexical hits to fetch before cutting to ``top_k``.
        q_op: The Solr query operator joining the keywords (``AND`` or ``OR``).
        entities: Named entities of the question, for the modes that search contents.
    """

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
    """The ranking a mode returned.

    Attributes:
        hits: The ranked tables, best first.
        lexical_hit_count: What the lexical branch matched, when the mode has one
            and reports it.
    """

    hits: list[RetrievalHit]
    lexical_hit_count: int | None = None
    """What the lexical branch matched, when the mode has one and reports it."""


@runtime_checkable
class ModeRetriever(Protocol):
    """One retrieval mode, built from the branches: ranks the lake's tables for a request."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Rank the lake's tables for ``request``."""
        ...


@dataclass
class Branches:
    """What the service prepared for the modes to compose from; None where the mode needs it not.

    Attributes:
        config: The retrieval configuration of the run.
        keyword: The Solr keyword retriever, always prepared.
        semantic: The Solr vector retriever.
        hybrid: The keyword plus vector retriever.
        pneuma: The Pneuma retriever.
        duckdb_agentic: The retriever that probes the local tables with DuckDB.
        duckdb_semantic: The DuckDB plus semantic hybrid retriever.
        extras: Room for a new mode's own branch without changing this class.
    """

    config: RetrievalConfig
    keyword: KeywordRetriever
    semantic: SemanticRetriever | None = None
    hybrid: HybridRetriever | None = None
    pneuma: PneumaRetriever | None = None
    duckdb_agentic: DuckDBAgenticRetriever | None = None
    duckdb_semantic: DuckDBSemanticRetriever | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    """Room for a new mode's own branch without changing this class."""

    def require(self, name: str) -> Any:
        """The branch called ``name``.

        Args:
            name: The attribute name of the branch.

        Returns:
            The branch.

        Raises:
            ValueError: When the service did not prepare it.
        """
        branch = getattr(self, name)
        if branch is None:
            raise ValueError(
                f"{self.config.mode.value} retrieval needs the {name} branch, which was not prepared"
            )
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
    """The modes registered so far, in registration order."""
    return tuple(_REGISTRY)


def build(mode: RetrievalMode | str, branches: Branches) -> ModeRetriever:
    """The retriever of ``mode``; an unregistered mode names the registered ones."""
    try:
        factory = _REGISTRY[RetrievalMode(mode)]
    except (KeyError, ValueError):
        known = ", ".join(sorted(item.value for item in _REGISTRY))
        raise ValueError(f"unsupported retrieval mode: {mode}; registered: {known}") from None
    return factory(branches)
