"""The table selectors: a protocol, a registry, and which one a run uses.

A ``Selector`` picks the tables that answer a question from the candidates
retrieval ranked, and says why. The divided architecture selects with the
``memgpt`` agent, registered here by its ``TableSelector`` value and built
with ``build(kind, discovery)``; ``choose(discovery)`` returns the configured
one. A selector that cannot run (no LLM for an agent) raises ``SelectorUnavailable``:
the round asks for new keywords, and the trace says so. memgpt takes the run's
LLM, keywords, feedback and streaming hooks from the request.

To add a selector: a value in ``TableSelector``, a class with ``kind``,
``title`` and ``select(request)``, one
``@register(TableSelector.NEW)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lakegen.config.experiment import DiscoveryConfig, TableSelector
from lakegen.config.retrieval import RetrievalMode
from lakegen.datastructures.types import ToolCallback

# Selectors that past runs recorded and that no longer run.
REMOVED = {
    TableSelector.LLM: (
        "discovery.selector=llm names the judge agent the MemGPT selector replaced on "
        "2026-10-21; it no longer runs. Use discovery.selector=memgpt (the default)."
    ),
    TableSelector.JEV: (
        "discovery.selector=jev names TypeSafe's Jev, removed on 2026-10-29; it no longer "
        "runs. Use discovery.selector=memgpt (the default)."
    ),
}


class SelectorUnavailable(RuntimeError):
    """The selector could not run; the architecture's fallback takes over."""


@dataclass(frozen=True)
class SelectionRequest:
    """What a selector selects from: the question, the retrieved candidates and the run's hooks.

    Attributes:
        question: The user's question.
        candidates: The retrieved tables, in retrieval order.
        metadata: The catalogue entries of the candidates, by table.
        table_dir: Folder of the lake's tables; None where the call site has none.
        discovery: Discovery settings of the run.
        llm: The run's LLM, for a selector that is an agent; None where the call site has none.
        keywords: The keywords retrieval ran with.
        portal_name: Name of the open-data portal.
        lake: The lake's name, for the portal's catalogue note (config/lakes.yaml); "" where
            the call site has none.
        hint: Feedback from the previous attempt (a reviewer's note, the coder's rejection).
        retrieval_mode: Retrieval mode that ranked the candidates, if known.
        prompts: The run's prompt manager; None for the shared one.
        stream_callback: Receives the selector's live output, if any.
        cancel_check: Raises when the run is cancelled.
        tool_callback: Receives each tool call of a selector that is an agent, for a surface
            that shows them as steps.
    """

    question: str
    candidates: list[str]
    metadata: Mapping[str, Any]
    table_dir: Path | None
    discovery: DiscoveryConfig
    llm: Any = None
    keywords: tuple[str, ...] = ()
    portal_name: str = ""
    lake: str = ""
    hint: str = ""
    retrieval_mode: RetrievalMode | None = None
    prompts: Any = None
    stream_callback: Callable[[str], None] | None = None
    cancel_check: Callable[[], None] | None = None
    tool_callback: ToolCallback | None = None


@dataclass(frozen=True)
class Selection:
    """A selector's answer: the tables it selected (none for a rejection) and why.

    Attributes:
        selected: The selected tables; empty when the selector rejected the candidates.
        reasoning: Why, as the chat shows it.
        trace: The selector's activity log.
        tokens: Tokens the selector used.
        llm_calls: Model calls the selector made.
        plan: A selection plan for the coder, when the selector writes one.
        advisories: Non-blocking notes for the coder.
        source: What the plan says made it (``memgpt_selector``).
        keep_tables: On a rejection: inspected candidates that partly cover the question;
            the next attempt keeps them.
        skip_tables: On a rejection: candidates proven irrelevant; the next attempts exclude them.
    """

    selected: list[str]
    reasoning: str
    trace: str
    tokens: int = 0
    llm_calls: int = 0
    plan: dict[str, Any] | None = None
    advisories: tuple[str, ...] = ()
    source: str = ""
    keep_tables: tuple[str, ...] = ()
    skip_tables: tuple[str, ...] = ()


@runtime_checkable
class Selector(Protocol):
    """A table selector, as the registry builds it.

    Attributes:
        kind: The ``TableSelector`` value it is registered under.
        title: How the chat names it ("MemGPT table selector").
    """

    kind: TableSelector
    title: str

    def select(self, request: SelectionRequest) -> Selection:
        """The tables among the request's candidates that answer its question, and why."""
        ...


Factory = Callable[[DiscoveryConfig], Selector]
_REGISTRY: dict[TableSelector, Factory] = {}


def register(kind: TableSelector) -> Callable[[Factory], Factory]:
    """Class decorator registering a selector factory under ``kind``."""

    def mark(factory: Factory) -> Factory:
        _REGISTRY[TableSelector(kind)] = factory
        return factory

    return mark


def registered_kinds() -> tuple[TableSelector, ...]:
    """The registered selector kinds, in registration order."""
    return tuple(_REGISTRY)


def build(kind: TableSelector | str, discovery: DiscoveryConfig) -> Selector:
    """The selector of ``kind``; an unregistered kind names the registered ones."""
    try:
        factory = _REGISTRY[TableSelector(kind)]
    except (KeyError, ValueError):
        known = ", ".join(sorted(item.value for item in _REGISTRY))
        raise ValueError(f"unsupported table selector: {kind}; registered: {known}") from None
    return factory(discovery)


def choose(discovery: DiscoveryConfig | None) -> Selector:
    """Build the configured selector; ``llm`` and ``jev``, both removed, are refused.

    Args:
        discovery: Discovery settings naming the selector; the defaults when None.

    Returns:
        The selector.

    Raises:
        ValueError: The settings name a removed selector.
    """
    discovery = discovery or DiscoveryConfig()
    if discovery.selector in REMOVED:
        raise ValueError(REMOVED[discovery.selector])
    return build(discovery.selector, discovery)


# Importing the selectors registers them.
from lakegen.pipeline.discovery.selectors import memgpt  # noqa: E402, F401
