"""The table selectors: a protocol, a registry, and which one a run uses.

A ``Selector`` picks the tables that answer a question from the candidates
retrieval ranked, and says why. Each architecture has its own LLM selector
(the judge agent of the divided architecture, the tool-free turn of the
orchestrated one); a run may configure another kind instead, which is
registered here by its ``TableSelector`` value and built with
``build(kind, discovery)``. ``choose(discovery)`` returns that selector, or
None when the architecture's own LLM selector should run. A selector that
cannot run (no key, a failed service) raises ``SelectorUnavailable`` and the
architecture falls back to its LLM selector, saying so in the trace.

To add a selector: a value in ``TableSelector``, a class with ``kind``,
``title`` and ``select(request)``, one ``@register(TableSelector.NEW)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from lakegen.experiment_config import DiscoveryConfig, TableSelector


class SelectorUnavailable(RuntimeError):
    """The selector could not run; the architecture's LLM selector takes over."""


@dataclass(frozen=True)
class SelectionRequest:
    question: str
    candidates: list[str]
    metadata: Mapping[str, Any]
    """The catalogue entries of the candidates, by table."""
    table_dir: Path | None
    discovery: DiscoveryConfig


@dataclass(frozen=True)
class Selection:
    selected: list[str]
    reasoning: str
    trace: str
    tokens: int = 0
    llm_calls: int = 0
    plan: dict[str, Any] | None = None
    """A selection plan for the coder, when the selector writes one."""
    advisories: tuple[str, ...] = ()
    source: str = ""
    """What the plan says made it (``jev_selector``)."""


@runtime_checkable
class Selector(Protocol):
    kind: TableSelector
    title: str
    """How the chat names it ("Jev table selector")."""

    def select(self, request: SelectionRequest) -> Selection: ...


Factory = Callable[[DiscoveryConfig], Selector]
_REGISTRY: dict[TableSelector, Factory] = {}


def register(kind: TableSelector) -> Callable[[Factory], Factory]:
    def mark(factory: Factory) -> Factory:
        _REGISTRY[TableSelector(kind)] = factory
        return factory

    return mark


def registered_kinds() -> tuple[TableSelector, ...]:
    return tuple(_REGISTRY)


def build(kind: TableSelector | str, discovery: DiscoveryConfig) -> Selector:
    """The selector of ``kind``; an unregistered kind names the registered ones."""
    try:
        factory = _REGISTRY[TableSelector(kind)]
    except (KeyError, ValueError):
        known = ", ".join(sorted(item.value for item in _REGISTRY))
        raise ValueError(f"unsupported table selector: {kind}; registered: {known}") from None
    return factory(discovery)


def choose(discovery: DiscoveryConfig | None) -> Selector | None:
    """The configured selector, or None for the architecture's own LLM selector."""
    if discovery is None or discovery.selector is TableSelector.LLM:
        return None
    return build(discovery.selector, discovery)


# Importing the selectors registers them.
from lakegen.discovery.selectors import jev  # noqa: E402, F401
