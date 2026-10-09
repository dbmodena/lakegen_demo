"""TypeSafe's Jev as a table selector: a scored answer per candidate, no LLM call."""

from __future__ import annotations

from lakegen.config.experiment import DiscoveryConfig, TableSelector
from lakegen.discovery.ledger import build_minimal_selection_fallback
from lakegen.discovery.selectors import (
    Selection,
    SelectionRequest,
    SelectorUnavailable,
    register,
)
from lakegen.discovery.selectors.jev_service import JevSelectorError, select_with_jev


@register(TableSelector.JEV)
class JevSelector:
    kind = TableSelector.JEV
    title = "Jev table selector"

    def __init__(self, discovery: DiscoveryConfig) -> None:
        self.discovery = discovery

    def select(self, request: SelectionRequest) -> Selection:
        try:
            jev = select_with_jev(
                request.question, list(request.candidates), dict(request.metadata), request.table_dir, self.discovery,
            )
        except JevSelectorError as exc:
            raise SelectorUnavailable(str(exc)) from exc
        plan, advisories = (
            build_minimal_selection_fallback(jev.selected, jev.reasoning) if jev.selected else ({}, [])
        )
        return Selection(
            selected=list(jev.selected), reasoning=jev.reasoning, trace=jev.trace,
            plan=dict(plan), advisories=tuple(advisories), source="jev_selector",
        )
