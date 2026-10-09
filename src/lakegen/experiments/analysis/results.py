"""What a stored result says, read the same way everywhere.

A results line of an API job holds the question, its source and the
``QueryResult`` as ``to_dict()`` wrote it. These readers are the one place
that knows where in that dict the final ranking, the selection, the outcome
and the costs live, so every analysis reads them alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lakegen.domain.tables import resource_id


def selected_tables(result: Mapping[str, Any]) -> list[str]:
    """The selected tables as resource ids, in selection order, without repeats."""
    return list(dict.fromkeys(resource_id(value) for value in result.get("tables", [])))


def final_ranking(result: Mapping[str, Any]) -> list[str]:
    """The last retrieval attempt's ranking as resource ids, best first.

    A result without a recorded ranking (older runs, failures before
    retrieval) falls back to its selection, so the selected tables still count
    as retrieved.
    """
    entries = [entry for entry in result.get("ranking", []) if isinstance(entry, dict)]
    if not entries:
        return selected_tables(result)
    final_attempt = max(int(entry.get("attempt", 1)) for entry in entries)
    final_entries = sorted(
        (entry for entry in entries if int(entry.get("attempt", 1)) == final_attempt),
        key=lambda entry: int(entry.get("rank", 0)),
    )
    return list(dict.fromkeys(
        resource_id(entry["resource_id"])
        for entry in final_entries
        if entry.get("resource_id")
    ))


def code_evaluation(result: Mapping[str, Any]) -> Mapping[str, Any]:
    return result.get("code_evaluation") or {}


def code_applicable(result: Mapping[str, Any]) -> bool:
    """Whether the run was evaluated against a reference result."""
    return bool(code_evaluation(result).get("applicable"))


def solved(result: Mapping[str, Any]) -> bool:
    """The final outcome: an exact or equivalent result, or one the semantic judge accepted."""
    return bool(code_evaluation(result).get("supported_correct"))


def expansion_used(result: Mapping[str, Any]) -> bool:
    return bool((result.get("discovery") or {}).get("expansion_used"))


@dataclass(frozen=True)
class Tokens:
    """LLM tokens a run spent, by phase: discovery (1 and 2), coding (3), answer (4)."""

    p1_p2: int = 0
    p3: int = 0
    p4: int = 0

    @property
    def total(self) -> int:
        return self.p1_p2 + self.p3 + self.p4


def tokens(result: Mapping[str, Any]) -> Tokens:
    spent = result.get("tokens") or {}
    return Tokens(*(int(spent.get(key, 0) or 0) for key in ("p1_p2", "p3", "p4")))
