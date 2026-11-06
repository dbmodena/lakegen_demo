"""How many times a run may try, and what it must not try again.

``Attempts`` counts a bounded loop of discovery rounds or coding attempts
and says when the current one is the last. ``DedupeGuard`` remembers the
selections the coder rejected and the plans the selection-contract gate
refused, so a round that reproduces one is caught before it is coded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Attempt:
    """One attempt of a bounded loop.

    Attributes:
        index: The attempt's index, from 0.
        limit: How many attempts the loop allows.
    """

    index: int
    limit: int

    @property
    def number(self) -> int:
        """The attempt's number, from 1."""
        return self.index + 1

    @property
    def last(self) -> bool:
        """Whether this is the loop's last attempt."""
        return self.index >= self.limit - 1


class Attempts:
    """``for attempt in Attempts(3):`` — the attempts 0, 1 and 2, the last marked."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    def __iter__(self) -> Iterator[Attempt]:
        for index in range(self.limit):
            yield Attempt(index, self.limit)


def rejected_selection_signature(
    tables: list[str], details: Mapping[str, Any] | None, reason: str
) -> dict[str, Any]:
    """Build stable feedback for retry exclusion without benchmark metadata."""
    details = details or {}
    missing = sorted(
        {str(item).strip() for item in details.get("missing_requirements", []) if str(item).strip()}
    )
    columns = sorted(
        set(
            re.findall(
                r"(?:column|columns?)\s+['\"]?([A-Za-z0-9 _/-]{2,80})",
                " ".join([reason, *missing]),
                flags=re.IGNORECASE,
            )
        )
    )
    return {
        "tables": sorted(table.casefold() for table in tables),
        "missing_requirements": missing,
        "category": str(details.get("category") or "tables_rejected"),
        "incompatible_columns": columns[:20],
    }


def selection_plan_signature(tables: list[str], selection_plan: Mapping[str, Any] | None) -> str:
    """Return a stable identity for detecting an unchanged rejected plan."""
    plan = selection_plan or {}
    semantic = (
        plan.get("coder_brief")
        or plan.get("semantic_plan")
        or {
            key: plan.get(key)
            for key in (
                "requirement_coverage",
                "requirements",
                "requirement_ledger",
                "table_roles",
                "combination_strategy",
                "uncovered_requirements",
            )
            if key in plan
        }
    )
    return json.dumps(
        {
            "tables": sorted(str(table).casefold() for table in tables),
            "plan": semantic,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


@dataclass
class DedupeGuard:
    """The selections the coder rejected and the plans the selection-contract gate refused.

    Attributes:
        rejected_keys: The rejected selections, as casefolded, sorted table names.
        rejected_signatures: The feedback of every rejected selection, in order.
        rejected_plans: Plan signature -> the correction the gate asked for.
    """

    rejected_keys: set[tuple[str, ...]] = field(default_factory=set)
    rejected_signatures: list[dict[str, Any]] = field(default_factory=list)
    rejected_plans: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def selection_key(selected: list[str]) -> tuple[str, ...]:
        """The selection's identity: its casefolded table names, sorted."""
        return tuple(sorted(table.casefold() for table in selected))

    def prior_rejection(self, selected: list[str]) -> dict[str, Any] | None:
        """The last recorded rejection of this very selection, if the coder rejected it before."""
        key = self.selection_key(selected)
        if not selected or key not in self.rejected_keys:
            return None
        return next(
            (item for item in reversed(self.rejected_signatures) if tuple(item["tables"]) == key),
            {},
        )

    def reject_selection(
        self, selected: list[str], details: Mapping[str, Any] | None, reason: str
    ) -> None:
        """Remember that the coder rejected this selection, with its details and reason."""
        self.rejected_keys.add(self.selection_key(selected))
        self.rejected_signatures.append(rejected_selection_signature(selected, details, reason))

    def repeated_plan(self, selected: list[str], signature: str) -> dict[str, Any] | None:
        """The gate's correction, when a non-empty selection reproduces a refused plan.

        An empty selection has no plan identity: it can come from a failed
        discovery, and treating it as a repeated plan would hide that.

        Args:
            selected: The round's selection.
            signature: The round's plan signature (``selection_plan_signature``).

        Returns:
            The correction the gate asked for, or None when the plan is new.
        """
        if not selected:
            return None
        return self.rejected_plans.get(signature)

    def reject_plan(self, signature: str, feedback: dict[str, Any]) -> None:
        """Remember that the gate refused this plan, and the correction it asked for."""
        self.rejected_plans[signature] = feedback
