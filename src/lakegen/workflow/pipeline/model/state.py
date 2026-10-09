"""The state of one run that the stages share and the rounds carry forward.

A question runs through up to three discovery rounds, each followed by
coding attempts; what one round learns (which tables the architect or the
coder proved unusable, which it found useful, which keywords failed) feeds
the next. ``RunState`` holds that, plus the current selection the coder
works on and the memory the discovery agent reads and writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lakegen.tools.unified import P12State


@dataclass
class RunState:
    # The current selection, as the last discovery round left it.
    selected: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    solr_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasoning: str = ""
    trace: str = ""
    hint: str = ""
    """What the next discovery round is told about the last one."""
    error: str = ""
    """The last failure, carried into the next coding attempt and the final result."""

    # The discovery agent's state: one per unified round, kept for the record.
    selection_state: P12State = field(default_factory=P12State)
    selection_states: list[P12State] = field(default_factory=list)
    selection_attempts: list[dict[str, Any]] = field(default_factory=list)
    last_coder_rejection: dict[str, Any] | None = None

    # Keywords tried, and the tables the rounds proved unusable or useful.
    attempted_keywords: list[str] = field(default_factory=list)
    excluded_tables: set[str] = field(default_factory=set)
    """Casefolded names of tables an architect or coder rejected outright."""
    carried_tables: list[str] = field(default_factory=list)
    """Tables a rejected round still found useful; the next round starts from them."""
    carried_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    carried_inspection: dict[str, str] = field(default_factory=dict)
    """The unified agent's cached inspections of the carried tables (casefolded names)."""

    # The keyword memory: the durable scope on disk, and what this run adds.
    keyword_memory_path: Path | None = None
    keyword_memory_scope: str = ""
    retrieval_question_key: str = ""
    retrieval_memory_summary: str = ""
    failed_keyword_combinations: list[frozenset[str]] = field(default_factory=list)
    banned_table_keyword_combinations: list[frozenset[str]] = field(default_factory=list)
    """Word sets that found only this question's excluded tables; carried, never persisted."""
    keyword_history: list[list[str]] = field(default_factory=list)
    persisted_keyword_count: int = 0

    last_generated: Any = None
    """The coder's last result object, for the record."""
    after_coder_rejection: bool = False
    forced_selection: bool = False
    """The last round's selection was forced on the last attempt, not chosen."""
    architect_feedback: str = ""
    """The architect's feedback when it rejected the last round; empty otherwise."""
    last_keyword_round: tuple[Any, ...] | None = None
    """The divided architecture's last approved keyword round, for a re-selection with the same keywords."""
    reuse_keywords: bool = False
    retrieval_memory_events: list[dict[str, Any]] = field(default_factory=list)
    """The agent's retrieval evidence of this run, when the memory scope is the session."""
    """The next discovery round follows a coder rejection (a gated review may only approve or abort)."""

    def carry(self, keep_tables: list[str], skip_tables: list[str], inspection_cache: dict[str, str] | None = None) -> None:
        """Apply a rejection's per-table judgment: exclude the skipped, carry the kept.

        A kept table carries its metadata from the current selection and,
        when the unified agent cached one, its inspection.
        """
        for table in skip_tables:
            self.excluded_tables.add(table.casefold())
        for table in keep_tables:
            if table.casefold() in self.excluded_tables:
                continue
            if table not in self.carried_tables:
                self.carried_tables.append(table)
            if table in self.solr_meta:
                self.carried_metadata[table] = self.solr_meta[table]
            cached_inspection = (inspection_cache or {}).get(table.casefold())
            if cached_inspection:
                self.carried_inspection[table.casefold()] = cached_inspection
        self.carried_tables = [
            table for table in self.carried_tables if table.casefold() not in self.excluded_tables
        ]

    def prune_inspection(self) -> None:
        """Forget the cached inspections of the tables excluded so far."""
        self.carried_inspection = {
            key: value for key, value in self.carried_inspection.items() if key not in self.excluded_tables
        }
