"""The state of one run that the stages share and the rounds carry forward.

A question runs through up to three discovery rounds, each followed by
coding attempts; what one round learns (which tables the architect or the
coder proved unusable, which it found useful, which keywords failed) feeds
the next. ``RunState`` holds that, plus the current selection the coder
works on and the memory the discovery agent reads and writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lakegen.pipeline.discovery.state import DiscoveryState


@dataclass
class RunState:
    """What the stages of one run share and the rounds carry forward.

    Attributes:
        selected: The current selection, as the last discovery round left it.
        keywords: The keywords of the current selection.
        solr_meta: The catalogue metadata of the selected tables, by table.
        reasoning: The last discovery round's reasoning.
        trace: The last discovery round's trace.
        hint: What the next discovery round is told about the last one.
        error: The last failure, carried into the next coding attempt and the final result.
        selection_state: The discovery round's state, of the current round.
        selection_states: The discovery round's state of every unified round, kept for the record.
        selection_attempts: The record of every discovery round.
        last_coder_rejection: The coder's last table rejection: its reason, details and tables.
        attempted_keywords: The keywords tried so far.
        excluded_tables: Casefolded names of tables an architect or coder rejected outright.
        carried_tables: Tables a rejected round still found useful; the next round starts from them.
        carried_metadata: The catalogue metadata of the carried tables.
        zero_result_bans: Word sets the divided keyword search proved to match no table in this
            run. Never loaded or persisted, so a question's search does not depend on the
            questions before it.
        keyword_searches: This round's keyword proposals: matched, empty, refused, or unchecked
            when the count failed.
        keyword_search_exhausted: This round's keyword search spent its budget without a
            proposal that matched a table.
        last_generated: The coder's last result object, for the record.
        after_coder_rejection: The next discovery round follows a coder rejection (a gated
            review may only approve or abort).
        forced_selection: The last round's selection was forced on the last attempt, not chosen.
        architect_feedback: The architect's feedback when it rejected the last round; empty
            otherwise.
        last_keyword_round: The divided architecture's last approved keyword round, for a
            re-selection with the same keywords.
        reuse_keywords: The next divided round reuses the last keyword round instead of
            writing new keywords.
    """

    # The current selection, as the last discovery round left it.
    selected: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    solr_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasoning: str = ""
    trace: str = ""
    hint: str = ""
    error: str = ""

    # The discovery round's state: one per unified round, kept for the record.
    selection_state: DiscoveryState = field(default_factory=DiscoveryState)
    selection_states: list[DiscoveryState] = field(default_factory=list)
    selection_attempts: list[dict[str, Any]] = field(default_factory=list)
    last_coder_rejection: dict[str, Any] | None = None

    # Keywords tried, and the tables the rounds proved unusable or useful.
    attempted_keywords: list[str] = field(default_factory=list)
    excluded_tables: set[str] = field(default_factory=set)
    carried_tables: list[str] = field(default_factory=list)
    carried_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    zero_result_bans: list[frozenset[str]] = field(default_factory=list)
    keyword_searches: list[dict[str, Any]] = field(default_factory=list)
    keyword_search_exhausted: bool = False

    last_generated: Any = None
    after_coder_rejection: bool = False
    forced_selection: bool = False
    architect_feedback: str = ""
    last_keyword_round: tuple[Any, ...] | None = None
    reuse_keywords: bool = False

    def carry(self, keep_tables: list[str], skip_tables: list[str]) -> None:
        """Apply a rejection's per-table judgment: exclude the skipped, carry the kept.

        A kept table carries its metadata from the current selection.

        Args:
            keep_tables: Tables the rejection found useful.
            skip_tables: Tables the rejection proved unusable.
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
        self.carried_tables = [
            table for table in self.carried_tables if table.casefold() not in self.excluded_tables
        ]
