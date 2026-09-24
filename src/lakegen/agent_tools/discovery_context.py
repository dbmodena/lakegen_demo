"""Working-memory management for the unified discovery agent.

The agent's history grows by one tool result per call and is re-read in full on
every later model call, so a result stays in context long after it stopped
being useful: the ten-candidate search listing (about 3k tokens) was re-read on
every call of a round that had moved on to inspecting, and three blocked
confirmations sat side by side. Meanwhile what the round had actually
established -- which candidate shows the requested period, which question
words each one matches, what budget is left -- was spread over those results
and the model's own prose, and a model that lost track of it re-called spent
tools.

Two DiscoveryConfig flags, applied by ``DiscoveryContext.edit`` to the agent's
in-run history right before each model call:

- ``state_board``: one short, system-kept summary of the round is appended to
  the newest tool result, and older copies are removed, so exactly one current
  board sits at the end of the context. The manager also flags a call that
  repeats a refusal (a spent tool, an unchanged blocked confirmation).
- ``compact_history``: results later calls have superseded are shortened in
  place. A candidate listing becomes a one-line index per candidate once a
  later call inspected one, an older blocked confirmation becomes its first
  sentence, a long refusal likewise. Inspection profiles and join checks are
  evidence and are never shortened.

Only message text changes: every tool result keeps its tool-call id, so the
call/result pairing the model API requires is untouched. The activity log and
the UI still show every result in full.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock

from lakegen.agent_tools.requirement_ledger import (
    inspected_period_evidence,
    question_periods,
)

if TYPE_CHECKING:
    from lakegen.agent_tools.tools_p12 import Phase12ToolsManager


BOARD_MARKER = "\n\n[DISCOVERY STATE"
COMPACTED = "[compacted] "
# Prefixed by the manager to a confirmation blocked for the same reason twice.
REPEATED_BLOCK = (
    "Blocked again, for exactly the same reason as your previous "
    "confirm_unified_selection call: resubmitting without the change it asks "
    "for cannot pass. "
)
# Refusals and notes shorter than this are left alone: shortening them saves
# less than the marker costs.
_MIN_COMPACTABLE_NOTE = 240
_LISTING_ENTRY = re.compile(r"^Candidate (\d+) \(retrieval rank", re.MULTILINE)
_WORD = re.compile(r"[a-z][a-z'’-]{2,}")
# Words that say how to compute or phrase the answer, never what a table holds.
_QUESTION_NOISE = set(
    "a an the of for in on at to and or by with from as is are was were be been "
    "that this these those what which who whom how many much per its their our "
    "your not no than then into over under between there here were did does do "
    "total average count counts correlation proportion percentage share rank "
    "distinct most largest smallest highest lowest top each combined since "
    "increase decrease change difference number place places placed held "
    "january february march april may june july august september october "
    "november december".split()
)


def _first_sentence(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", flat)
    sentence = match.group(1) if match else flat
    return sentence if len(sentence) <= limit else sentence[: limit - 3].rstrip() + "..."


def _strip_board(text: str) -> str:
    index = text.find(BOARD_MARKER)
    return text if index < 0 else text[:index]


def _numbers_as_ranges(numbers: list[int]) -> str:
    parts: list[str] = []
    for number in sorted(numbers):
        if parts and parts[-1][1] == number - 1:
            parts[-1] = (parts[-1][0], number)
        else:
            parts.append((number, number))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def _mentions(text: str, word: str) -> bool:
    """A question word occurs in ``text`` as a word start. Long words keep only
    their first letters, so "licences" also finds "Licensing"."""
    stem = word[: max(4, len(word) - 3)] if len(word) >= 6 else word.rstrip("s")
    return bool(re.search(rf"(?<![a-z]){re.escape(stem)}", text))


class DiscoveryContext:
    """Records what each tool call returned and edits the agent's history."""

    def __init__(self, manager: Phase12ToolsManager) -> None:
        self.manager = manager
        # Tool output text -> what it is, so an edit can tell a listing from a
        # profile without re-parsing: "listing", "profile", "blocked_confirm",
        # "evidence" (join checks) or "note" (refusals and other short replies).
        self.kinds: dict[str, str] = {}

    @property
    def enabled(self) -> bool:
        discovery = self.manager.discovery
        return discovery.state_board or discovery.compact_history

    # ------------------------------------------------------------ recording

    def record(self, tool_name: str, text: str, *, error: bool = False) -> None:
        if tool_name == "check_join_union":
            kind = "evidence"
        elif tool_name == "confirm_unified_selection" and error:
            kind = "blocked_confirm"
        elif _LISTING_ENTRY.search(text):
            kind = "listing"
        elif tool_name == "inspect_columns" and (
            text.startswith("Schema for ") or text.startswith("Cached inspection (attempt 1/")
        ):
            kind = "profile"
        else:
            kind = "note"
        self.kinds[text] = kind

    # --------------------------------------------------------------- editing

    def edit(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Return the history the model should see next (see module doc)."""
        discovery = self.manager.discovery
        tool_positions = [
            index for index, message in enumerate(messages)
            if message.role == MessageRole.TOOL
        ]
        if not tool_positions:
            return messages
        last = tool_positions[-1]
        edited = list(messages)
        inspected_later = False
        blocked_later = False
        for index in reversed(tool_positions):
            original = messages[index].content or ""
            text = _strip_board(original)
            kind = self.kinds.get(text)
            if discovery.compact_history and index != last and not text.startswith(COMPACTED):
                if kind == "listing" and inspected_later:
                    text = self._compact_listing(text)
                elif kind == "blocked_confirm" and blocked_later:
                    text = COMPACTED + "Earlier confirmation, superseded: " + _first_sentence(
                        text.removeprefix(REPEATED_BLOCK)
                    )
                elif kind == "note" and len(text) > _MIN_COMPACTABLE_NOTE:
                    text = COMPACTED + _first_sentence(text)
            if kind == "profile":
                inspected_later = True
            elif kind == "blocked_confirm":
                blocked_later = True
            if discovery.state_board and index == last and last == len(messages) - 1:
                text += "\n\n" + self.render_board()
            if text != original:
                edited[index] = messages[index].model_copy(
                    update={"blocks": [TextBlock(text=text)]}
                )
        return edited

    def _compact_listing(self, text: str) -> str:
        state = self.manager.state
        visible = state.all_candidates[: state.visible_candidate_count]
        header = text.split("Candidates in retrieval order", 1)[0]
        header = header.split("\nRevealed", 1)[0]
        header = " ".join(header.split())[:300]
        lines = [
            COMPACTED + (header + " " if header else "")
            + "Full previews were shown earlier; one line per candidate:"
        ]
        # Inspected this round, so the profile really is in the history; a
        # carried table's cached profile is summarized by the board instead.
        inspected = set(state.inspection_counts)
        for match in _LISTING_ENTRY.finditer(text):
            number = int(match.group(1))
            if not 1 <= number <= len(visible):
                continue
            name = visible[number - 1]
            meta = state.solr_meta.get(name, {})
            title = " ".join(str(meta.get("title") or "Unknown").split())[:80]
            if name.casefold() in inspected:
                lines.append(f"Candidate {number}: {title} -- inspected, profile below")
                continue
            description = " ".join(str(meta.get("description") or "").split())[:110]
            columns = [str(column)[:30] for column in meta.get("columns.name", [])][:6]
            lines.append(
                f"Candidate {number}: {title}"
                + (f" -- {description}" if description else "")
                + (f" | columns: {', '.join(columns)}" if columns else "")
            )
        return "\n".join(lines)

    # ---------------------------------------------------------------- board

    def render_board(self) -> str:
        manager = self.manager
        state, discovery = manager.state, manager.discovery
        visible = state.all_candidates[: state.visible_candidate_count]
        lines = [
            "[DISCOVERY STATE -- kept current by the system after every tool "
            "call; it replaces every earlier state block]"
        ]
        periods = question_periods(manager.question)
        words = [
            word for word in dict.fromkeys(_WORD.findall(manager.question.casefold()))
            if word not in _QUESTION_NOISE
        ]

        searches = [
            f"[{', '.join(map(str, attempt.get('keywords') or [])) or 'question'}]"
            f" -> {len(attempt.get('current_candidates') or [])} local candidates"
            for attempt in state.search_attempts[-4:]
        ]
        if searches:
            lines.append("Searches: " + "; ".join(searches))
        if state.failed_keyword_combinations:
            banned = [
                "{" + ", ".join(sorted(item)) + "}"
                for item in state.failed_keyword_combinations[:6]
            ]
            lines.append(
                "Zero-result word sets (any search containing one is refused): "
                + " ".join(banned)
            )
        if state.banned_table_keyword_combinations:
            banned = [
                "{" + ", ".join(sorted(item)) + "}"
                for item in state.banned_table_keyword_combinations[:6]
            ]
            lines.append(
                "Word sets that found only banned tables (refused too): "
                + " ".join(banned)
            )

        inspected = {name.casefold() for name in state.inspected_candidates()}
        carried = {name.casefold() for name in state.carried_tables}
        uninspected: list[int] = []
        for number, name in enumerate(visible, 1):
            if name.casefold() not in inspected:
                uninspected.append(number)
                continue
            lines.append(self._candidate_line(number, name, periods, words, carried))
        if uninspected:
            lines.append("Not yet inspected: Candidates " + _numbers_as_ranges(uninspected))

        used = len(state.inspection_counts)
        if state.expansion_count == 0:
            limit = (
                f"{used} of {discovery.initial_shortlist_size} used"
                f" ({discovery.max_inspected_candidates} after expand_candidates)"
            )
        else:
            limit = f"{used} of {discovery.max_inspected_candidates} used"
        expansions_left = discovery.max_expansions - state.expansion_count
        lines.append(
            f"Budgets: inspections {limit}; expand_candidates "
            + (f"{expansions_left} left" if expansions_left > 0 else "spent")
            + "; search_tables "
            + ("available" if manager.is_tool_available("search_tables") else "spent")
        )
        if state.last_confirm_error:
            lines.append(
                f"Last confirm_unified_selection was blocked"
                f" ({state.confirm_blocks} blocked so far): "
                + " ".join(state.last_confirm_error.split())[:400]
            )
        lines.append(
            "Tools still worth calling: "
            + ", ".join(name for name in manager.tool_names() if manager.is_tool_available(name))
        )
        return "\n".join(lines)

    def _candidate_line(
        self,
        number: int,
        name: str,
        periods: list[str],
        words: list[str],
        carried: set[str],
    ) -> str:
        state = self.manager.state
        meta = state.solr_meta.get(name, {})
        inspection = str(state.inspection_cache.get(name.casefold(), ""))
        title = " ".join(str(meta.get("title") or "Unknown").split())[:60]
        facts: list[str] = []
        rows = re.search(r"^Rows: (.+)$", inspection, re.MULTILINE)
        if rows:
            facts.append(f"{rows.group(1)} rows")
        for period in periods:
            evidence = inspected_period_evidence(inspection, period)
            if evidence is None:
                facts.append(f"{period}: not in any inspected value or date range")
            elif evidence[1] == "values":
                facts.append(f"{period}: in {evidence[0]} values")
            else:
                facts.append(f"{period}: within {evidence[0]} date range")
        metadata_text = " ".join(
            str(part) for part in (
                meta.get("title"), meta.get("description"),
                " ".join(map(str, meta.get("tags") or [])), meta.get("publisher"),
            ) if part
        ).casefold()
        sampled = {
            match.group(1): match.group(2).casefold()
            for match in re.finditer(
                r"^- (.+?) \(Category sample\): (.*)$", inspection, re.MULTILINE
            )
        }
        column_names = " ".join(
            re.findall(r"^- (.+?)(?: \(|: )", inspection, re.MULTILINE)
        ).casefold()
        in_values = []
        unmatched = []
        for word in words:
            column = next(
                (column for column, values in sampled.items() if _mentions(values, word)),
                None,
            )
            if column is not None:
                in_values.append(f"{word} in {column}")
            elif not (_mentions(metadata_text, word) or _mentions(column_names, word)):
                unmatched.append(word)
        if in_values:
            facts.append("question words in sampled values: " + ", ".join(in_values[:5]))
        if unmatched:
            facts.append("question words found nowhere: " + ", ".join(unmatched[:6]))
        origin = " (kept from the previous attempt)" if name.casefold() in carried else ""
        return f"- Candidate {number} \"{title}\"{origin}: " + "; ".join(facts)
