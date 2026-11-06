"""The MemGPT selector's loop, its tools and their gates.

The agent does not run inside the production agent runner. Each turn compiles one fresh request
(the read-only block as the system message; the candidates, the summary and FIFO of recent turns,
and STATE as the user message), asks the model for one tool call, executes it against ``Memory``
and appends the exchange to the FIFO. Nothing depends on a provider's tool-history format, and a
repeated call is answered from STATE instead of ending the run.

Gates (the fix for early stoppage):
  confirm  refused while the selection is empty, while a requirement is open, or while a visible
           uninspected candidate is still queued in STATE; the refusal names the candidates
  reject   refused while the selection covers every requirement, while an inspected table shows
           an open item (a partition to add), or while a queued or hidden candidate still matches
           an open requirement
Both gates open when the actions are spent or in decision mode (a third identical call, or the
last two turns): then only confirm and reject are offered and confirm needs only a non-empty
selection.

What inspect shows (``evidence``): the shared profile with what the whole table holds -- years kept
as integers, numeric ranges, the most frequent text values, the step of the dates, and the
question's words found in the values; named items and periods are scanned in the text values of
every inspected table. A requirement that names nothing but the portal's own place ("location:
Valencia") is waived when the requirements are defined: every table of the portal covers it, and
no card could show it.

The system writes the plan: requirement_coverage from the select calls (with the columns of the
table the requirement's words match), the combination strategy from the measured relationships,
and the production ledger's complaints as advisories instead of blocks.
"""

from __future__ import annotations

import ast
import inspect
import io
import json
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

from lakegen.config.registry import catalogue_note
from lakegen.datastructures.errors import WorkflowCancelled
from lakegen.datastructures.types import ToolCall, ToolCallback
from lakegen.pipeline.discovery.selectors.memgpt import display
from lakegen.pipeline.discovery.selectors.memgpt.evidence import MAX_WORDS, evidence_profile
from lakegen.pipeline.discovery.selectors.memgpt.items import named_items as listed_items
from lakegen.pipeline.discovery.selectors.memgpt.items import question_periods, scan_values
from lakegen.pipeline.discovery.selectors.memgpt.memory import Exchange, Memory, Requirement
from lakegen.pipeline.discovery.selectors.memgpt.tables import relationship, table_profile
from lakegen.pipeline.discovery.selectors.memgpt.text import (
    content_words,
    mentions,
    only_place,
    portal_places,
)
from lakegen.pipeline.evidence.requirement_ledger import (
    build_requirement_ledger,
    requirement_ledger_blockers,
)
from lakegen.pipeline.evidence.table_inspection import (
    _rank_for_missing_requirements,
    _requirement_terms,
)

INITIAL_VISIBLE = 10
EXPANSION_SIZE = 5
MAX_EXPANSIONS = 2
INSPECT_BATCH = 4
ACTIONS = 12
MAX_TURNS = 30
RESULT_CHARS = 9000
MAX_CARRIED = 8
"""Inspected tables a rejection carries to the next attempt (the judge it replaced carried at most 8)."""
TERMINAL = ("confirm", "reject")
PROMPT = "memgpt_selector"

# The model reads each tool method's ``__doc__`` as the tool's description (``MemGPTAgent.tools``; the
# experiments' agents copy or extend it). These two descriptions are longer than one line, so they are
# kept here, byte for byte, and set as the methods' ``__doc__`` after the class.
DEFINE_REQUIREMENTS_TOOL = (
    'Define what the question needs from the data, one item each as "<kind>: <text>" (kinds: subject, measure,\n'
    "dimension, filter, period, relationship), and list in named_items every entity, dataset or category the question\n"
    "asks about one by one (each sensor, gate, party list, dataset, staff category...), exactly as the question writes\n"
    "it; [] when it lists none. First call, once. Each named item (and each period, when the question names several)\n"
    "becomes its own requirement."
)
SELECT_TOOL = (
    'Add an inspected table to the selection with the requirement ids it covers, e.g. table=3, covers=["R1", "R3"],\n'
    'note="R3 by subject: the dataset is the 2024 Olympics routes".'
)


# Which tool a reply's plain-JSON text was meant for, by the keys it carries (most specific first).
RECOVERY = (
    ("define_requirements", {"requirements"}),
    ("select", {"table", "covers"}),
    ("waive", {"requirement", "reason"}),
    ("unselect", {"table", "reason"}),
    ("inspect", {"candidates"}),
    ("expand", {"missing"}),
    ("reject", {"reasoning", "suggestion"}),
    ("recall", {"table"}),
    ("confirm", {"reasoning"}),
)


def recover_call(content: str, offered: list[str]) -> tuple[str, dict] | None:
    """A tool call the model wrote as text, in any of the shapes seen.

    The shapes: ``tool(key=value, ...)``, the first JSON object of the reply as
    ``{"name": tool, "arguments": {...}}``, ``{"<tool>": {...}}`` or the
    arguments alone matched to a tool by their keys.

    Args:
        content: The reply's text.
        offered: The tools offered this turn.

    Returns:
        The tool's name and its arguments, or None when the text holds no call to an offered tool.
    """
    text = str(content or "")
    call = _python_call(text, offered)
    if call is not None:
        return call
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
        return (obj["name"], obj["arguments"]) if obj["name"] in offered else None
    if len(obj) == 1:
        ((name, args),) = obj.items()
        name = str(name).replace("functions.", "")
        if name in offered and isinstance(args, dict):
            return name, args
    keys = set(obj)
    for name, required in RECOVERY:
        if required <= keys and name in offered:
            return name, obj
    return None


_CALL = re.compile(r"(?:^|[\s`*])(?:functions\.)?([a-z_]+)\s*\(")


def _python_call(text: str, offered: list[str]) -> tuple[str, dict] | None:
    """``tool(key=value, ...)`` written as text: the keyword arguments as Python literals."""
    for m in _CALL.finditer(text):
        name = m.group(1)
        if name not in offered:
            continue
        end = _closing(text, m.end() - 1)
        if end is None:
            continue
        try:
            node = ast.parse(text[m.start(1) : end + 1].strip(), mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(node, ast.Call) or node.args:
            continue
        try:
            args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
        except ValueError:
            continue
        return name, args
    return None


def _closing(text: str, open_at: int) -> int | None:
    """The index of the bracket closing ``text[open_at]``, string literals skipped; None when it never closes."""
    depth, quote, i = 0, "", open_at
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _numbers(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (str, int)):
        values = re.findall(r"\d+", str(values)) if isinstance(values, str) else [values]
    out = []
    for v in values:
        m = re.search(r"\d+", str(v))
        if m and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


def _flatten(requirements: Any) -> list[str]:
    """The requirements as "<kind>: <text>" lines, however the model wrote them."""
    if isinstance(requirements, str):
        requirements = [line for line in re.split(r"[\n;]+", requirements) if line.strip()]
    flat = []
    for item in requirements or []:
        if isinstance(item, dict):
            flat.extend(f"{k}: {v}" for k, v in item.items() if str(v).strip())
        else:
            flat.append(str(item))
    return flat


class MemGPTAgent:
    """One selection: the memory, the budgets, the tools and the loop that drives the model.

    Each constructor argument is kept as the attribute of the same name.

    Attributes:
        evidence_profiles: inspect shows what the whole table holds (``evidence``); off, the shared
            profile alone (before 2026-10-22).
        tool_choice: Ask a turn that produced no tool call again at once with ``tool_choice="required"``,
            when the client declares support (``llm.tool_choice_supported``). The first request of every
            turn stays unconstrained, so the model reasons freely; the required retry only has to write
            the call (the endpoint decodes it under the tool grammar, without reasoning). Off, the loop
            before 2026-10-24: a note asks the model again on the next turn.
        waive_portal_place: Waive the requirements that name only the portal's own place; off, they need
            a table that shows it.
        queue_gate: confirm is refused while STATE queues an uninspected candidate that matches the
            requirements as well as the selection (an alternative never looked at); off, only an open
            requirement refuses it. Measured in experiments/coverage_lab (decision2): the 73 refusals of
            this kind led to inspections that changed no selection.
        memory_cls: The memory the agent keeps (an experiment's variant subclasses it).
        memory: The selection's working memory.
        actions: Actions spent (inspections and expansions).
        expansions: Expansions spent.
        turn: The current turn.
        decision_mode: Whether only confirm and reject are offered.
        outcome: ``confirmed``, ``rejected``, ``max_turns``, ``error: ...``, or empty while running.
        final_reasoning: The reasoning of the confirmation or rejection.
        rejection: The reasoning and the suggestion of a rejection.
        trace: The run's activity log.
        items_added: Requirements the system added for named items and periods.
        signatures: Times each identical call was made.
        calls: Calls per tool.
        refusals: Refused confirm, reject and other calls.
        cover_refusals: Selects refused for items the table does not show.
        items_scanned: (candidate, requirement id) pairs whose text values were scanned.
        no_call_turns: Turns without a tool call.
        recovered_calls: Calls recovered from the reply's text.
        provider_errors: Consecutive provider errors.
        constrained_retries: Turns asked again with a tool call required.
        constrained_calls: How many of those retries produced one.

    Args:
        question: The user's question.
        candidates: The retrieved tables, in retrieval order.
        metadata: Catalogue metadata by table file.
        table_dir: Folder of the lake's tables.
        llm: The run's LLM.
        prompts: The run's prompt manager.
        portal: Name of the open-data portal.
        lake: The lake's name, for the portal's catalogue note.
        keywords: The keywords retrieval ran with.
        hint: Feedback from the previous attempt.
        question_only: Whether retrieval ranked the candidates by the question alone.
        emit: Receives the live lines of the run, if any.
        cancel_check: Raises when the run is cancelled.
        tool_callback: Receives each tool call, for a surface that shows them as steps.
    """

    evidence_profiles = True
    tool_choice = True
    waive_portal_place = True
    queue_gate = True
    memory_cls: type[Memory] = Memory

    def __init__(
        self,
        *,
        question: str,
        candidates: Sequence[str],
        metadata: dict[str, dict[str, Any]],
        table_dir: Path,
        llm: Any,
        prompts: Any,
        portal: str = "",
        lake: str = "",
        keywords: Sequence[str] = (),
        hint: str = "",
        question_only: bool = False,
        emit: Callable[[str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        tool_callback: ToolCallback | None = None,
    ) -> None:
        self.question, self.table_dir, self.llm, self.prompts = (
            question,
            Path(table_dir),
            llm,
            prompts,
        )
        self.portal, self.lake, self.keywords = portal, lake, list(keywords)
        self.hint, self.question_only = str(hint or "").strip(), question_only
        self.emit, self.cancel_check, self.tool_callback = emit, cancel_check, tool_callback
        self.candidates = list(candidates)
        self.metadata = metadata
        self.memory = self.memory_cls(
            question, self.candidates, metadata, min(INITIAL_VISIBLE, len(self.candidates))
        )
        self.cover_refusals = 0
        self.items_added: list[str] = []
        self.actions = 0
        self.expansions = 0
        self.turn = 0
        self.decision_mode = False
        self.signatures: dict[str, int] = {}
        self.calls: dict[str, int] = {}
        self.refusals: dict[str, int] = {"confirm": 0, "reject": 0, "other": 0}
        self.outcome = ""
        self.final_reasoning = ""
        self.rejection = ("", "")
        self.trace = io.StringIO()
        self.items_scanned: set[tuple[int, str]] = set()
        self.no_call_turns = 0
        self.recovered_calls = 0
        self.provider_errors = 0
        self.constrained_retries = 0
        self.constrained_calls = 0

    # ---------------------------------------------------------------- helpers

    @property
    def actions_left(self) -> int:
        """Actions still to spend."""
        return ACTIONS - self.actions

    def available(self) -> list[str]:
        """The tools offered now."""
        if self.decision_mode:
            return list(TERMINAL)
        if not self.memory.requirements:
            return ["define_requirements"]
        names = ["select", "unselect", "waive", "recall", *TERMINAL]
        if self.actions_left > 0:
            if any(n not in self.memory.cards for n in range(1, self.memory.visible + 1)):
                names.insert(0, "inspect")
            if self.expansions < MAX_EXPANSIONS and self.memory.visible < len(self.candidates):
                names.insert(1, "expand")
        return names

    def status(self) -> list[str]:
        """STATE's closing lines: the budgets, the tools offered and the next step."""
        return [
            f"Budget: actions {self.actions} of {ACTIONS} used; turn {self.turn} of {MAX_TURNS}; expansions "
            f"{self.expansions} of {MAX_EXPANSIONS}; {len(self.memory.cards)} tables inspected of {len(self.candidates)}"
            + (" -- DECISION MODE: only confirm or reject" if self.decision_mode else ""),
            "Tools offered now: " + ", ".join(self.available()),
            "Next step: " + self.next_step(),
        ]

    def next_step(self) -> str:
        """What STATE tells the agent to do next."""
        m = self.memory
        if not m.requirements:
            return "define_requirements from the question"
        if self.decision_mode:
            return "confirm the selection, or reject if it is empty and nothing covers the question"
        queue = m.queue()
        opened = m.open_requirements()
        if queue and self.actions_left > 0:
            return f"inspect({[n for n, _ in queue[:INSPECT_BATCH]]})"
        for r in opened:
            showing = [
                n for n in m.inspected() if r.id in m.cards[n].shows and n not in m.selection
            ]
            if showing:
                return f"select an inspected table that covers {r.id}: candidates {showing[:4]}"
        if (
            opened
            and m.hidden_matches()
            and self.expansions < MAX_EXPANSIONS
            and self.actions_left > 0
        ):
            return f"expand('{opened[0].text[:50]}')"
        if opened:
            return f"{', '.join(r.id for r in opened)} open and no candidate left mentions them: select a table covering them by subject, waive with a reason, or reject"
        if m.selection:
            return "every requirement is covered: confirm"
        return "select the inspected table(s) covering the requirements"

    def _profile(self, number: int) -> str:
        name = self.candidates[number - 1]
        if not self.evidence_profiles:
            return table_profile(self.table_dir, name)
        return evidence_profile(self.table_dir, name, self.search_words())

    def search_words(self) -> list[str]:
        """The words an evidence profile looks for in the values: the question's, then the requirements'."""
        words = content_words(self.question)
        for r in self.memory.requirements:
            words += content_words(r.text)
        return list(dict.fromkeys(words))[:MAX_WORDS]

    def _scan_items(self) -> None:
        """Scan the inspected tables' text values for the named items and periods, as select does."""
        m = self.memory
        for number in m.inspected():
            changed = False
            for r in m.requirements:
                if (
                    r.item is None
                    or (number, r.id) in self.items_scanned
                    or m.item_evidence(number, r)
                ):
                    continue
                self.items_scanned.add((number, r.id))
                found = scan_values(self.table_dir / self.candidates[number - 1].strip(), r.item)
                if found:
                    m.scanned[(number, r.id)] = found
                    changed = True
            if changed:
                m.refresh_shows(number)

    def _waive_portal_place(self) -> list[str]:
        """Waive the requirements that name only the portal's own place; their ids."""
        places = portal_places(self.portal)
        waived = []
        for r in self.memory.requirements:
            if places and not r.waived and only_place(r.text, places):
                r.item = None
                r.waived = f"the portal's own place: every table of the {self.portal} covers it"
                waived.append(r.id)
        return waived

    # ---------------------------------------------------------------- tools

    def define_requirements(
        self, requirements: list[str], named_items: list[str] | None = None
    ) -> str:
        """The define_requirements tool: fix the requirements, once; named items and periods get their own.

        The model reads ``DEFINE_REQUIREMENTS_TOOL`` as this tool's description.

        Args:
            requirements: The requirements as "<kind>: <text>", however the model wrote them.
            named_items: The entities, datasets or categories the question asks about one by one.

        Returns:
            The tool's result for the model.
        """
        if self.memory.requirements:
            return "The requirements are already defined (read-only)."
        items = [*question_periods(self.question), *listed_items(self.question, named_items)]
        self.items_added = self.memory.define(_flatten(requirements), items)
        if not self.memory.requirements:
            return "Error: give at least one requirement as '<kind>: <text>'."
        text = "Requirements fixed: " + "; ".join(r.label for r in self.memory.requirements)
        tracked = [r.id for r in self.memory.requirements if r.item is not None]
        if tracked:
            text += (
                f". {', '.join(tracked)} are items the question names one by one: each needs a table that shows it "
                "(in its title, description, dates or values)"
            )
        if self.items_added:
            text += f"; the system added {len(self.items_added)} of them because no requirement named them on its own"
        text += "."
        waived = self._waive_portal_place() if self.waive_portal_place else []
        if waived:
            text += (
                f" The system waived {', '.join(waived)}: the portal's own place, which every table of the "
                f"{self.portal} covers."
            )
        return text

    def inspect(self, candidates: list[int]) -> str:
        """Profile up to 4 visible candidates by number, e.g. [2, 5, 7]: rows, columns and types, dates, sample values."""
        numbers = _numbers(candidates)
        if not numbers:
            return "Error: pass candidate numbers, e.g. [1, 3]."
        fresh = [n for n in numbers if 1 <= n <= self.memory.visible and n not in self.memory.cards]
        already = [n for n in numbers if n in self.memory.cards]
        hidden = [n for n in numbers if not 1 <= n <= self.memory.visible]
        parts = []
        if already:
            parts.append(
                f"Already inspected (no action spent; their cards are in STATE): {already}."
            )
        if hidden:
            parts.append(f"Not visible: {hidden} (visible: 1-{self.memory.visible}).")
        if not fresh:
            return " ".join(parts) + " " + self.next_step() + "."
        if self.actions_left <= 0:
            return "No actions left: decide with the inspected tables."
        self.actions += 1
        if len(fresh) > INSPECT_BATCH:
            parts.append(
                f"Only the first {INSPECT_BATCH} were profiled: {fresh[INSPECT_BATCH:]} wait."
            )
            fresh = fresh[:INSPECT_BATCH]
        with ThreadPoolExecutor(max_workers=4) as pool:
            profiles = dict(zip(fresh, pool.map(self._profile, fresh), strict=True))
        for n, profile in profiles.items():
            self.memory.record_profile(n, profile)
            parts.append(f"[#{n}]\n{profile}")
        if self.evidence_profiles:
            self._scan_items()
        return "\n\n".join(parts)

    def expand(self, missing: str) -> str:
        """Reveal the next 5 hidden candidates, those whose catalogue entry best matches the missing requirement first."""
        if self.expansions >= MAX_EXPANSIONS:
            return "No expansions left."
        if self.memory.visible >= len(self.candidates):
            return "Every candidate is already visible."
        if self.actions_left <= 0:
            return "No actions left."
        self.actions += 1
        terms = _requirement_terms(missing) or _requirement_terms(
            " ".join(r.text for r in self.memory.open_requirements())
        )
        start = self.memory.visible
        hidden = self.candidates[start:]
        matched = (
            _rank_for_missing_requirements(hidden, self.metadata, terms, EXPANSION_SIZE)
            if terms
            else []
        )
        revealed = [*matched, *(c for c in hidden if c not in matched)][:EXPANSION_SIZE]
        rest = [c for c in hidden if c not in revealed]
        self.candidates[start:] = [*revealed, *rest]
        self.memory.candidates = self.candidates
        self.memory.visible = start + len(revealed)
        self.expansions += 1
        return (
            f"Revealed #{start + 1}-#{self.memory.visible} ({len(matched)} matched '{', '.join(terms[:5])}'):\n"
            + "\n".join(
                f"#{n} {self.memory.title(n)[:100]}"
                for n in range(start + 1, self.memory.visible + 1)
            )
        )

    def _measure_with_selection(self, number: int) -> list[str]:
        others = [n for n in self.memory.selection if n != number][:5]
        if not others:
            return []

        def one(other):
            a, b = sorted((number, other))
            return (a, b), relationship(
                self.table_dir, self.candidates[a - 1], self.candidates[b - 1]
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            measured = dict(pool.map(one, others))
        self.memory.relations.update(measured)
        return [f"#{a}-#{b}: {s}" for (a, b), s in measured.items()]

    def _split_covers(self, number: int, covers: Any) -> tuple[list[str], list[str], list[str]]:
        """The requirement ids a select claims: accepted, unknown, and refused (items the table does not show)."""
        if isinstance(covers, str):
            covers = re.findall(r"R?\d+", covers, re.IGNORECASE)
        rids, unknown = [], []
        for ref in covers or []:
            r = self.memory.requirement(ref)
            (rids if r else unknown).append(r.id if r else str(ref))
        kept, refused = [], []
        for rid in rids:
            r = self.memory.requirement(rid)
            (kept if r.item is None or self._item_shown(number, r) else refused).append(rid)
        if refused:
            self.cover_refusals += 1
        return kept, unknown, refused

    def select(self, table: int, covers: list[str], note: str = "") -> str:
        """The select tool: add an inspected table to the selection with the requirement ids it covers.

        The model reads ``SELECT_TOOL`` as this tool's description.

        Args:
            table: The candidate's number.
            covers: The requirement ids the table covers.
            note: Why, for a requirement the table's data does not show.

        Returns:
            The tool's result for the model.
        """
        number = (_numbers(table) or [0])[0]
        if number not in self.memory.cards:
            return f"#{number} is not inspected: inspect it first."
        rids, unknown, refused = self._split_covers(number, covers)
        if not rids:
            if refused:
                return self._refuse(self._cover_refusal(number, refused))
            return 'Error: name the requirement ids this table covers, e.g. covers=["R1", "R2"].'
        previous = self.memory.selection.get(number, [])
        self.memory.selection[number] = sorted(set(previous) | set(rids), key=lambda x: int(x[1:]))
        self.memory.unselected.pop(number, None)
        card = self.memory.cards[number]
        unverified = [rid for rid in rids if rid not in card.shows]
        lines = [f"#{number} selected, covering {', '.join(self.memory.selection[number])}."]
        if unverified:
            lines.append(
                f"No column, value, date or title of #{number} mentions {', '.join(unverified)}; kept on your word"
                + (f" ({note[:120]})" if note else "")
                + "."
            )
        if unknown:
            lines.append(f"Unknown requirement ids ignored: {unknown}.")
        if refused:
            lines.append(self._cover_refusal(number, refused))
        relations = self._measure_with_selection(number)
        if relations:
            lines.append("Measured against the selection: " + " | ".join(relations))
        return " ".join(lines[:3]) + ("\n" + "\n".join(lines[3:]) if len(lines) > 3 else "")

    def _item_shown(self, number: int, r: Requirement) -> bool:
        """Whether the table shows the item: catalogue, profile, or (scanned once) its text values.

        An item no candidate's catalogue mentions cannot be checked from
        metadata and is accepted on the agent's word.

        Args:
            number: The candidate's number.
            r: The item's requirement.

        Returns:
            True when the table shows the item or no candidate can be checked for it.
        """
        m = self.memory
        if m.item_evidence(number, r) or (number, r.id) in m.scanned:
            return True
        found = scan_values(self.table_dir / self.candidates[number - 1].strip(), r.item)
        if found:
            m.scanned[(number, r.id)] = found
            m.refresh_shows(number)
            return True
        return not any(m.item_evidence(n, r) for n in range(1, len(self.candidates) + 1))

    def _cover_refusal(self, number: int, rids: list[str]) -> str:
        m = self.memory
        parts = []
        for rid in rids:
            r = m.requirement(rid)
            showing = [
                n
                for n in range(1, len(self.candidates) + 1)
                if n != number and m.item_evidence(n, r)
            ]
            where = ", ".join(f"#{n}" if n <= m.visible else f"#{n} (hidden)" for n in showing[:6])
            parts.append(f"{rid} ({r.item.label})" + (f" -- shown by {where}" if where else ""))
        return (
            f"Not accepted: #{number} does not show "
            + "; ".join(parts)
            + " (not in its title, description, dates or values). Select the table that shows each of them."
        )

    def unselect(self, table: int, reason: str) -> str:
        """Remove a table from the selection, with the reason."""
        number = (_numbers(table) or [0])[0]
        if number not in self.memory.selection:
            return f"#{number} is not in the selection."
        self.memory.selection.pop(number)
        self.memory.unselected[number] = str(reason)
        self.memory.relations = {k: v for k, v in self.memory.relations.items() if number not in k}
        return f"#{number} removed from the selection."

    def waive(self, requirement: str, reason: str) -> str:
        """Mark a requirement as needing no data (a computation, a formatting detail), with the reason."""
        r = self.memory.requirement(requirement)
        if r is None:
            return f"Unknown requirement {requirement!r}."
        if r.item is not None:
            return f"{r.id} is an item the question names ({r.item.label}): it needs data and cannot be waived."
        if len(str(reason).strip()) < 10:
            return "Give the reason this requirement needs no data."
        r.waived = " ".join(str(reason).split())[:200]
        return f"{r.id} waived."

    def recall(self, table: int) -> str:
        """Re-read the full profile of an inspected table."""
        number = (_numbers(table) or [0])[0]
        if number not in self.memory.profiles:
            return f"#{number} is not inspected."
        return f"Full profile of #{number}:\n{self.memory.profiles[number]}"

    def confirm(self, reasoning: str) -> str:
        """Confirm the selection in STATE (no table list: the selection is what gets confirmed)."""
        m = self.memory
        if not m.selection:
            self.refusals["confirm"] += 1
            return self._refuse(
                "Confirm refused: the selection is empty. select the inspected tables that cover the requirements, or reject."
            )
        if not self.decision_mode and self.actions_left > 0:
            opened = m.open_requirements()
            queue = m.queue()
            if opened:
                self.refusals["confirm"] += 1
                hint = (
                    f"inspect {[n for n, _ in queue[:INSPECT_BATCH]]}"
                    if queue
                    else "expand"
                    if m.hidden_matches() and self.expansions < MAX_EXPANSIONS
                    else "select a table covering them (by column, value, date or subject) or waive them"
                )
                return self._refuse(
                    f"Confirm refused: {', '.join(r.id for r in opened)} still open. Next: {hint}."
                )
            if queue and self.queue_gate:
                self.refusals["confirm"] += 1
                return self._refuse(
                    "Confirm refused: the catalogue of "
                    + ", ".join(f"#{n}" for n, _ in queue[:INSPECT_BATCH])
                    + " matches the requirements as well as your selection, and they are not inspected. Inspect them, "
                    "then keep the better tables."
                )
        self.outcome = "confirmed"
        self.final_reasoning = str(reasoning)
        return "CONFIRMED"

    def reject(self, reasoning: str, suggestion: str = "") -> str:
        """Reject: no candidate can cover an essential requirement."""
        m = self.memory
        if not self.decision_mode and self.actions_left > 0:
            opened = m.open_requirements()
            if m.selection and not opened:
                self.refusals["reject"] += 1
                return self._refuse(
                    "Reject refused: your selection covers every requirement. Confirm it, or unselect what is wrong."
                )
            pending = [n for n, why in m.queue() if any(r.id in why for r in opened)]
            partitions = [n for n, _ in m.partitions_to_add()]
            if opened and partitions:
                self.refusals["reject"] += 1
                return self._refuse(
                    f"Reject refused: {', '.join(r.id for r in opened)} open, and the inspected "
                    + ", ".join(f"#{n}" for n in partitions[:6])
                    + " show them: select them for those requirements."
                )
            if opened and (pending or (m.hidden_matches() and self.expansions < MAX_EXPANSIONS)):
                self.refusals["reject"] += 1
                where = (
                    f"inspect {pending[:INSPECT_BATCH]}"
                    if pending
                    else "expand: hidden candidates match them"
                )
                return self._refuse(
                    f"Reject refused: {', '.join(r.id for r in opened)} may still be covered. {where}."
                )
        self.outcome = "rejected"
        self.rejection = (str(reasoning), str(suggestion))
        self.final_reasoning = f"{reasoning} Suggestion: {suggestion}"
        return "REJECTED"

    def _refuse(self, text: str) -> str:
        self.memory.last_refusal = text
        return text

    # ---------------------------------------------------------------- what the selection hands on

    def selected(self) -> list[str]:
        """The selected table files, in candidate order."""
        m = self.memory
        return [m.cards[n].name for n in sorted(m.selection)]

    def fallback(self) -> list[str]:
        """The selection when the run stopped before confirming; else the inspected table showing the most requirements."""
        m = self.memory
        if m.selection:
            return self.selected()
        ranked = sorted(m.inspected(), key=lambda n: (-len(m.cards[n].shows), n))
        return [m.cards[ranked[0]].name] if ranked else []

    def carried_tables(self) -> list[str]:
        """On a rejection: the selected tables, then the inspected ones showing the most requirements."""
        m = self.memory
        shown = sorted(
            (n for n in m.inspected() if n not in m.selection and m.cards[n].shows),
            key=lambda n: (-len(m.cards[n].shows), n),
        )
        return [m.cards[n].name for n in [*sorted(m.selection), *shown]][:MAX_CARRIED]

    def _coverage(self) -> dict[str, dict[str, Any]]:
        m = self.memory
        coverage: dict[str, dict[str, Any]] = {}
        for number, rids in sorted(m.selection.items()):
            card = m.cards[number]
            for rid in rids:
                r = m.requirement(rid)
                columns = [
                    c
                    for c in card.columns
                    if any(mentions(c.casefold(), w) for w in content_words(r.text))
                ][:3]
                coverage.setdefault(
                    r.text,
                    {
                        "table": card.name,
                        "columns": columns or card.columns[:1],
                        "by": card.shows.get(rid, "subject (agent)"),
                    },
                )
        return coverage

    def strategy(self) -> str:
        """How the selected tables combine, from the measured relationships."""
        rels = list(self.memory.relations.values())
        if len(self.memory.selection) == 1:
            return "single_table"
        if rels and all("UNION" in s and "NO UNION" not in s for s in rels):
            return "concat_partitions"
        if any(s.startswith("JOIN") for s in rels):
            return "join"
        return "unspecified"

    def plan(self) -> tuple[dict[str, Any], list[str]]:
        """The selection plan for the coder, in the judge's keys, and the ledger's complaints as advisories."""
        m = self.memory
        coverage = self._coverage()
        ledger = build_requirement_ledger(
            self.question,
            coverage,
            {},
            [],
            None,
            inspections={m.cards[n].name.casefold(): m.profiles[n] for n in m.cards},
        )
        plan = {
            "requirement_coverage": coverage,
            "table_roles": {
                m.cards[n].name: "covers "
                + "; ".join(f"[{r.kind}] {r.text}" for r in map(m.requirement, rids))
                for n, rids in sorted(m.selection.items())
            },
            "combination_strategy": self.strategy(),
            "uncovered_requirements": [r.text for r in m.open_requirements()],
            "alternatives_rejected": {
                m.cards[n].name: why for n, why in m.unselected.items() if n in m.cards
            },
            "requirement_ledger": ledger,
            "relationships": {
                f"{m.cards[a].name} + {m.cards[b].name}": s
                for (a, b), s in sorted(m.relations.items())
            },
            "waived": {r.text: r.waived for r in m.requirements if r.waived},
        }
        advisories = [
            f"The inspected data of the selection does not show: {request}"
            for request in requirement_ledger_blockers(ledger, self.selected())
        ]
        return plan, advisories

    # ---------------------------------------------------------------- the loop

    def functions(self) -> dict[str, Callable[..., str]]:
        """The tools by name, as bound methods."""
        return {
            "define_requirements": self.define_requirements,
            "inspect": self.inspect,
            "expand": self.expand,
            "select": self.select,
            "unselect": self.unselect,
            "waive": self.waive,
            "recall": self.recall,
            "confirm": self.confirm,
            "reject": self.reject,
        }

    def tools(self) -> dict[str, FunctionTool]:
        """The tools by name, each described to the model by its method's ``__doc__``."""
        return {
            name: FunctionTool.from_defaults(
                fn=fn, name=name, description=(fn.__doc__ or "").strip()
            )
            for name, fn in self.functions().items()
        }

    def system_message(self) -> str:
        """The read-only block: instructions, tools, question, retrieval notes and requirements."""
        return self.prompts.render(
            PROMPT,
            "system_prompt",
            batch=INSPECT_BATCH,
            max_expansions=MAX_EXPANSIONS,
            actions=ACTIONS,
            turns=MAX_TURNS,
            portal=self.portal,
            catalogue_notes=catalogue_note(self.lake),
            question=self.question,
            retrieval_notes=self.retrieval_notes(),
            requirements=self.memory.render_requirements(),
        )

    def retrieval_notes(self) -> str:
        """The read-only block's lines on retrieval: the keywords it ran with, the previous attempt's feedback."""
        lines = []
        if self.keywords:
            terms = " ".join(self.keywords)
            lines.append(
                f"Retrieval ranked the candidates by the question alone; these concepts were not searched: {terms}"
                if self.question_only
                else f"Retrieval concepts: {terms}"
            )
        if self.hint:
            lines.append(f"Feedback from the previous attempt: {self.hint}")
        return "\n".join(lines)

    def run(self) -> None:
        """Drive the model turn by turn until it confirms, rejects, or the turns run out."""
        tools = self.tools()
        note = ""
        while self.turn < MAX_TURNS and not self.outcome:
            if self.cancel_check is not None:
                self.cancel_check()
            self.turn += 1
            if self.turn >= MAX_TURNS - 1:
                self.decision_mode = True
            calls, note = self._ask(tools, note)
            if calls is None:
                break
            for name, args in calls:
                self._execute(tools, name, args)
                if self.outcome:
                    break
        if not self.outcome:
            self.outcome = "max_turns"

    def _ask(
        self, tools: dict[str, FunctionTool], note: str
    ) -> tuple[list[tuple[str, dict]] | None, str]:
        """One turn: the calls the reply makes and the note for the next turn (None: the provider keeps failing)."""
        offered = [tools[name] for name in self.available()]
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.system_message()),
            ChatMessage(
                role=MessageRole.USER, content=self.memory.render_turn(self.status(), note)
            ),
        ]
        try:
            calls, content = self._request(offered, messages, {})
            if not calls and self._constrains() and recover_call(content, self.available()) is None:
                self.constrained_retries += 1
                self._log(
                    f"no tool call: {content[:300]}\n-> asked again with a tool call required"
                )
                self.show(
                    display.note(
                        f"Turn {self.turn}: the model answered without a tool call; it is asked again with a tool call required."
                    )
                )
                calls, content = self._request(offered, messages, {"tool_choice": "required"})
                self.constrained_calls += bool(calls)
            self.provider_errors = 0
        except WorkflowCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- a provider error costs a turn and is shown to the model
            self._log(f"provider error: {str(exc)[:300]}")
            self.show(
                display.note(
                    f"Turn {self.turn}: the provider failed ({str(exc)[:200]}); the turn is repeated."
                )
            )
            self.provider_errors += 1
            if self.provider_errors >= 3:
                self.outcome = "error: " + str(exc)[:200]
                return None, ""
            return [], (
                "The previous request failed on the provider's side, not on yours; the turn is repeated. "
                "Answer with exactly one tool call."
            )
        if calls:
            return calls, ""
        recovered = recover_call(content, self.available())
        if recovered is None:
            self.no_call_turns += 1
            self._log(f"no tool call: {content[:300]}")
            self.show(
                display.note(
                    f"Turn {self.turn}: the model answered without a tool call; it is asked again."
                )
            )
            return (
                [],
                "You answered without a tool call. Answer with exactly one tool call from the offered tools.",
            )
        self.recovered_calls += 1
        self._log(f"tool call recovered from the reply text: {recovered[0]}")
        return [recovered], ""

    def _constrains(self) -> bool:
        return bool(self.tool_choice and getattr(self.llm, "tool_choice_supported", False))

    def _request(
        self, offered: list[FunctionTool], messages: list[ChatMessage], choice: dict[str, Any]
    ) -> tuple[list[tuple[str, dict]], str]:
        """One request: the tool calls of the reply (at most two, the first is executed) and its text."""
        # LakeGen's OCI client sends tool_choice as a string the OCI SDK rejects, so a choice is passed only
        # to a client that declares support (``_constrains``); elsewhere a turn without a call gets a note.
        # The choice goes as ``tool_choice`` itself: the client's ``tool_required`` flag reaches the SDK as auto.
        response = self.llm.chat_with_tools(offered, chat_history=messages, **choice)
        selections = self.llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        calls = []
        for selection in selections[:2]:
            name = re.split(r"<\||\s", str(selection.tool_name).replace("functions.", ""))[
                0
            ].strip()
            calls.append(
                (name, selection.tool_kwargs if isinstance(selection.tool_kwargs, dict) else {})
            )
        return calls, str(response.message.content or "")

    def _execute(self, tools: dict[str, FunctionTool], name: str, args: dict) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        signature = name + json.dumps(args, sort_keys=True, default=str)
        self.signatures[signature] = self.signatures.get(signature, 0) + 1
        if name not in tools or name not in self.available():
            result = f"{name} is not offered now. Offered: {', '.join(self.available())}."
            self.refusals["other"] += 1
        else:
            result = self._call(name, args)
        if self.signatures[signature] >= 3 and not self.outcome:
            self.decision_mode = True
            result += (
                " (Third identical call: decision mode -- only confirm or reject are offered now.)"
            )
        elif self.signatures[signature] == 2 and not self.outcome:
            result = (
                "Repeated call, identical to an earlier one: nothing new. "
                + self.next_step()
                + ". "
                + result
            )
        call = f"{name}({json.dumps(args, ensure_ascii=False, default=str)[:300]})"
        self._log(f"{call}\n{result[:2000]}")
        self.show(display.tool_call(self, name, args, result))
        self.memory.push(
            Exchange(self.turn, call, result[:RESULT_CHARS], self._summary(name, args, result))
        )

    def show(self, call: ToolCall) -> None:
        """A tool call for the person watching: a step of its own in the webapp, else one line of the stream."""
        if self.tool_callback is not None:
            self.tool_callback(call)
        elif self.emit is not None:
            when = "" if call.name == "contrast" else f"turn {self.turn} · "
            self.emit(f"- {when}{call.title}: {call.summary}\n")

    def _call(self, name: str, args: dict) -> str:
        fn = self.functions()[name]
        accepted = inspect.signature(fn).parameters
        kwargs = {k: v for k, v in args.items() if k in accepted}
        self.memory.last_refusal = ""
        try:
            return str(fn(**kwargs))
        except TypeError as exc:
            return f"Error: bad arguments for {name}: {exc}"
        except WorkflowCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- a tool that breaks (an unreadable table) answers with the error
            return f"Error: {name} failed: {type(exc).__name__}: {str(exc)[:200]}"

    def _log(self, text: str) -> None:
        self.trace.write(f"\n[turn {self.turn}] {text}\n")

    def _summary(self, name: str, args: dict, result: str) -> str:
        if name == "inspect":
            got = [n for n in _numbers(args.get("candidates")) if n in self.memory.cards]
            return (
                f"inspected {got} (cards in STATE)"
                if got
                else f"inspect {args.get('candidates')}: {result[:80]}"
            )
        if name == "select":
            return f"select #{args.get('table')} covers {args.get('covers')}: {result[:120]}"
        return f"{name}({json.dumps(args, ensure_ascii=False, default=str)[:100]}): {' '.join(result.split())[:140]}"

    def summary_line(self) -> str:
        """The run in one line, for the trace."""
        m = self.memory
        return (
            f"outcome={self.outcome} turns={self.turn} actions={self.actions} expansions={self.expansions} "
            f"inspected={len(m.inspected())} selected={len(m.selection)} refusals={self.refusals} "
            f"no_call_turns={self.no_call_turns} recovered_calls={self.recovered_calls} "
            f"cover_refusals={self.cover_refusals}"
        )


MemGPTAgent.define_requirements.__doc__ = DEFINE_REQUIREMENTS_TOOL
MemGPTAgent.select.__doc__ = SELECT_TOOL
