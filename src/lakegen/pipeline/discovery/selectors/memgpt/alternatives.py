"""The alternatives check: once the selection is final, one call lists the other selections that also answer.

A generic question ("how many parishes were there in 2017?") is answered by several tables of the lake
alike: the same boundaries at three resolutions, the same list in two languages, the same subject with or
without some areas the question does not name. The benchmark's gold is one of them and the selector hands
on one selection; the person asking may prefer another. The selections one change away from the selection
(a selected table swapped for one of its closest look-alikes, as the contrast check builds them, or one of
several tables left out) are shown side by side with it, and the model keeps every one whose tables hold
every requirement, best first, with a line on what differs. The selection itself never moves: the coder
runs on it, the alternatives go to the person with the answer. The tables of the alternatives the agent
never inspected are profiled first (no LLM).

Validated in experiments/alternatives_lab (arm ``vet``, 228 dev questions): the gold selection is the
selection or an offered alternative in 0.754 of the questions against 0.684 for the selection alone (16 of
72 misses recovered, of the 23 one change away), at 0.5 alternatives per question, three quarters of them
accepted by an LLM judge; 0.8 calls and 2.3k tokens per question. Copies of a selected table found on the
files added nothing, so no file is read here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lakegen.pipeline.discovery.selectors.memgpt import contrast, display
from lakegen.pipeline.discovery.selectors.memgpt.memory import Memory

PROMPT = "memgpt_selector"
MAX_POOL = 12
MAX_OFFERED = 5
_LABEL = re.compile(r"A\d+")


@dataclass
class Alternative:
    """One selection offered beside the selection.

    Attributes:
        label: ``A1``, ``A2``, ... (``A0`` is the selection).
        tables: The alternative's tables.
        change: (selected table, the look-alike that replaces it), or (selected table, None) when it is left out.
        note: One line on what differs from the selection and how the answer would change.
    """

    label: str
    tables: list[str]
    change: tuple[str, str | None]
    note: str = ""

    def record(self) -> dict[str, Any]:
        """The alternative as the plan records it."""
        return {
            "label": self.label,
            "tables": list(self.tables),
            "change": list(self.change),
            "note": self.note,
        }


@dataclass
class Offered:
    """The outcome of the alternatives check.

    Attributes:
        alternatives: The alternatives kept, best first.
        pool: Alternatives the model read.
        calls: Model calls the check made.
        reasoning: The model's reason.
    """

    alternatives: list[Alternative] = field(default_factory=list)
    pool: int = 0
    calls: int = 0
    reasoning: str = ""


def _number(memory: Memory, name: str) -> int:
    return memory.candidates.index(name) + 1


def pool(
    metadata: dict[str, dict[str, Any]], selected: list[str], candidates: list[str]
) -> list[Alternative]:
    """The selections one change away: each selected table swapped for its closest look-alikes, then each left out."""
    out: list[Alternative] = []
    seen = {frozenset(selected)}

    def offer(tables: list[str], change: tuple[str, str | None]) -> None:
        if tables and frozenset(tables) not in seen and len(out) < MAX_POOL:
            seen.add(frozenset(tables))
            out.append(Alternative(f"A{len(out) + 1}", tables, change))

    chosen = set(selected)
    for table in selected:
        for sibling in contrast.siblings(metadata, table, candidates, chosen):
            offer([sibling if t == table else t for t in selected], (table, sibling))
    if len(selected) > 1:
        for table in selected:
            offer([t for t in selected if t != table], (table, None))
    return out


def describe(memory: Memory, alternative: Alternative) -> str:
    """One line per alternative: its tables and what it changes (the columns a swap gains and loses)."""
    old, new = alternative.change
    if new is None:
        tables = ", ".join(f"T{_number(memory, t)}" for t in alternative.tables)
        return f"{alternative.label}: {tables} -- T{_number(memory, old)} left out"
    swap = contrast.Alternative(alternative.label, alternative.tables, [(old, new)])
    return contrast.describe(memory, swap)


def comparison(memory: Memory, selected: list[str], alts: list[Alternative]) -> str:
    """What the model reads: the question, the requirements, the tables side by side, the selection and the alternatives."""
    tables = list(dict.fromkeys([*selected, *(t for a in alts for t in a.tables)]))
    requirements = "\n".join(
        f"  {r.label}" + (f" (waived: {r.waived[:60]})" if r.waived else "")
        for r in memory.requirements
    )
    current = "A0 (the current selection): " + ", ".join(f"T{_number(memory, t)}" for t in selected)
    return "\n\n".join(
        [
            f"QUESTION: {memory.question}",
            "REQUIREMENTS:\n" + requirements,
            "TABLES:\n" + "\n".join(contrast.table_summary(memory, t) for t in tables),
            "ALTERNATIVE SELECTIONS:\n"
            + "\n".join([current, *(describe(memory, a) for a in alts)]),
        ]
    )


def keep(alternatives: list[str], notes: list[str], reasoning: str) -> str:
    """alternatives: the labels of the alternatives to keep, best first ("A2"); [] when none. notes: one line per kept alternative on what differs from A0 and how the answer changes."""
    return reasoning


def _kept(args: dict, alts: list[Alternative]) -> list[Alternative]:
    """The alternatives the call named, in its order, each with its note (by position, else after " -- ")."""
    raw = args.get("alternatives")
    if isinstance(raw, list):
        entries: list[Any] = raw
    elif " -- " in str(raw or ""):
        entries = str(raw).splitlines()
    else:
        entries = re.split(r"[,;\s]+", str(raw or ""))
    notes = args.get("notes")
    notes = [str(n) for n in notes] if isinstance(notes, list) else []
    by_label = {a.label: a for a in alts}
    out: list[Alternative] = []
    for i, item in enumerate(entries):
        raw_text = (
            str(item.get("alternative") or item.get("label") or item)
            if isinstance(item, dict)
            else str(item)
        )
        text = " ".join(raw_text.split())
        found = _LABEL.search(text.upper())
        alt = by_label.get(found.group()) if found else None
        if alt is None or any(a.tables == alt.tables for a in out):
            continue
        inline = text.partition(" -- ")[2].strip()
        note = " ".join(notes[i].split()) if i < len(notes) and notes[i].strip() else inline
        note = re.sub(
            r"^A\d+\s*[:\-\u2013\u2014]\s*", "", note
        )  # the model's own label, relabelled below
        out.append(Alternative(f"A{len(out) + 1}", list(alt.tables), alt.change, note[:300]))
        if len(out) == MAX_OFFERED:
            break
    return out


def check(agent: Any, llm: Any, prompts: Any) -> Offered | None:
    """List the other selections that also answer the question; None when there is no selection to compare with."""
    m = agent.memory
    if agent.outcome == "rejected" or not m.selection:
        return None
    selected = agent.selected()
    alts = pool(m.metadata, selected, m.candidates)
    if not alts:
        return Offered()
    for name in dict.fromkeys(t for a in alts for t in a.tables):
        number = _number(m, name)
        if number not in m.profiles:
            m.record_profile(number, agent._profile(number))
    if agent.cancel_check is not None:
        agent.cancel_check()
    args, calls = contrast._ask(
        llm, prompts.render(PROMPT, "alternatives_system"), comparison(m, selected, alts), fn=keep
    )
    kept = _kept(args or {}, alts)
    reasoning = " ".join(str((args or {}).get("reasoning", "")).split())[:400]
    note = (
        f"offered {len(kept)} of {len(alts)}"
        if kept
        else f"none of {len(alts)} held every requirement"
    )
    agent.trace.write(
        f"\n[alternatives] {note} ({calls} call{'s' if calls != 1 else ''}){': ' + reasoning if reasoning else ''}\n"
    )
    agent.show(display.alternatives(agent, kept, len(alts), reasoning))
    return Offered(kept, len(alts), calls, reasoning)
