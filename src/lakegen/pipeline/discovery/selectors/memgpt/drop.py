"""The drop check: when two or more tables are selected, two calls ask whether every one of them holds a requirement.

The selector's other remaining error on the questions that need several tables is the extra table: next to the
right ones it keeps a table that holds none of the requirements (the polling sectors beside the results, a
second year's copy, the map beside the list) and the coder joins it. The options are the whole selection and the
selection without each of its tables. The tables are lettered by the requirements they show and their title,
never by their window number, and the options are read twice, in two orders labelled by position (S0, S1, ...),
so that a position bias cannot pick: a table is left out only when both calls name the same drop, and any other
answer keeps the selection. Every selected table was inspected, so there is nothing to profile.

Validated in experiments/coverage_lab (runs ``decision1`` and ``decision2``, after the contrast check): 19
selections fixed and none broken over five samples of the 188 dev questions in three candidate orders (McNemar
p 0.001 and 0.008); 2 calls and about 3.6k tokens on the questions with two or more selected tables. Swapping a
look-alike in the same rank-free setting fixed nothing and broke twins the question cannot tell apart, so the
contrast check keeps its swaps and this check only drops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lakegen.pipeline.discovery.selectors.memgpt import contrast, display
from lakegen.pipeline.discovery.selectors.memgpt.memory import Memory

PROMPT = "memgpt_selector"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Dropped:
    """The outcome of the drop check.

    Attributes:
        options: Options the model read: the whole selection, then one per table left out.
        calls: Model calls the check made.
        dropped: The table left out, when both calls named the same one.
        picks: The option each call picked, as an index into the options; None when unreadable.
        reasoning: The calls' reasons, joined.
    """

    options: int
    calls: int
    dropped: str | None = None
    picks: list[int | None] = field(default_factory=list)
    reasoning: str = ""


def _number(memory: Memory, name: str) -> int:
    return memory.candidates.index(name) + 1


def rank_free_key(memory: Memory, name: str) -> tuple:
    """How the tables are ordered and lettered: the requirements they show, then their title, then their file name."""
    n = _number(memory, name)
    card = memory.cards.get(n)
    return (-(len(card.shows) if card else 0), memory.heading(n).casefold(), name)


def options(memory: Memory, selected: list[str]) -> list[list[str]]:
    """The whole selection first, then the selection without each of its tables, in rank-free order."""
    tables = sorted(selected, key=lambda t: rank_free_key(memory, t))
    return [tables, *([t for t in tables if t != left] for left in tables)]


def letters(memory: Memory, selected: list[str]) -> dict[str, str]:
    """Table name -> "Table A", "Table B", ...: lettered by ``rank_free_key``, never by the window number."""
    names = sorted(selected, key=lambda t: rank_free_key(memory, t))
    return {
        name: f"Table {LETTERS[i] if i < len(LETTERS) else i + 1}" for i, name in enumerate(names)
    }


def comparison(
    memory: Memory, opts: list[list[str]], names: dict[str, str], order: list[int]
) -> str:
    """What the model reads: the question, the requirements, the tables lettered, the options in ``order``."""
    whole = set(opts[0])
    requirements = "\n".join(
        f"  {r.label}" + (f" (waived: {r.waived[:60]})" if r.waived else "")
        for r in memory.requirements
    )
    lines = []
    for k, i in enumerate(order):
        tables = ", ".join(names[t] for t in opts[i])
        left = whole - set(opts[i])
        lines.append(
            f"S{k}: {tables}" + (f" -- {names[next(iter(left))]} left out" if left else "")
        )
    return "\n\n".join(
        [
            f"QUESTION: {memory.question}",
            "REQUIREMENTS:\n" + requirements,
            "TABLES:\n"
            + "\n".join(contrast.table_summary(memory, t, label=names[t]) for t in opts[0]),
            "SELECTIONS:\n" + "\n".join(lines),
        ]
    )


def choose(alternative: str, reasoning: str) -> str:
    """Choose the selection (S0, S1, ...) whose tables hold the data the question needs, with the reason."""
    return alternative


def _pick(args: dict | None, order: list[int]) -> tuple[int | None, str]:
    args = args or {}
    label = str(args.get("alternative", "")).strip().upper()
    found = re.match(r"S(\d+)", label)
    pick = order[int(found.group(1))] if found and int(found.group(1)) < len(order) else None
    return pick, " ".join(str(args.get("reasoning", "")).split())[:300]


def check(agent: Any, llm: Any, prompts: Any) -> Dropped | None:
    """Ask twice whether a selected table holds nothing and leave it out when both answers agree; None below two tables."""
    m = agent.memory
    if agent.outcome == "rejected" or len(m.selection) < 2:
        return None
    selected = agent.selected()
    opts = options(m, selected)
    names = letters(m, selected)
    system = prompts.render(PROMPT, "drop_system")
    picks: list[int | None] = []
    calls, reasons = 0, []
    for order in (list(range(len(opts))), list(reversed(range(len(opts))))):
        if agent.cancel_check is not None:
            agent.cancel_check()
        args, n = contrast._ask(llm, system, comparison(m, opts, names, order), fn=choose)
        calls += n
        pick, reason = _pick(args, order)
        picks.append(pick)
        if reason:
            reasons.append(reason)
    agreed = picks[0] is not None and picks[0] == picks[1]
    dropped = next(iter(set(opts[0]) - set(opts[picks[0]]))) if agreed and picks[0] else None
    reasoning = " | ".join(reasons)[:400]
    if dropped:
        apply(agent, dropped, reasoning)
        note = f"left out #{_number(m, dropped)}"
    else:
        note = f"kept the {len(selected)} tables" + ("" if agreed else " (the two calls disagreed)")
    agent.trace.write(
        f"\n[drop] {note} ({calls} call{'s' if calls != 1 else ''}){': ' + reasoning if reasoning else ''}\n"
    )
    agent.show(display.drop(agent, selected, names, dropped, reasoning, agreed))
    return Dropped(len(opts), calls, dropped, picks, reasoning)


def apply(agent: Any, dropped: str, reasoning: str) -> None:
    """Leave the table out of the selection, the way the agent's own unselect does."""
    m = agent.memory
    n = _number(m, dropped)
    if n not in m.selection:
        return
    m.selection.pop(n)
    m.unselected[n] = "left out by the drop check" + (f": {reasoning[:120]}" if reasoning else "")
    m.relations = {k: v for k, v in m.relations.items() if n not in k}
