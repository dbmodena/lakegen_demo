"""How a person reads the MemGPT selector's tool calls: a title, a markdown result and a one-line summary.

The agent's tool results are written for the model (file names, ``#N`` numbers, raw profiles). Here
each call becomes a ``ToolCall`` that names tables by their titles, lists the requirements, shows
the profile of an inspected table in a code block under what its card says it shows, and marks a
refused call as failed. The webapp shows each one as a step; a terminal prints the summary.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from lakegen.datastructures.types import ToolCall

logger = logging.getLogger(__name__)

FAILED = "⚠"
"""Marks a refused or broken call (a text symbol: every font has it)."""
TITLE_CHARS = 70
"""A table's title in an output; a step's title shortens each table further (``SHORT_CHARS``)."""
SHORT_CHARS = 56
_REFUSED = re.compile(
    r"^(Confirm refused|Reject refused|Not accepted|Error|Unknown requirement|No actions left|No expansions left"
    r"|Give the reason|#\d+ is not (inspected|in the selection)|.+ is not offered now|R\d+ is an item)"
)
_PROFILED = re.compile(r"^\[#(\d+)\]$", re.MULTILINE)
_PAIR = re.compile(r"^#(\d+)-#(\d+):")


def table(agent: Any, number: int, chars: int = TITLE_CHARS) -> str:
    """A candidate as a person names it: its number and title."""
    if not 1 <= number <= len(agent.memory.candidates):
        return f"#{number}"
    return f"#{number} {shorten(agent.memory.heading(number), chars)}"


def shorten(text: str, chars: int) -> str:
    """The text cut in the middle at word boundaries: look-alike titles differ at the end ("... lista PASQUINO")."""
    if len(text) <= chars:
        return text
    words = text.split()
    head: list[str] = []
    while words and len(" ".join([*head, words[0]])) <= chars * 2 // 5:
        head.append(words.pop(0))
    tail: list[str] = []
    while words and len(" ".join([*head, "…", words[-1], *tail])) <= chars:
        tail.insert(0, words.pop())
    if not head and not tail:
        return text[: chars - 1] + "…"
    return " ".join([*head, "…", *tail])


def _number(value: Any) -> int:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else 0


def _quote(text: Any) -> str:
    text = " ".join(str(text or "").split())
    return f"> {text}" if text else ""


def _fenced(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text.strip()}\n{fence}"


def _sentence(text: str) -> str:
    text = text.strip(" .")
    return text[:1].upper() + text[1:] + "." if text else ""


def _one_line(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _requirements(agent: Any) -> str:
    covered = agent.memory.covered_by()
    lines = []
    for r in agent.memory.requirements:
        if r.waived:
            mark = f" — *waived: {r.waived}*"
        elif covered.get(r.id):
            mark = " — covered by " + ", ".join(f"#{n}" for n in covered[r.id])
        elif r.item is not None:
            mark = " — *a named item: needs a table that shows it*"
        else:
            mark = ""
        lines.append(f"- **{r.id}** {r.kind}: {r.text}{mark}")
    return "\n".join(lines)


def _shows(agent: Any, number: int) -> str:
    card = agent.memory.cards.get(number)
    if card is None or not card.shows:
        return "Shows none of the requirements."
    return "Shows " + "; ".join(f"**{rid}** in {where}" for rid, where in card.shows.items()) + "."


def _define(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    items = args.get("named_items") or []
    parts = [_requirements(agent) or result]
    if items:
        parts.append(
            "Named items: "
            + ", ".join(f"`{i}`" for i in (items if isinstance(items, list) else [items]))
        )
    parts += [f"*{_sentence(s)}*" for s in re.split(r"[.;] ", result) if "system" in s]
    summary = _one_line(
        "; ".join(f"{r.id} {r.kind}: {r.text}" for r in agent.memory.requirements), 240
    )
    ids = [r.id for r in agent.memory.requirements]
    detail = f"{ids[0]}–{ids[-1]}" if len(ids) > 1 else "".join(ids)
    return detail, "\n\n".join(parts), summary or result


def _inspect(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    profiled = [int(n) for n in _PROFILED.findall(result)]
    asked = (
        [_number(n) for n in (args.get("candidates") or [])]
        if isinstance(args.get("candidates"), list)
        else profiled
    )
    title = ", ".join(
        table(agent, n, SHORT_CHARS) if len(profiled or asked) == 1 else f"#{n}"
        for n in (profiled or asked)
    )
    blocks = []
    head = (
        result.split("\n\n[#", 1)[0]
        if profiled and not result.startswith("[#")
        else ("" if profiled else result)
    )
    if head:
        blocks.append(f"*{head.strip()}*")
    for n in profiled:
        profile = agent.memory.profiles.get(n, "")
        card = agent.memory.cards.get(n)
        rows = f" — {card.rows} rows" if card and card.rows else ""
        blocks.append(f"**{table(agent, n)}**{rows}\n\n{_shows(agent, n)}\n\n{_fenced(profile)}")
    summary = "; ".join(
        f"#{n} {len(agent.memory.cards[n].shows)} requirements shown"
        for n in profiled
        if n in agent.memory.cards
    ) or _one_line(result)
    return title, "\n\n".join(blocks), summary


def _expand(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    first, _, rest = result.partition("\n")
    revealed = [_number(line) for line in rest.splitlines() if line.startswith("#")]
    lines = "\n".join(f"- {table(agent, n)}" for n in revealed)
    missing = _one_line(args.get("missing", ""), 60)
    title = f"for “{missing}”" if missing else ""
    return title, f"*{first}*" + (f"\n\n{lines}" if lines else ""), _one_line(first)


def _select(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    n = _number(args.get("table"))
    covers = args.get("covers")
    covers = ", ".join(map(str, covers)) if isinstance(covers, list) else str(covers or "")
    now = ", ".join(agent.memory.selection.get(n, []))
    head = (
        f"Covers **{covers}**" + (f" (#{n} now covers {now})" if now and now != covers else "")
        if covers
        else ""
    )
    remarks, _, measured = result.partition("Measured against the selection: ")
    notes = [s.strip() for s in re.split(r"(?<=\.) (?=[A-Z#])|\n", remarks) if s.strip()]
    notes = [
        f"- {s.replace('kept on your word', 'accepted on the selector’s word')}"
        for s in notes
        if not re.match(r"#\d+ selected, covering", s)
    ]
    relations = [r.strip() for r in measured.split(" | ") if r.strip()]
    links = ["- " + _PAIR.sub(r"#\1 ↔ #\2:", r) for r in relations]
    parts = [
        head,
        _quote(args.get("note")),
        "\n".join(notes),
        ("**Measured against the selection**\n\n" + "\n".join(links)) if links else "",
    ]
    return (
        table(agent, n, SHORT_CHARS) + (f" → {covers}" if covers else ""),
        "\n\n".join(p for p in parts if p),
        _one_line(result),
    )


def _unselect(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    n = _number(args.get("table"))
    return (
        table(agent, n, SHORT_CHARS),
        "\n\n".join(p for p in (_quote(args.get("reason")), result) if p),
        _one_line(result),
    )


def _waive(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    rid = str(args.get("requirement", "")).strip()
    return rid, "\n\n".join(p for p in (_quote(args.get("reason")), result) if p), _one_line(result)


def _recall(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    n = _number(args.get("table"))
    profile = agent.memory.profiles.get(n)
    return (
        table(agent, n, SHORT_CHARS),
        _fenced(profile) if profile else result,
        _one_line(result, 80),
    )


def _confirm(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    if result != "CONFIRMED":
        return "refused", result, _one_line(result)
    m = agent.memory
    selection = "\n".join(
        f"- **{table(agent, n)}** covers {', '.join(rids)}"
        for n, rids in sorted(m.selection.items())
    )
    parts = [
        "**Confirmed.**",
        _quote(args.get("reasoning")),
        f"**Selection**\n\n{selection}",
        f"**Requirements**\n\n{_requirements(agent)}",
    ]
    chosen = ", ".join(f"#{n}" for n in sorted(m.selection))
    return chosen, "\n\n".join(p for p in parts if p), f"confirmed: {chosen}"


def _reject(agent: Any, args: dict, result: str) -> tuple[str, str, str]:
    if result != "REJECTED":
        return "refused", result, _one_line(result)
    parts = ["**Rejected:** retrieval is asked for other keywords.", _quote(args.get("reasoning"))]
    if args.get("suggestion"):
        parts.append(f"Suggestion: {args['suggestion']}")
    return (
        "",
        "\n\n".join(p for p in parts if p),
        _one_line(f"rejected: {args.get('reasoning', '')}"),
    )


BUILDERS = {
    "define_requirements": _define,
    "inspect": _inspect,
    "expand": _expand,
    "select": _select,
    "unselect": _unselect,
    "waive": _waive,
    "recall": _recall,
    "confirm": _confirm,
    "reject": _reject,
}


def tool_call(agent: Any, name: str, args: dict, result: str) -> ToolCall:
    """One tool call of the agent, for a person; a formatting failure falls back to the raw result."""
    failed = bool(_REFUSED.match(result)) or result.startswith("Repeated call")
    try:
        detail, output, summary = BUILDERS[name](agent, args, result)
    except Exception:  # noqa: BLE001 -- the display must never break a selection
        logger.debug("No display for the %s call; showing its raw result", name, exc_info=True)
        detail, output, summary = "", _fenced(result), _one_line(result)
    if failed:
        detail, output = ("refused" if name in ("confirm", "reject") else detail), result
    title = (f"{FAILED} " if failed else "") + name + (f" · {detail}" if detail else "")
    return ToolCall(name, title, output, summary, failed)


def note(text: str) -> ToolCall:
    """A turn without a tool call, or a provider error: shown so the turn count adds up."""
    return ToolCall("note", f"{FAILED} no tool call", text, _one_line(text), failed=True)


def contrast(agent: Any, alternatives: list[Any], picked: Any, reasoning: str) -> ToolCall:
    """The contrast check: the selection, the look-alike swaps it was compared with, and the choice."""
    m = agent.memory

    def number(name: str) -> int:
        return m.candidates.index(name) + 1

    lines = []
    for alt in alternatives:
        if not alt.swaps:
            lines.append(
                f"- **{alt.label}** (the selection): "
                + ", ".join(table(agent, number(t)) for t in alt.tables)
            )
        else:
            lines.append(
                f"- **{alt.label}**: "
                + "; ".join(
                    f"{table(agent, number(old))} → {table(agent, number(new))}"
                    for old, new in alt.swaps
                )
            )
    if picked.swaps:
        verdict = f"**Chose {picked.label}:** " + "; ".join(
            f"#{number(o)} replaced by #{number(n)}" for o, n in picked.swaps
        )
        summary = "swapped " + "; ".join(f"#{number(o)} for #{number(n)}" for o, n in picked.swaps)
    else:
        verdict = "**Kept the selection.**"
        swaps = len(alternatives) - 1
        summary = f"kept the selection over {swaps} look-alike swap{'s' if swaps != 1 else ''}"
    parts = [
        "Compared with the selections that swap one table for a look-alike:",
        "\n".join(lines),
        verdict,
        _quote(reasoning),
    ]
    return ToolCall(
        "contrast", f"contrast check · {summary}", "\n\n".join(p for p in parts if p), summary
    )


def drop(
    agent: Any,
    selected: list[str],
    names: dict[str, str],
    dropped: str | None,
    reasoning: str,
    agreed: bool,
) -> ToolCall:
    """The drop check: the selected tables as the model saw them lettered, and whether one was left out."""
    m = agent.memory
    lines = [
        f"- **{names[t]}**: {table(agent, m.candidates.index(t) + 1)}"
        for t in sorted(selected, key=names.get)
    ]
    if dropped:
        verdict = f"**Left out {names[dropped]}** (#{m.candidates.index(dropped) + 1}): both calls named it."
        summary = f"left out #{m.candidates.index(dropped) + 1}"
    else:
        verdict = "**Kept the selection.**" + (
            "" if agreed else " The two calls disagreed, which keeps every table."
        )
        summary = f"kept the {len(selected)} tables"
    parts = [
        "Asked twice, over two orders of the options, whether every selected table holds a requirement:",
        "\n".join(lines),
        verdict,
        _quote(reasoning),
    ]
    return ToolCall("drop", f"drop check · {summary}", "\n\n".join(p for p in parts if p), summary)


def candidates(agent: Any) -> ToolCall:
    """What retrieval handed the selector: the candidates in order, those shown first and those it can reveal."""
    m = agent.memory
    titles = [m.heading(n) for n in range(1, len(m.candidates) + 1)]
    lines = []
    for n, name in enumerate(m.candidates, 1):
        if n == m.visible + 1:
            lines.append(
                f"\n*Hidden at first (expand reveals {len(m.candidates) - m.visible} more, 5 at a time):*\n"
            )
        title = titles[n - 1]
        description = " ".join(str(m.metadata.get(name, {}).get("description") or "").split())
        vague = titles.count(title) > 1 or len(title.split()) < 3
        lines.append(
            f"{n}. {title}" + (f" — *{shorten(description, 110)}*" if vague and description else "")
        )
    summary = f"{len(m.candidates)} candidates, {m.visible} shown first"
    return ToolCall("candidates", f"retrieval · {summary}", "\n".join(lines), summary)


def alternatives(agent: Any, offered: list[Any], pool: int, reasoning: str) -> ToolCall:
    """The alternatives check: the other selections that also answer the question, each with what differs."""
    m = agent.memory
    lines = []
    for alt in offered:
        tables = ", ".join(table(agent, m.candidates.index(t) + 1) for t in alt.tables)
        lines.append(f"- **{alt.label}**: {tables}" + (f" — {alt.note}" if alt.note else ""))
    if offered:
        verdict = (
            f"**{len(offered)} other selections also answer the question.**"
            if len(offered) != 1
            else "**1 other selection also answers the question.**"
        )
        summary = f"{len(offered)} offered of {pool}"
    else:
        verdict = "**No other selection holds every requirement.**"
        summary = f"none of {pool} offered"
    parts = [
        "The selection stays; of the selections one change away from it, these also hold every requirement:",
        "\n".join(lines),
        verdict,
        _quote(reasoning),
    ]
    return ToolCall(
        "alternatives", f"alternatives · {summary}", "\n\n".join(p for p in parts if p), summary
    )
