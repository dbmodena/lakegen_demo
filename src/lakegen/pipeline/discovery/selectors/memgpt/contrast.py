"""The contrast check: once the selector holds a selection, one call compares its tables with their look-alikes.

The selector's main remaining error is the sibling: it inspects the right table and keeps one that looks like
it (the votes of the lists for the preference votes of a list, another year's sites). For each selected table,
the three candidates that look most like it -- title words, indexed columns, the same dataset -- make the
alternatives: A0 is the selection, A1... swap one table for one look-alike (at most seven). One call shows the
tables side by side (rows, columns, dates, sampled values, the requirements each shows) and what each
alternative changes; the model chooses, keeps A0 unless another alternative clearly holds a requirement A0's
tables do not, and A0 stays when the call fails. The tables of the alternatives the agent never inspected are
profiled first (no LLM).

Validated in experiments/sibling_lab (arm ``contrast``): 6 questions fixed and none broken over 478, +1.6 points
of exact selections on the 250 held-out questions, 0.9 calls per question.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import FunctionTool

from lakegen.datastructures.errors import WorkflowCancelled
from lakegen.pipeline.discovery.selectors.memgpt import display
from lakegen.pipeline.discovery.selectors.memgpt.memory import Memory

logger = logging.getLogger(__name__)

PROMPT = "memgpt_selector"
THRESHOLD = 0.3
PER_TABLE = 3
MAX_ALTERNATIVES = 7
ATTEMPTS = 2
_WORD = re.compile(r"[a-zà-ÿ0-9]{3,}")
_SAMPLE = re.compile(r"^- (.+?) \(Category sample\): (.*)$", re.MULTILINE)
_COVERAGE = re.compile(r"^- (.+?): (\S+) to (\S+)", re.MULTILINE)


@dataclass
class Alternative:
    """One selection the contrast check offers: the current one (A0) or one with a look-alike swapped in.

    Attributes:
        label: ``A0``, ``A1``, ...
        tables: The alternative's tables.
        swaps: (selected table, the look-alike that replaces it).
    """

    label: str
    tables: list[str]
    swaps: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Checked:
    """The outcome of the contrast check.

    Attributes:
        chosen: The alternative the model chose; A0 when the call failed.
        alternatives: Alternatives offered, A0 included.
        calls: Model calls the check made.
        reasoning: The model's reason.
    """

    chosen: Alternative
    alternatives: int
    calls: int
    reasoning: str = ""


def _words(text) -> set[str]:
    return set(_WORD.findall(str(text or "").casefold()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def similarity(metadata: dict[str, dict[str, Any]], a: str, b: str) -> float:
    """How much two candidates look alike: title words, plus half the column overlap, plus half for the same dataset."""
    ma, mb = metadata.get(a, {}), metadata.get(b, {})
    columns = lambda m: {str(c).casefold() for c in m.get("columns.name") or []}  # noqa: E731
    same_dataset = "___" in a and a.split("___")[0] == b.split("___")[0]
    return (
        _jaccard(_words(ma.get("title")), _words(mb.get("title")))
        + 0.5 * _jaccard(columns(ma), columns(mb))
        + (0.5 if same_dataset else 0.0)
    )


def siblings(
    metadata: dict[str, dict[str, Any]],
    name: str,
    pool: list[str],
    exclude: set[str],
    k: int = PER_TABLE,
) -> list[str]:
    """The k candidates of ``pool`` that look most like ``name`` (at least THRESHOLD), closest first."""
    scored = sorted(
        ((similarity(metadata, name, o), o) for o in pool if o != name and o not in exclude),
        reverse=True,
    )
    return [o for s, o in scored if s >= THRESHOLD][:k]


def alternatives(
    metadata: dict[str, dict[str, Any]], selected: list[str], candidates: list[str]
) -> list[Alternative]:
    """A0 = the selection; then the selection with one table swapped for one of its closest look-alikes."""
    out = [Alternative("A0", list(selected))]
    chosen = set(selected)
    for table in selected:
        for sibling in siblings(metadata, table, candidates, chosen):
            if len(out) >= MAX_ALTERNATIVES:
                return out
            out.append(
                Alternative(
                    f"A{len(out)}",
                    [sibling if t == table else t for t in selected],
                    [(table, sibling)],
                )
            )
    return out


def _number(memory: Memory, name: str) -> int:
    return memory.candidates.index(name) + 1


def table_summary(
    memory: Memory, name: str, *, max_columns: int = 30, label: str | None = None
) -> str:
    """A table as the checks show it, headed by ``label`` (``T<n>`` unless the check letters its tables)."""
    n = _number(memory, name)
    label = label or f"T{n}"
    card, profile = memory.cards.get(n), memory.profiles.get(n, "")
    if card is None:
        return f'{label} "{memory.heading(n)}" -- not profiled'
    lines = [f'{label} "{card.title}"' + (f" -- {card.rows} rows" if card.rows else "")]
    if memory.other_title(n):
        lines.append(f"  also titled: {memory.other_title(n)}")
    if memory.publisher(n):
        lines.append(f"  publisher: {memory.publisher(n)}")
    description = " ".join(str(memory.metadata.get(name, {}).get("description") or "").split())[
        :200
    ]
    if description:
        lines.append(f"  description: {description}")
    columns = card.columns[:max_columns]
    lines.append(
        f"  columns ({len(card.columns)}): {', '.join(columns)}"
        + (" ..." if len(card.columns) > max_columns else "")
    )
    dates = [f"{c} {a}..{b}" for c, a, b in _COVERAGE.findall(profile)[:3]]
    if dates:
        lines.append(f"  dates: {'; '.join(dates)}")
    samples = [f"{c}: {v[:120]}" for c, v in _SAMPLE.findall(profile)[:6]]
    if samples:
        lines.append("  sampled values: " + " | ".join(samples))
    shows = "; ".join(f"{rid} in {where}" for rid, where in card.shows.items())
    lines.append(f"  shows: {shows or 'none of the requirements'}")
    return "\n".join(lines)


def describe(memory: Memory, alternative: Alternative) -> str:
    """One line per alternative: its tables and, for a swap, the columns it gains and loses."""
    tables = ", ".join(f"T{_number(memory, t)}" for t in alternative.tables)
    if not alternative.swaps:
        return f"{alternative.label} (the current selection): {tables}"
    changes = []
    for old, new in alternative.swaps:
        a, b = memory.cards.get(_number(memory, old)), memory.cards.get(_number(memory, new))
        only_new = [c for c in (b.columns if b else []) if c not in (a.columns if a else [])][:8]
        only_old = [c for c in (a.columns if a else []) if c not in (b.columns if b else [])][:8]
        changes.append(
            f"T{_number(memory, old)} replaced by T{_number(memory, new)}"
            + (f" (gains columns {', '.join(only_new)}" if only_new else " (no new columns")
            + (f"; loses {', '.join(only_old)})" if only_old else ")")
        )
    return f"{alternative.label}: {tables} -- " + "; ".join(changes)


def comparison(memory: Memory, alts: list[Alternative]) -> str:
    """What the model reads: the question, the requirements, the tables side by side, the alternatives."""
    tables = list(dict.fromkeys(t for a in alts for t in a.tables))
    requirements = "\n".join(
        f"  {r.label}" + (f" (waived: {r.waived[:60]})" if r.waived else "")
        for r in memory.requirements
    )
    return "\n\n".join(
        [
            f"QUESTION: {memory.question}",
            "REQUIREMENTS:\n" + requirements,
            "TABLES:\n" + "\n".join(table_summary(memory, t) for t in tables),
            "ALTERNATIVE SELECTIONS:\n" + "\n".join(describe(memory, a) for a in alts),
        ]
    )


def choose(alternative: str, reasoning: str) -> str:
    """Choose the alternative (A0, A1, ...) whose tables hold the data the question needs, with the reason."""
    return alternative


def _ask(
    llm: Any, system: str, user: str, fn: Callable[..., str] = choose
) -> tuple[dict | None, int]:
    """One forced call (two tries) of the tool ``fn``, named after it: the call's arguments, or None; and the calls spent."""
    tool = FunctionTool.from_defaults(
        fn=fn, name=fn.__name__, description=(fn.__doc__ or "").strip()
    )
    note, calls = "", 0
    for _ in range(ATTEMPTS):
        calls += 1
        try:
            response = llm.chat_with_tools(
                [tool],
                chat_history=[
                    ChatMessage(role=MessageRole.SYSTEM, content=system),
                    ChatMessage(role=MessageRole.USER, content=user + note),
                ],
            )
            selections = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
        except WorkflowCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- a failed call keeps the selection
            logger.warning("A forced-choice call failed: %s: %s", type(exc).__name__, exc)
            note = f"\n\n(Your last reply failed: {str(exc)[:100]}. Answer with one tool call.)"
            continue
        if selections and isinstance(selections[0].tool_kwargs, dict):
            return selections[0].tool_kwargs, calls
        found = re.search(r"\{.*\}", str(response.message.content or ""), re.S)
        if found:
            try:
                return json.loads(found.group()), calls
            except json.JSONDecodeError:
                pass
        note = "\n\nAnswer with exactly one tool call."
    return None, calls


def _chosen(alts: list[Alternative], label: Any) -> Alternative:
    text = str(label or "").strip().upper()
    return next((a for a in alts if a.label == text or text.startswith(a.label + " ")), alts[0])


def check(agent: Any, llm: Any, prompts: Any) -> Checked | None:
    """Compare the agent's selection with its look-alikes and apply the choice; None when there is nothing to compare."""
    m = agent.memory
    if agent.outcome == "rejected" or not m.selection:
        return None
    selected = agent.selected()
    alts = alternatives(m.metadata, selected, m.candidates)
    if len(alts) == 1:
        return Checked(alts[0], 1, 0)
    for name in dict.fromkeys(t for a in alts for t in a.tables):
        number = _number(m, name)
        if number not in m.profiles:
            m.record_profile(number, agent._profile(number))
    if agent.cancel_check is not None:
        agent.cancel_check()
    args, calls = _ask(llm, prompts.render(PROMPT, "contrast_system"), comparison(m, alts))
    args = args or {}
    picked = _chosen(alts, args.get("alternative"))
    reasoning = " ".join(str(args.get("reasoning", "")).split())[:400]
    if picked.swaps:
        apply(agent, picked, reasoning)
    note = (
        f"kept the selection over {len(alts) - 1} look-alike swaps"
        if not picked.swaps
        else "swapped "
        + "; ".join(f"#{_number(m, o)} for #{_number(m, n)}" for o, n in picked.swaps)
    )
    agent.trace.write(
        f"\n[contrast] {note} ({calls} call{'s' if calls != 1 else ''}){': ' + reasoning if reasoning else ''}\n"
    )
    agent.show(display.contrast(agent, alts, picked, reasoning))
    return Checked(picked, len(alts), calls, reasoning)


def apply(agent: Any, alternative: Alternative, reasoning: str) -> None:
    """Swap the tables in the selection: the look-alike takes over the requirements the table covered."""
    m = agent.memory
    for old, new in alternative.swaps:
        o, n = _number(m, old), _number(m, new)
        if o not in m.selection:
            continue
        m.selection[n] = m.selection.pop(o)
        m.unselected[o] = f"replaced by #{n} in the contrast check" + (
            f": {reasoning[:120]}" if reasoning else ""
        )
        m.unselected.pop(n, None)
        m.relations = {k: v for k, v in m.relations.items() if o not in k}
        agent._measure_with_selection(n)
