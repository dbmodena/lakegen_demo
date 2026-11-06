"""The MemGPT table selector: the divided architecture's selector (``discovery.selector: memgpt``).

It replaced the judge agent on 2026-10-21. The judge saw 10 of the 20 retrieved candidates,
could inspect 8 of them and make 15 tool calls: given a question that needs six tables it would
select four and run out of calls before it inspected the other two. This selector keeps its
working memory the way
MemGPT does: a read-only block (the instructions, the tools, the question and the requirements it
defines first), a STATE the system writes from its tool calls (the selection with the
requirements each table covers, a card per inspected table, the tables still worth inspecting,
the measured relationships, the budgets) and a FIFO of the last turns behind a summary of the
earlier ones. It may reveal the hidden candidates twice and inspect four tables per action, and
its confirm and reject are refused while STATE still lists a table worth inspecting or a
requirement is open. Once it holds a selection, the contrast check (``contrast``) compares the
selected tables with their closest look-alikes in one call and may swap one, and, with two or more
tables selected, the drop check (``drop``) asks twice, over two orders of the options, whether one of
them holds none of the requirements and leaves it out when both answers agree. The alternatives check
(``alternatives``) then lists, for the person asking, the other selections one change away that also
answer the question, each with what differs; the selection itself stays. Since 2026-10-22 an
inspection shows what the whole table holds (``evidence``: years kept as integers, ranges, frequent
values, the step of the dates, the question's words found in the values), and a requirement that
names only the portal's own place is waived (experiments/profile_lab).

Validated in experiments/selector_lab as arm ``memgpt2b`` on gpt-oss-20b, over frozen retrieval
contexts of the five lakes: exact selections 0.66 against the judge's 0.38 on the same 50
questions (OCI dedicated endpoint), 0.54 against 0.30 on 127 questions, 97 of them needing two
tables or more (OCI on demand).

``agent`` runs the loop and the tools, ``contrast`` checks the selection against its look-alikes,
``drop`` leaves out a selected table that holds nothing (experiments/coverage_lab), ``alternatives``
offers the other selections that also answer (experiments/alternatives_lab), ``evidence`` adds the
whole table's evidence to a profile, ``memory`` keeps and renders the state, ``items`` turns
the periods and named items of a question into requirements that need evidence, ``tables``
profiles tables and measures how two combine, ``text`` holds the word matching they share. A
selection the agent could not make (no LLM, a failure, a run that ends with nothing selected)
raises ``SelectorUnavailable``, and the round asks for new keywords.
"""

from __future__ import annotations

from pathlib import Path

from lakegen.config.experiment import DiscoveryConfig, TableSelector
from lakegen.datastructures.errors import WorkflowCancelled
from lakegen.llm.prompts import get_prompt_manager
from lakegen.llm.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.pipeline.discovery.selectors import (
    Selection,
    SelectionRequest,
    SelectorUnavailable,
    register,
)
from lakegen.pipeline.discovery.selectors.memgpt import alternatives, contrast, display, drop
from lakegen.pipeline.discovery.selectors.memgpt.agent import MemGPTAgent
from lakegen.pipeline.evidence.requirement_ledger import build_minimal_selection_fallback

TRACE_HEADER = "--- MemGPT Table Selector ---"
SOURCE = "memgpt_selector"
"""What the plan says made it; ``memgpt_fallback`` when the agent stopped before confirming."""


@register(TableSelector.MEMGPT)
class MemGPTSelector:
    """The MemGPT agent's selection, checked against its look-alikes and for tables that hold nothing, with its alternatives.

    Attributes:
        agent_cls: The agent the selector runs (an experiment's variant subclasses it).

    Args:
        discovery: Discovery settings of the run.
    """

    kind = TableSelector.MEMGPT
    title = "MemGPT table selector"
    agent_cls: type[MemGPTAgent] = MemGPTAgent

    def __init__(self, discovery: DiscoveryConfig) -> None:
        self.discovery = discovery

    def select(self, request: SelectionRequest) -> Selection:
        """Run the agent over the candidates, then the contrast, drop and alternatives checks.

        Args:
            request: The question, the candidates and the run's LLM and hooks.

        Returns:
            The selection, the rejection, or the fallback of a run cut short.

        Raises:
            SelectorUnavailable: The request has no LLM or table directory, the
                agent or a check failed, or the run ended with nothing selected.
            WorkflowCancelled: The run was cancelled.
        """
        if request.llm is None or request.table_dir is None:
            raise SelectorUnavailable(
                "the MemGPT selector needs the run's LLM and the table directory"
            )
        prompts = request.prompts or get_prompt_manager()
        agent = self.agent_cls(
            question=request.question,
            candidates=request.candidates,
            metadata=dict(request.metadata),
            table_dir=Path(request.table_dir),
            llm=request.llm,
            prompts=prompts,
            portal=request.portal_name,
            lake=request.lake,
            keywords=request.keywords,
            hint=request.hint,
            question_only=bool(
                request.retrieval_mode and request.retrieval_mode.ranks_question_only
            ),
            emit=request.stream_callback,
            cancel_check=request.cancel_check,
            tool_callback=request.tool_callback,
        )
        if request.tool_callback is not None:
            request.tool_callback(display.candidates(agent))
        elif request.stream_callback is not None:
            request.stream_callback(
                f"\n**Phase 2 – {self.title} started**\n- {len(agent.candidates)} candidates; "
                f"{agent.memory.visible} shown first, the rest by expansion\n"
            )
        reset_llm_token_usage(request.llm)
        try:
            agent.run()
            checked = contrast.check(agent, request.llm, prompts)
            dropped = drop.check(agent, request.llm, prompts)
            offered = alternatives.check(agent, request.llm, prompts)
        except WorkflowCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 -- a selector that breaks asks for new keywords
            raise SelectorUnavailable(
                f"the MemGPT selector failed: {type(exc).__name__}: {exc}"
            ) from exc
        return selection_of(agent, get_llm_token_usage(request.llm), checked, dropped, offered)


def selection_of(
    agent: MemGPTAgent,
    tokens: int,
    checked: contrast.Checked | None = None,
    dropped: drop.Dropped | None = None,
    offered: alternatives.Offered | None = None,
) -> Selection:
    """The finished run as the pipeline's ``Selection``: confirmed, rejected, or the fallback of a run cut short."""
    trace = f"{TRACE_HEADER}\n{agent.trace.getvalue().strip()}\n{agent.summary_line()}"
    calls = agent.turn + sum(check.calls for check in (checked, dropped, offered) if check)
    common = {"trace": trace, "tokens": tokens, "llm_calls": calls}
    if agent.outcome == "confirmed":
        plan, advisories = agent.plan()
        if checked is not None:
            plan["contrast"] = {
                "chosen": checked.chosen.label,
                "alternatives": checked.alternatives,
                "swaps": [list(s) for s in checked.chosen.swaps],
            }
        if dropped is not None:
            plan["drop"] = {"options": dropped.options, "dropped": dropped.dropped}
        if offered is not None:
            plan["alternatives"] = {
                "pool": offered.pool,
                "offered": [a.record() for a in offered.alternatives],
            }
        return Selection(
            selected=agent.selected(),
            reasoning=agent.final_reasoning,
            plan=plan,
            advisories=tuple(advisories),
            source=SOURCE,
            **common,
        )
    if agent.outcome == "rejected":
        reasoning, suggestion = agent.rejection
        return Selection(
            selected=[],
            reasoning=f"REJECT_KEYWORDS: {reasoning}\nSuggestion: {suggestion}",
            keep_tables=tuple(agent.carried_tables()),
            source=SOURCE,
            **common,
        )
    selected = agent.fallback()
    if not selected:
        raise SelectorUnavailable(
            f"the MemGPT selector ended without a selection ({agent.outcome})"
        )
    reasoning = (
        f"The MemGPT selector stopped before confirming ({agent.outcome}); "
        f"it keeps {', '.join(selected)}."
    )
    plan, advisories = (
        agent.plan()
        if agent.memory.selection
        else build_minimal_selection_fallback(selected, reasoning)
    )
    return Selection(
        selected=selected,
        reasoning=reasoning,
        plan=dict(plan),
        advisories=tuple(advisories),
        source="memgpt_fallback",
        **common,
    )
