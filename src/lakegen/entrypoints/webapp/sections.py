"""The Markdown summaries the chat shows for each phase of a run, read from the session."""

from __future__ import annotations

from typing import Any

from lakegen.entrypoints.webapp.i18n import t
from lakegen.entrypoints.webapp.state import LakeGenSession
from lakegen.llm.agent_activity import (
    extract_phase2_activity_log,
    format_phase2_solr_results,
)
from lakegen.pipeline.discovery.selectors.memgpt import TRACE_HEADER as MEMGPT_TRACE


def _status_label(status: str) -> str:
    key = status.lower().replace(" ", "_")
    return t(f"status.{key}", default=status)


def _fenced_text(content: str) -> str:
    fence = "```"
    while fence in content:
        fence += "`"
    return f"{fence}text\n{content}\n{fence}"


def format_hint(hint: str) -> str:
    """The person's hint in backticks, or the "none" marker when there is none."""
    return f"`{hint}`" if hint else t("summary.none")


def format_retrieval_keywords(session: LakeGenSession, keywords: list[str]) -> str:
    """Keywords as the chat shows them, flagged when retrieval never searches them."""
    listed = ", ".join(f"`{kw}`" for kw in keywords) or t("summary.none")
    if session.runtime.retrieval.mode.ranks_question_only:
        return t("summary.question_only_keywords", keywords=listed)
    return listed


def build_phase1_summary(session: LakeGenSession, approval_hint: str = "") -> str:
    """The keyword step's summary: the confirmed keywords, the hint, the tokens and each round.

    Args:
        session: The question's session.
        approval_hint: The hint the person gave for the approved round.

    Returns:
        The summary as Markdown.
    """
    keywords = ", ".join(f"`{kw}`" for kw in session.keywords)
    if session.runtime.retrieval.mode.ranks_question_only:
        keywords = format_retrieval_keywords(session, session.keywords)
    lines = [
        f"**{t('summary.confirmed_keywords')}**",
        keywords or t("summary.no_keywords"),
        "",
        f"- {t('summary.approval_hint')}: {format_hint(approval_hint)}",
        f"- {t('summary.total_tokens')}: `{session.tokens['p1']}`",
    ]
    if session.phase1_runs:
        lines.extend(["", f"**{t('summary.generation_attempts')}**"])
        for idx, run in enumerate(session.phase1_runs, 1):
            run_keywords = ", ".join(f"`{kw}`" for kw in run.get("keywords", []))
            lines.append(
                f"{idx}. {run.get('label', 'Generation')} - "
                f"{t('summary.hint')}: "
                f"{format_hint(run.get('hint', ''))}; "
                f"{t('summary.tokens')}: `{run.get('tokens', 0)}`; "
                f"{t('summary.keywords')}: "
                f"{run_keywords or t('summary.none')}"
            )
    return "\n".join(lines)


def is_memgpt_trace(trace: str) -> bool:
    """Whether the selection came from the MemGPT selector, whose tool calls the step shows one by one."""
    return MEMGPT_TRACE in (trace or "")[:400]


def _titled(session: LakeGenSession, table: str) -> str:
    title = " ".join(str(session.solr_metadata_map.get(table, {}).get("title") or "").split())
    return f"**{title}** · `{table}`" if title else f"`{table}`"


def alternatives_block(session: LakeGenSession) -> str:
    """The other selections the selector offers, each with what differs; empty when it offers none."""
    lines = []
    for alt in session.alternatives:
        tables = ", ".join(_titled(session, table) for table in alt.get("tables", []))
        note = " ".join(str(alt.get("note") or "").split())
        lines.append(f"- {tables}" + (f" — {note}" if note else ""))
    if not lines:
        return ""
    return f"**{t('summary.alternatives')}**\n\n" + "\n".join(lines) + "\n\n"


def build_memgpt_summary(session: LakeGenSession, approval_hint: str = "") -> str:
    """The MemGPT selector's outcome: the tables by title, its alternatives and why; the candidates and its calls are the steps above."""
    selected = "\n".join(f"- {_titled(session, table)}" for table in session.tables)
    return f"""
**{t("summary.selected_tables")}**

{selected or t("summary.no_tables")}

{alternatives_block(session)}**{t("summary.selector_reasoning")}**

{session.architect_reasoning or t("summary.no_architect_reasoning")}

_{t("summary.selector_steps")}_

- {t("summary.approval_hint")}: {format_hint(approval_hint)}
- {t("summary.total_tokens")}: `{session.tokens["p2"]}`
""".strip()


def build_phase2_summary(session: LakeGenSession, approval_hint: str = "") -> str:
    """The selection step's summary: the candidates, the tables chosen, the reasoning and the log.

    A MemGPT selection gets ``build_memgpt_summary`` instead, since its
    candidates and calls are shown as steps of their own.

    Args:
        session: The question's session.
        approval_hint: The hint the person gave for the approved round.

    Returns:
        The summary as Markdown.
    """
    if is_memgpt_trace(session.full_trace):
        return build_memgpt_summary(session, approval_hint)
    selected = "\n".join(f"- `{table}`" for table in session.tables)
    activity_log = extract_phase2_activity_log(session.full_trace)
    candidate_log = format_phase2_solr_results(
        session.candidates,
        session.solr_metadata_map,
        heading=t("summary.candidate_tables"),
    )
    return f"""
{candidate_log}

**{t("summary.selected_tables")}**

{selected or t("summary.no_tables")}

**{t("summary.architect_reasoning")}**

{session.architect_reasoning or t("summary.no_architect_reasoning")}

**{t("summary.activity_log")}**

{activity_log or t("summary.no_activity_log")}

**{t("summary.metadata")}**

- {t("summary.approval_hint")}: {format_hint(approval_hint)}
- {t("summary.total_tokens")}: `{session.tokens["p2"]}`
""".strip()


def build_phase3_summary(
    session: LakeGenSession,
    code_attempts: list[dict[str, Any]],
) -> str:
    """The coding step's summary: each attempt with its status, tokens and error.

    Args:
        session: The question's session.
        code_attempts: The attempts as the driver records them.

    Returns:
        The summary as Markdown.
    """
    lines = [f"**{t('summary.code_attempts')}**"]
    if not code_attempts:
        lines.append(t("summary.no_code_attempts"))
    for attempt in code_attempts:
        status = _status_label(attempt.get("status", "generated"))
        error = attempt.get("error") or ""
        line = (
            f"- {t('summary.attempt')} `{attempt.get('attempt')}`: {status}; "
            f"{t('summary.tokens')}: `{attempt.get('tokens', 0)}`"
        )
        if error:
            line += f"\n\n{t('summary.error')}:\n\n{_fenced_text(error)}"
        lines.append(line)
    lines.append(f"\n- {t('summary.total_tokens')}: `{session.tokens['p3']}`")
    return "\n".join(lines)


def build_phase4_summary(session: LakeGenSession, answer: str) -> str:
    """The answer step's summary: the tokens the synthesis spent (the answer itself is shown above it)."""
    return f"""
**{t("summary.synthesized_answer")}**

- {t("summary.tokens").title()}: `{session.tokens["p4"]}`
""".strip()
