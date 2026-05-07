from __future__ import annotations

from typing import Any

from lakegen.ui.i18n import t
from lakegen.ui.state import LakeGenSession
from lakegen.phase2_logging import (
    extract_phase2_activity_log,
    format_phase2_solr_results,
)


def _status_label(status: str) -> str:
    key = status.lower().replace(" ", "_")
    return t(f"status.{key}", default=status)


def format_hint(hint: str) -> str:
    return f"`{hint}`" if hint else t("summary.none")


def build_phase1_summary(session: LakeGenSession, approval_hint: str = "") -> str:
    keywords = ", ".join(f"`{kw}`" for kw in session.keywords)
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


def build_phase2_summary(session: LakeGenSession, approval_hint: str = "") -> str:
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
- {t("summary.total_tokens")}: `{session.tokens['p2']}`
""".strip()


def build_phase3_summary(
    session: LakeGenSession,
    code_attempts: list[dict[str, Any]],
) -> str:
    lines = [f"**{t('summary.code_attempts')}**"]
    if not code_attempts:
        lines.append(t("summary.no_code_attempts"))
    for attempt in code_attempts:
        status = _status_label(attempt.get("status", "generated"))
        feedback = attempt.get("feedback") or ""
        line = (
            f"- {t('summary.attempt')} `{attempt.get('attempt')}`: {status}; "
            f"{t('summary.tokens')}: `{attempt.get('tokens', 0)}`"
        )
        if feedback:
            line += f"; {t('summary.feedback')}: `{feedback}`"
        lines.append(line)
    lines.append(f"\n- {t('summary.total_tokens')}: `{session.tokens['p3']}`")
    return "\n".join(lines)


def build_phase4_summary(execution_attempts: list[dict[str, Any]]) -> str:
    lines = [f"**{t('summary.execution_attempts')}**"]
    if not execution_attempts:
        lines.append(t("summary.no_execution_attempts"))
    for attempt in execution_attempts:
        status = _status_label(attempt.get("status", ""))
        lines.append(f"- {t('summary.attempt')} `{attempt.get('attempt')}`: {status}")
    return "\n".join(lines)


def build_phase5_summary(session: LakeGenSession, answer: str) -> str:
    return f"""
**{t("summary.synthesized_answer")}**

{answer}

- {t("summary.tokens").title()}: `{session.tokens['p5']}`
""".strip()
