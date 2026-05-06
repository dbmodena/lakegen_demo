from typing import Any

import streamlit as st

from lakegen_app.phase2_logging import (
    extract_phase2_activity_log,
    format_phase2_solr_results,
)


PhaseSection = dict[str, Any]


def phase_section(
    title: str,
    body: str = "",
    code_blocks: list[dict[str, str]] | None = None,
) -> PhaseSection:
    section: PhaseSection = {"title": title, "body": body.strip()}
    if code_blocks:
        section["code_blocks"] = code_blocks
    return section


def code_block(
    label: str,
    code: str,
    language: str = "text",
) -> dict[str, str]:
    return {"label": label, "code": code, "language": language}


def format_hint(hint: str) -> str:
    return f"`{hint}`" if hint else "_none_"


def build_phase1_section(approval_hint: str) -> PhaseSection:
    runs = st.session_state.get("phase1_runs", [])
    keywords = ", ".join(f"`{kw}`" for kw in st.session_state.keywords)
    lines = [
        "**Confirmed keywords**",
        keywords or "_No keywords confirmed._",
        "",
        f"- Approval hint: {format_hint(approval_hint)}",
        f"- Total tokens: `{st.session_state.tokens['p1']}`",
    ]

    if runs:
        lines.extend(["", "**Generation attempts**"])
        for idx, run in enumerate(runs, 1):
            run_keywords = ", ".join(f"`{kw}`" for kw in run.get("keywords", []))
            lines.append(
                f"{idx}. {run.get('label', 'Generation')} - "
                f"hint: {format_hint(run.get('hint', ''))}; "
                f"tokens: `{run.get('tokens', 0)}`; "
                f"keywords: {run_keywords or '_none_'}"
            )

    code_blocks = []
    for idx, run in enumerate(runs, 1):
        label = run.get("label", f"attempt {idx}")
        reasoning = str(run.get("reasoning") or "").strip()
        if reasoning:
            code_blocks.append(
                code_block(
                    f"Model reasoning - {label}",
                    reasoning,
                )
            )
        code_blocks.append(
            code_block(
                f"Raw model output - {label}",
                str(run.get("raw_output") or "_No raw output captured._"),
            )
        )
    return phase_section("Phase 1 - Keyword Generation", "\n".join(lines), code_blocks)


def build_phase2_section(approval_hint: str = "") -> PhaseSection:
    selected = "\n".join(f"- `{table}`" for table in st.session_state.tables)
    activity_log = extract_phase2_activity_log(st.session_state.full_trace)
    candidate_log = format_phase2_solr_results(
        st.session_state.candidates,
        st.session_state.solr_metadata_map,
        heading="Candidate tables",
    )

    body = f"""
{candidate_log}

**Selected tables**

{selected or "_No tables selected._"}

**Architect reasoning**

{st.session_state.architect_reasoning or "_No architect reasoning captured._"}

**Activity log**

{activity_log or "_No activity log captured._"}

**Metadata**

- Approval hint: {format_hint(approval_hint)}
- Total tokens: `{st.session_state.tokens['p2']}`
"""
    return phase_section("Phase 2 - Table Selection", body)


def build_phase3_section(code_attempts: list[dict[str, Any]]) -> PhaseSection:
    lines = ["**Code generation attempts**"]
    code_blocks = []

    if not code_attempts:
        lines.append("_No code generation attempts captured._")

    for attempt in code_attempts:
        status = attempt.get("status", "generated")
        feedback = attempt.get("feedback") or ""
        line = (
            f"- Attempt `{attempt.get('attempt')}`: {status}; "
            f"tokens: `{attempt.get('tokens', 0)}`"
        )
        if feedback:
            line += f"; feedback: `{feedback}`"
        lines.append(line)

        code = attempt.get("clean_code") or attempt.get("raw_response") or ""
        language = "python" if attempt.get("clean_code") else "text"
        code_blocks.append(
            code_block(
                f"Generated code - attempt {attempt.get('attempt')}",
                code,
                language=language,
            )
        )

    lines.append(f"\n- Total tokens: `{st.session_state.tokens['p3']}`")
    return phase_section("Phase 3 - Code Generation", "\n".join(lines), code_blocks)


def build_phase4_section(execution_attempts: list[dict[str, Any]]) -> PhaseSection:
    lines = ["**Execution attempts**"]
    code_blocks = []

    if not execution_attempts:
        lines.append("_No execution attempts captured._")

    for attempt in execution_attempts:
        status = attempt.get("status", "unknown")
        lines.append(f"- Attempt `{attempt.get('attempt')}`: {status}")
        output = attempt.get("output") or attempt.get("error") or ""
        if output:
            code_blocks.append(
                code_block(
                    f"Execution {status} - attempt {attempt.get('attempt')}",
                    str(output),
                )
            )

    return phase_section("Phase 4 - Execution", "\n".join(lines), code_blocks)


def build_phase5_section(raw_result: str, answer: str) -> PhaseSection:
    body = f"""
**Synthesized answer**

{answer}

- Tokens: `{st.session_state.tokens['p5']}`
"""
    return phase_section(
        "Phase 5 - Synthesis",
        body,
        [code_block("Raw execution result", raw_result or "_No raw result captured._")],
    )
