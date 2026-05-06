from typing import Any

import streamlit as st

from lakegen_app.phase2_logging import (
    extract_phase2_activity_log,
    format_phase2_solr_results,
)
from lakegen_app.phases import (
    phase1_generate_keywords,
    phase1_retrieve_candidates,
    phase2_select_tables,
    phase3_generate_code,
    phase4_execute,
    phase5_synthesize,
)
from lakegen_app.resources import make_streamlit_stream_callback
from lakegen_app.state import handle_phase2_keyword_rejection, reset_conversation_state
from src.utils import BASE_DIR, save_experiment_log


PhaseSection = dict[str, Any]


def _phase_section(
    title: str,
    body: str = "",
    code_blocks: list[dict[str, str]] | None = None,
) -> PhaseSection:
    section: PhaseSection = {"title": title, "body": body.strip()}
    if code_blocks:
        section["code_blocks"] = code_blocks
    return section


def _render_phase_section(section: PhaseSection) -> None:
    with st.expander(str(section.get("title", "Phase details")), expanded=False):
        body = str(section.get("body") or "").strip()
        if body:
            st.markdown(body)

        for code_block in section.get("code_blocks", []):
            label = code_block.get("label", "")
            code = code_block.get("code", "")
            language = code_block.get("language", "text")
            if label:
                st.markdown(label)
            if code:
                st.code(code, language=language)


def _code_block(
    label: str,
    code: str,
    language: str = "text",
) -> dict[str, str]:
    return {"label": label, "code": code, "language": language}


def _format_hint(hint: str) -> str:
    return f"`{hint}`" if hint else "_none_"


def render_header() -> None:
    st.markdown("""
<div class="header-bar">
    <div class="header-title-row">
        <img class="header-logo" src="app/static/favicon.png" alt="LakeGen logo">
        <h1>LakeGen — Data Assistant</h1>
    </div>
    <p>Ask natural-language questions over your Data Lake</p>
</div>
""", unsafe_allow_html=True)


def render_sidebar() -> tuple[str, str, str, int]:
    with st.sidebar:
        st.header("⚙️ Configuration")

        ollama_url = st.text_input("Ollama Server URL", value="http://127.0.0.1:11434")
        model_name = st.selectbox("Model", [
            "gemma4:26b", "qwen3.5:latest", "llama3.1:8b", "gpt-oss:20b"
        ])
        solr_core = st.selectbox("Open Data Lake", ["nyc", "valencia", "bologna", "paris"])
        num_ctx = st.slider("Context window", 4096, 32768, 12288, step=1024)

        st.session_state.csv_dir = BASE_DIR / f"data/{solr_core}/datasets/csv"
        st.session_state.db_path = BASE_DIR / f"data/blend_{solr_core}.db"

        st.divider()
        if st.button("🔄 Reset Conversation", use_container_width=True):
            reset_conversation_state()
            st.rerun()

    return ollama_url, model_name, solr_core, num_ctx


def render_chat_history() -> None:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for section in msg.get("phase_sections", []):
                _render_phase_section(section)
            if "code" in msg:
                with st.expander("View generated Python code"):
                    st.code(msg["code"], language="python")


def render_phase_router(llm_v, llm_i, solr, pm, all_csv: list[str]) -> None:
    if st.session_state.phase == "idle":
        query = st.chat_input("Ask a question about your data...")
        if query:
            st.session_state.query = query
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.phase1_runs = []
            st.session_state.tokens = {"p1": 0, "p2": 0, "p3": 0, "p5": 0}
            with st.chat_message("user"):
                st.markdown(query)
            with st.chat_message("assistant"):
                stream_box = st.empty()
                with st.spinner("🔍 Extracting keywords…"):
                    kws, raw, tok = phase1_generate_keywords(
                        query, llm_i, pm, stream_placeholder=stream_box)
                    st.session_state.keywords = kws
                    st.session_state.raw_keywords = raw
                    st.session_state.tokens["p1"] = tok
                    _record_phase1_run("Initial generation", "", kws, raw, tok)
            st.session_state.phase = "keyword_approval"
            st.rerun()

    elif st.session_state.phase == "keyword_approval":
        _render_keyword_approval(llm_i, llm_v, solr, pm, all_csv)

    elif st.session_state.phase == "table_approval":
        _render_table_approval(llm_v, pm, all_csv)

    elif st.session_state.phase == "execution":
        _render_execution(llm_v, llm_i, pm)

    elif st.session_state.phase == "fallback_approval_tables":
        _render_table_fallback(llm_v, pm, all_csv)

    elif st.session_state.phase == "fallback_approval_keywords":
        _render_keyword_fallback(llm_i, pm)


def _record_phase1_run(
    label: str,
    hint: str,
    keywords: list[str],
    raw_output: str,
    tokens: int,
) -> None:
    st.session_state.phase1_runs.append({
        "label": label,
        "hint": hint,
        "keywords": keywords,
        "raw_output": raw_output,
        "tokens": tokens,
    })


def _build_phase1_section(approval_hint: str) -> PhaseSection:
    runs = st.session_state.get("phase1_runs", [])
    keywords = ", ".join(f"`{kw}`" for kw in st.session_state.keywords)
    lines = [
        "**Confirmed keywords**",
        keywords or "_No keywords confirmed._",
        "",
        f"- Approval hint: {_format_hint(approval_hint)}",
        f"- Total tokens: `{st.session_state.tokens['p1']}`",
    ]

    if runs:
        lines.extend(["", "**Generation attempts**"])
        for idx, run in enumerate(runs, 1):
            run_keywords = ", ".join(f"`{kw}`" for kw in run.get("keywords", []))
            lines.append(
                f"{idx}. {run.get('label', 'Generation')} - "
                f"hint: {_format_hint(run.get('hint', ''))}; "
                f"tokens: `{run.get('tokens', 0)}`; "
                f"keywords: {run_keywords or '_none_'}"
            )

    code_blocks = [
        _code_block(
            f"Raw model output - {run.get('label', f'attempt {idx}')}",
            str(run.get("raw_output") or "_No raw output captured._"),
        )
        for idx, run in enumerate(runs, 1)
    ]
    return _phase_section("Phase 1 - Keyword Generation", "\n".join(lines), code_blocks)


def _render_keyword_approval(llm_i, llm_v, solr, pm, all_csv: list[str]) -> None:
    with st.chat_message("assistant"):
        with st.expander("Phase 1 - Keyword Generation", expanded=True):
            st.markdown('<span class="phase-badge phase-1">PHASE 1</span> **Keyword Generation**',
                        unsafe_allow_html=True)
            chips = " ".join(f'<span class="kw-chip">{k}</span>'
                             for k in st.session_state.keywords)
            st.markdown(f"Extracted keywords: {chips}", unsafe_allow_html=True)

            container = st.empty()
            with container.container():
                hint = st.text_input("Modify keywords? (e.g. 'add inflation')",
                                     key="hint_kw")
                c1, c2 = st.columns(2)
                approve = c1.button("✅ Approve & Proceed", use_container_width=True)
                recalc = c2.button("🔄 Recalculate", use_container_width=True)

        if approve:
            container.empty()
            kw_list = ", ".join(st.session_state.keywords)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"✅ **Keywords confirmed:** `{kw_list}`"
                           + (f"\n*(hint: {hint})*" if hint else ""),
                "phase_sections": [_build_phase1_section(approval_hint=hint)],
            })
            stream_box = st.empty()
            with st.spinner("🗂️ Searching & selecting tables…"):
                stream_callback = make_streamlit_stream_callback(stream_box)
                cands, smeta, activity_log_parts = phase1_retrieve_candidates(
                    st.session_state.keywords,
                    solr,
                    all_csv,
                    stream_callback=stream_callback,
                )
                sel, cands, smeta, reasoning, trace, tok2 = phase2_select_tables(
                    st.session_state.query,
                    llm_v,
                    pm,
                    all_csv,
                    cands,
                    smeta,
                    st.session_state.csv_dir,
                    st.session_state.db_path,
                    activity_log_parts=activity_log_parts,
                    hint=hint,
                    stream_callback=stream_callback,
                )
                if _append_phase2_rejection_message(
                    cands, smeta, reasoning, trace, tok2,
                    accumulate_tokens=False,
                    hint=hint,
                ):
                    st.rerun()

                st.session_state.tables = sel
                st.session_state.candidates = cands
                st.session_state.solr_metadata_map = smeta
                st.session_state.architect_reasoning = reasoning
                st.session_state.full_trace = trace
                st.session_state.tokens["p2"] = tok2
            st.session_state.phase = "table_approval"
            st.rerun()

        elif recalc:
            container.empty()
            stream_box = st.empty()
            with st.spinner("Recalculating keywords…"):
                kws, raw, tok = phase1_generate_keywords(
                    st.session_state.query, llm_i, pm, hint=hint,
                    stream_placeholder=stream_box)
                st.session_state.keywords = kws
                st.session_state.raw_keywords = raw
                st.session_state.tokens["p1"] += tok
                _record_phase1_run("Recalculation", hint, kws, raw, tok)
            st.rerun()


def _build_phase2_section(approval_hint: str = "") -> PhaseSection:
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

- Approval hint: {_format_hint(approval_hint)}
- Total tokens: `{st.session_state.tokens['p2']}`
"""
    return _phase_section("Phase 2 - Table Selection", body)


def _append_phase2_rejection_message(
    candidates: list[str],
    solr_metadata: dict[str, dict[str, Any]],
    reasoning: str,
    trace: str,
    tokens: int,
    accumulate_tokens: bool,
    hint: str,
) -> bool:
    rejected = handle_phase2_keyword_rejection(
        candidates,
        solr_metadata,
        reasoning,
        trace,
        tokens,
        accumulate_tokens=accumulate_tokens,
    )
    if not rejected:
        return False

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": (
            "⚠️ **Keywords rejected by Data Architect.**\n"
            f"Feedback: *{st.session_state.fallback_reason}*"
        ),
        "phase_sections": [_build_phase2_section(approval_hint=hint)],
    })
    return True


def _render_table_approval(llm_v, pm, all_csv: list[str]) -> None:
    with st.chat_message("assistant"):
        section = _build_phase2_section()
        with st.expander(str(section["title"]), expanded=True):
            st.markdown('<span class="phase-badge phase-2">PHASE 2</span> **Table Selection**',
                        unsafe_allow_html=True)
            st.markdown(str(section.get("body") or ""))

        container = st.empty()
        with container.container():
            hint_tb = st.text_input("Suggestions? (e.g. 'Add 2019.csv')",
                                    key="hint_tb")
            c1, c2 = st.columns(2)
            approve = c1.button("🚀 Approve & Run Code",
                                use_container_width=True)
            recalc = c2.button("🔄 Recalculate Tables",
                               use_container_width=True)

        if approve:
            container.empty()
            tbl_md = "\n".join(f"- `{t}`" for t in st.session_state.tables)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"✅ **Tables confirmed:**\n{tbl_md}"
                           + (f"\n*(hint: {hint_tb})*" if hint_tb else ""),
                "phase_sections": [_build_phase2_section(approval_hint=hint_tb)],
            })
            st.session_state.phase = "execution"
            st.rerun()

        elif recalc:
            container.empty()
            stream_box = st.empty()
            with st.spinner("Re-selecting tables…"):
                sel, cands, smeta, reasoning, trace, tok2 = phase2_select_tables(
                    st.session_state.query,
                    llm_v,
                    pm,
                    all_csv,
                    st.session_state.candidates,
                    st.session_state.solr_metadata_map,
                    st.session_state.csv_dir,
                    st.session_state.db_path,
                    hint=hint_tb,
                    stream_callback=make_streamlit_stream_callback(stream_box),
                )
                if _append_phase2_rejection_message(
                    cands, smeta, reasoning, trace, tok2,
                    accumulate_tokens=True,
                    hint=hint_tb,
                ):
                    st.rerun()

                st.session_state.tables = sel
                st.session_state.candidates = cands
                st.session_state.solr_metadata_map = smeta
                st.session_state.architect_reasoning = reasoning
                st.session_state.full_trace = trace
                st.session_state.tokens["p2"] += tok2
            st.rerun()


def _build_phase3_section(code_attempts: list[dict[str, Any]]) -> PhaseSection:
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
            _code_block(
                f"Generated code - attempt {attempt.get('attempt')}",
                code,
                language=language,
            )
        )

    lines.append(f"\n- Total tokens: `{st.session_state.tokens['p3']}`")
    return _phase_section("Phase 3 - Code Generation", "\n".join(lines), code_blocks)


def _build_phase4_section(execution_attempts: list[dict[str, Any]]) -> PhaseSection:
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
                _code_block(
                    f"Execution {status} - attempt {attempt.get('attempt')}",
                    str(output),
                )
            )

    return _phase_section("Phase 4 - Execution", "\n".join(lines), code_blocks)


def _build_phase5_section(raw_result: str, answer: str) -> PhaseSection:
    body = f"""
**Synthesized answer**

{answer}

- Tokens: `{st.session_state.tokens['p5']}`
"""
    return _phase_section(
        "Phase 5 - Synthesis",
        body,
        [_code_block("Raw execution result", raw_result or "_No raw result captured._")],
    )


def _render_execution(llm_v, llm_i, pm) -> None:
    with st.chat_message("assistant"):
        max_retries = 3
        retries = 0
        error_msg = ""
        final_code = ""
        raw_result = None
        err = None
        code_attempts: list[dict[str, Any]] = []
        execution_attempts: list[dict[str, Any]] = []

        with st.expander("Phase 3 - Code Generation", expanded=True):
            progress = st.empty()
        with st.expander("Phase 4 - Execution", expanded=False):
            execution_status = st.empty()
        with st.expander("Phase 5 - Synthesis", expanded=False):
            synthesis_status = st.empty()

        while retries < max_retries:
            progress.markdown(
                f'<span class="phase-badge phase-3">PHASE 3</span> '
                f'Generating code… (attempt {retries+1}/{max_retries})',
                unsafe_allow_html=True)

            generation_feedback = error_msg
            with st.spinner("✍️ Writing code…"):
                code_raw, tok3 = phase3_generate_code(
                    st.session_state.query, st.session_state.tables,
                    st.session_state.candidates,
                    st.session_state.solr_metadata_map,
                    st.session_state.architect_reasoning,
                    llm_v, pm, st.session_state.csv_dir,
                    retries=retries, error_msg=error_msg)
                st.session_state.tokens["p3"] += tok3

            generation_attempt = {
                "attempt": retries + 1,
                "feedback": generation_feedback,
                "raw_response": code_raw,
                "tokens": tok3,
                "status": "generated",
            }

            if "REJECT_TABLES" in code_raw:
                reason = code_raw.replace("REJECT_TABLES:", "").replace("REJECT_TABLES", "").strip()
                st.session_state.fallback_reason = reason
                generation_attempt["status"] = "rejected tables"
                code_attempts.append(generation_attempt)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": (
                        "⚠️ **Tables rejected by Code Generator.**\n"
                        f"Feedback: *{reason}*"
                    ),
                    "phase_sections": [_build_phase3_section(code_attempts)],
                })
                st.session_state.phase = "fallback_approval_tables"
                st.rerun()

            execution_status.markdown(
                f'<span class="phase-badge phase-4">PHASE 4</span> '
                f'Executing script… (attempt {retries+1}/{max_retries})',
                unsafe_allow_html=True)
            with st.spinner("⚡ Executing script…"):
                raw_result, err, clean_code = phase4_execute(code_raw)
                final_code = clean_code
                generation_attempt["clean_code"] = clean_code
                code_attempts.append(generation_attempt)

            if err is None:
                execution_attempts.append({
                    "attempt": retries + 1,
                    "status": "success",
                    "output": raw_result or "",
                })
                execution_status.success("Script executed successfully.")
                break
            execution_attempts.append({
                "attempt": retries + 1,
                "status": "error",
                "error": err,
            })
            execution_status.error(f"Execution failed. Retrying with feedback: {err}")
            error_msg = err
            retries += 1

        if raw_result is None:
            raw_result = f"Execution failed after {max_retries} attempts. Last error: {error_msg}"

        synthesis_status.markdown(
            '<span class="phase-badge phase-5">PHASE 5</span> Synthesizing final answer…',
            unsafe_allow_html=True)
        with st.spinner("📝 Synthesizing final answer…"):
            answer, tok5 = phase5_synthesize(
                st.session_state.query, raw_result, llm_i, pm)
            st.session_state.tokens["p5"] = tok5
        synthesis_status.success("Final answer ready.")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"### 📊 Final Result\n{answer}",
            "phase_sections": [
                _build_phase3_section(code_attempts),
                _build_phase4_section(execution_attempts),
                _build_phase5_section(raw_result, answer),
            ],
            "code": final_code,
        })

        save_experiment_log(
            question=st.session_state.query,
            code=final_code,
            result=raw_result if raw_result else "",
            retries=retries,
            reasoning=st.session_state.architect_reasoning,
            tables=st.session_state.tables,
            raw_keywords=st.session_state.raw_keywords,
            final_keywords=st.session_state.keywords,
            debug_raw="",
            final_result=answer,
            full_trace=st.session_state.full_trace,
            tokens_phase1=st.session_state.tokens["p1"],
            tokens_phase2=st.session_state.tokens["p2"],
            tokens_phase3=st.session_state.tokens["p3"],
            tokens_phase5=st.session_state.tokens["p5"],
            error=err if err is not None else ""
        )

        st.session_state.phase = "idle"
        st.rerun()


def _render_table_fallback(llm_v, pm, all_csv: list[str]) -> None:
    with st.chat_message("assistant"):
        with st.expander("Phase 3 - Table Validation Feedback", expanded=True):
            st.warning("⚠️ **Tables Rejected by Coder!**\n\nThe Coder has analyzed the selected tables and determined they do not contain the necessary data.")
            st.info(f"**Coder Feedback:**\n{st.session_state.fallback_reason}")
            st.markdown("Do you want to allow the Data Architect to re-evaluate the candidate tables based on this feedback, or do you want to force execution ignoring the warning?")

            container = st.empty()
            with container.container():
                c1, c2 = st.columns(2)
                allow_recheck = c1.button("🔄 Allow Re-evaluation (Back to Phase 2)", use_container_width=True)
                force_execution = c2.button("⚠️ Force Execution (Ignore warning)", use_container_width=True)

            if allow_recheck:
                st.session_state.force_execution = False
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ **Tables rejected by Coder.**\nFeedback: *{st.session_state.fallback_reason}*\n\n🔄 Re-evaluating candidate tables..."
                })
                stream_box = st.empty()
                with st.spinner("Re-selecting tables…"):
                    sel, cands, smeta, reasoning, trace, tok2 = phase2_select_tables(
                        st.session_state.query,
                        llm_v,
                        pm,
                        all_csv,
                        st.session_state.candidates,
                        st.session_state.solr_metadata_map,
                        st.session_state.csv_dir, st.session_state.db_path,
                        hint=f"PREVIOUS SELECTION REJECTED. CODER FEEDBACK: {st.session_state.fallback_reason}",
                        stream_callback=make_streamlit_stream_callback(stream_box),
                    )

                    if _append_phase2_rejection_message(
                        cands, smeta, reasoning, trace, tok2,
                        accumulate_tokens=True,
                        hint=(
                            "PREVIOUS SELECTION REJECTED. CODER FEEDBACK: "
                            f"{st.session_state.fallback_reason}"
                        ),
                    ):
                        st.rerun()

                    st.session_state.tables = sel
                    st.session_state.candidates = cands
                    st.session_state.solr_metadata_map = smeta
                    st.session_state.architect_reasoning = reasoning
                    st.session_state.full_trace = trace
                    st.session_state.tokens["p2"] += tok2
                st.session_state.phase = "table_approval"
                st.rerun()

            if force_execution:
                st.session_state.force_execution = True
                st.session_state.phase = "execution"
                st.rerun()


def _render_keyword_fallback(llm_i, pm) -> None:
    with st.chat_message("assistant"):
        with st.expander("Phase 2 - Keyword Feedback", expanded=True):
            st.warning("⚠️ **Keywords Rejected by Data Architect!**\n\nThe Data Architect has analyzed the candidate tables and found none that match the Coder's needs.")
            st.info(f"**Architect Feedback:**\n{st.session_state.fallback_reason}")
            st.markdown("Do you want to allow the Keyword Generator to generate completely new keywords based on this feedback?")

            container = st.empty()
            with container.container():
                regenerate_keywords = st.button("🔄 Generate New Keywords (Back to Phase 1)", use_container_width=True)

            if regenerate_keywords:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ **Candidates rejected by Architect.**\nFeedback: *{st.session_state.fallback_reason}*\n\n🔄 Re-generating keywords..."
                })
                stream_box = st.empty()
                with st.spinner("Recalculating keywords…"):
                    feedback_hint = (
                        "The previous keywords led to bad tables. "
                        f"Architect feedback: {st.session_state.fallback_reason}. "
                        "Generate COMPLETELY DIFFERENT keywords."
                    )
                    kws, raw, tok = phase1_generate_keywords(
                        st.session_state.query, llm_i, pm,
                        hint=feedback_hint,
                        stream_placeholder=stream_box)
                    st.session_state.keywords = kws
                    st.session_state.raw_keywords = raw
                    st.session_state.tokens["p1"] += tok
                    _record_phase1_run(
                        "Fallback regeneration",
                        feedback_hint,
                        kws,
                        raw,
                        tok,
                    )
                st.session_state.phase = "keyword_approval"
                st.rerun()
