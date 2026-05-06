from typing import Any

import streamlit as st

from lakegen_app.phases import (
    phase3_generate_code,
    phase4_execute,
    phase5_synthesize,
)
from lakegen_app.ui.chat import chat_message
from lakegen_app.ui.sections import (
    build_phase3_section,
    build_phase4_section,
    build_phase5_section,
)
from src.utils import save_experiment_log


def render_execution(llm, pm) -> None:
    with chat_message("assistant"):
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
                    llm, pm, st.session_state.csv_dir,
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
                    "phase_sections": [build_phase3_section(code_attempts)],
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
                st.session_state.query, raw_result, llm, pm)
            st.session_state.tokens["p5"] = tok5
        synthesis_status.success("Final answer ready.")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"### 📊 Final Result\n{answer}",
            "phase_sections": [
                build_phase3_section(code_attempts),
                build_phase4_section(execution_attempts),
                build_phase5_section(raw_result, answer),
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
