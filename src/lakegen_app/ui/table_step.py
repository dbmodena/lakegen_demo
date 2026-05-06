import streamlit as st

from lakegen_app.phases import phase2_select_tables
from lakegen_app.resources import make_streamlit_stream_callback
from lakegen_app.ui.chat import chat_message
from lakegen_app.ui.phase2_flow import append_phase2_rejection_message
from lakegen_app.ui.sections import build_phase2_section


def render_table_approval(llm, pm, all_csv: list[str]) -> None:
    with chat_message("assistant"):
        section = build_phase2_section()
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
                "phase_sections": [build_phase2_section(approval_hint=hint_tb)],
            })
            st.session_state.phase = "execution"
            st.rerun()

        elif recalc:
            container.empty()
            stream_box = st.empty()
            with st.spinner("Re-selecting tables…"):
                sel, cands, smeta, reasoning, trace, tok2 = phase2_select_tables(
                    st.session_state.query,
                    llm,
                    pm,
                    all_csv,
                    st.session_state.candidates,
                    st.session_state.solr_metadata_map,
                    st.session_state.csv_dir,
                    st.session_state.db_path,
                    hint=hint_tb,
                    stream_callback=make_streamlit_stream_callback(stream_box),
                )
                if append_phase2_rejection_message(
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


def render_table_fallback(llm, pm, all_csv: list[str]) -> None:
    with chat_message("assistant"):
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
                        llm,
                        pm,
                        all_csv,
                        st.session_state.candidates,
                        st.session_state.solr_metadata_map,
                        st.session_state.csv_dir, st.session_state.db_path,
                        hint=f"PREVIOUS SELECTION REJECTED. CODER FEEDBACK: {st.session_state.fallback_reason}",
                        stream_callback=make_streamlit_stream_callback(stream_box),
                    )

                    if append_phase2_rejection_message(
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
