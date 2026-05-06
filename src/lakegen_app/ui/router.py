import streamlit as st

from lakegen_app.ui.chat import chat_message
from lakegen_app.ui.execution_step import render_execution
from lakegen_app.ui.keyword_step import (
    record_phase1_run,
    render_phase1_generation_expander,
    render_keyword_approval,
    render_keyword_fallback,
)
from lakegen_app.ui.table_step import render_table_approval, render_table_fallback


def render_phase_router(llm, solr, pm, all_csv: list[str]) -> None:
    if st.session_state.phase == "idle":
        query = st.chat_input("Ask a question about your data...")
        if query:
            st.session_state.query = query
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.phase1_runs = []
            st.session_state.tokens = {"p1": 0, "p2": 0, "p3": 0, "p5": 0}
            with chat_message("user"):
                st.markdown(query)
            with chat_message("assistant"):
                kws, raw, tok, reasoning = render_phase1_generation_expander(
                    query,
                    llm,
                    pm,
                    spinner_label="Extracting keywords…",
                )
                st.session_state.keywords = kws
                st.session_state.raw_keywords = raw
                st.session_state.tokens["p1"] = tok
                record_phase1_run("Initial generation", "", kws, raw, tok, reasoning)
            st.session_state.phase = "keyword_approval"
            st.rerun()

    elif st.session_state.phase == "keyword_approval":
        render_keyword_approval(llm, solr, pm, all_csv)

    elif st.session_state.phase == "table_approval":
        render_table_approval(llm, pm, all_csv)

    elif st.session_state.phase == "execution":
        render_execution(llm, pm)

    elif st.session_state.phase == "fallback_approval_tables":
        render_table_fallback(llm, pm, all_csv)

    elif st.session_state.phase == "fallback_approval_keywords":
        render_keyword_fallback(llm, pm)
