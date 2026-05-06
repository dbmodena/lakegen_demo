import streamlit as st

from lakegen_app.phases import (
    phase1_generate_keywords,
    phase1_retrieve_candidates,
    phase2_select_tables,
)
from lakegen_app.resources import make_streamlit_stream_callback
from lakegen_app.ui.chat import chat_message
from lakegen_app.ui.phase2_flow import append_phase2_rejection_message
from lakegen_app.ui.sections import build_phase1_section


PHASE1_TITLE = "Phase 1 - Keyword Generation"


def render_phase1_header() -> None:
    st.markdown(
        '<span class="phase-badge phase-1">PHASE 1</span> **Keyword Generation**',
        unsafe_allow_html=True,
    )


def render_phase1_keywords(keywords: list[str]) -> None:
    chips = " ".join(f'<span class="kw-chip">{k}</span>' for k in keywords)
    st.markdown(f"Extracted keywords: {chips}", unsafe_allow_html=True)


def render_phase1_reasoning_traces() -> None:
    runs = st.session_state.get("phase1_runs", [])
    reasoning_runs = [
        run for run in runs
        if str(run.get("reasoning") or "").strip()
    ]
    if not reasoning_runs:
        return

    st.markdown("**Model reasoning traces**")
    for idx, run in enumerate(reasoning_runs, 1):
        label = run.get("label", f"attempt {idx}")
        st.markdown(f"{idx}. {label}")
        st.code(str(run.get("reasoning") or ""), language="text")


def stream_phase1_generation(
    query: str,
    llm,
    pm,
    *,
    hint: str = "",
    spinner_label: str = "Extracting keywords…",
) -> tuple[list[str], str, int, str]:
    st.markdown("**Keyword stream**")
    stream_box = st.empty()
    st.markdown("**Model reasoning**")
    reasoning_box = st.empty()
    with st.spinner(spinner_label):
        kws, raw, tok, reasoning = phase1_generate_keywords(
            query,
            llm,
            pm,
            hint=hint,
            stream_placeholder=stream_box,
            reasoning_placeholder=reasoning_box,
        )
    reasoning_box.markdown(reasoning or "_No model reasoning streamed by this model._")
    if raw:
        stream_box.markdown(raw)
    render_phase1_keywords(kws)
    return kws, raw, tok, reasoning


def render_phase1_generation_expander(
    query: str,
    llm,
    pm,
    *,
    hint: str = "",
    spinner_label: str = "Extracting keywords…",
) -> tuple[list[str], str, int, str]:
    with st.expander(PHASE1_TITLE, expanded=True):
        render_phase1_header()
        return stream_phase1_generation(
            query,
            llm,
            pm,
            hint=hint,
            spinner_label=spinner_label,
        )


def record_phase1_run(
    label: str,
    hint: str,
    keywords: list[str],
    raw_output: str,
    tokens: int,
    reasoning: str = "",
) -> None:
    st.session_state.phase1_runs.append({
        "label": label,
        "hint": hint,
        "keywords": keywords,
        "raw_output": raw_output,
        "tokens": tokens,
        "reasoning": reasoning,
    })


def render_keyword_approval(llm, solr, pm, all_csv: list[str]) -> None:
    with chat_message("assistant"):
        with st.expander(PHASE1_TITLE, expanded=True):
            render_phase1_header()
            render_phase1_keywords(st.session_state.keywords)

            container = st.empty()
            with container.container():
                hint = st.text_input("Modify keywords? (e.g. 'add inflation')",
                                     key="hint_kw")
                c1, c2 = st.columns(2)
                approve = c1.button("✅ Approve & Proceed", use_container_width=True)
                recalc = c2.button("🔄 Recalculate", use_container_width=True)

            if recalc:
                container.empty()
                kws, raw, tok, reasoning = stream_phase1_generation(
                    st.session_state.query,
                    llm,
                    pm,
                    hint=hint,
                    spinner_label="Recalculating keywords…",
                )
                st.session_state.keywords = kws
                st.session_state.raw_keywords = raw
                st.session_state.tokens["p1"] += tok
                record_phase1_run("Recalculation", hint, kws, raw, tok, reasoning)
                st.rerun()

            render_phase1_reasoning_traces()

        if approve:
            container.empty()
            kw_list = ", ".join(st.session_state.keywords)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"✅ **Keywords confirmed:** `{kw_list}`"
                           + (f"\n*(hint: {hint})*" if hint else ""),
                "phase_sections": [build_phase1_section(approval_hint=hint)],
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
                    llm,
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
                
                if append_phase2_rejection_message(
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


def render_keyword_fallback(llm, pm) -> None:
    with chat_message("assistant"):
        regenerate_keywords = False
        container = None
        with st.expander("Phase 2 - Keyword Feedback", expanded=True):
            st.warning("⚠️ **Keywords Rejected by Data Architect!**\n\nThe Data Architect has analyzed the candidate tables and found none that match the Coder's needs.")
            st.info(f"**Architect Feedback:**\n{st.session_state.fallback_reason}")
            st.markdown("Do you want to allow the Keyword Generator to generate completely new keywords based on this feedback?")

            container = st.empty()
            with container.container():
                regenerate_keywords = st.button("🔄 Generate New Keywords (Back to Phase 1)", use_container_width=True)

        if regenerate_keywords:
            if container is not None:
                container.empty()
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ **Candidates rejected by Architect.**\nFeedback: *{st.session_state.fallback_reason}*\n\n🔄 Re-generating keywords..."
            })
            feedback_hint = (
                "The previous keywords led to bad tables. "
                f"Architect feedback: {st.session_state.fallback_reason}. "
                "Generate COMPLETELY DIFFERENT keywords."
            )
            kws, raw, tok, reasoning = render_phase1_generation_expander(
                st.session_state.query,
                llm,
                pm,
                hint=feedback_hint,
                spinner_label="Recalculating keywords…",
            )
            st.session_state.keywords = kws
            st.session_state.raw_keywords = raw
            st.session_state.tokens["p1"] += tok
            record_phase1_run(
                "Fallback regeneration",
                feedback_hint,
                kws,
                raw,
                tok,
                reasoning,
            )
            st.session_state.phase = "keyword_approval"
            st.rerun()
