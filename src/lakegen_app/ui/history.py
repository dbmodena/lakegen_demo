import streamlit as st

from lakegen_app.ui.chat import chat_message
from lakegen_app.ui.sections import PhaseSection


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


def render_chat_history() -> None:
    for msg in st.session_state.chat_history:
        with chat_message(msg["role"]):
            st.markdown(msg["content"])
            for section in msg.get("phase_sections", []):
                _render_phase_section(section)
            if "code" in msg:
                with st.expander("View generated Python code"):
                    st.code(msg["code"], language="python")
