import streamlit as st

from lakegen_app.state import reset_conversation_state
from src.utils import BASE_DIR


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


def render_sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.header("⚙️ Configuration")

        ollama_url = st.text_input("Ollama Server URL", value="http://127.0.0.1:11434")
        model_name = st.selectbox("Model", [
            "gemma4:26b", "qwen3.5:latest", "llama3.1:8b", "gpt-oss:20b"
        ])
        solr_core = st.selectbox("Open Data Lake", ["nyc", "valencia", "bologna", "paris"])

        st.session_state.csv_dir = BASE_DIR / f"data/{solr_core}/datasets/csv"
        st.session_state.db_path = BASE_DIR / f"data/blend_{solr_core}.db"

        st.divider()
        if st.button("🔄 Reset Conversation", use_container_width=True):
            reset_conversation_state()
            st.rerun()

    return ollama_url, model_name, solr_core
