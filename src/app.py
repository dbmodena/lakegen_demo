"""
LakeGen Interactive – Streamlit Web Application
Run with:  uv run streamlit run src/app.py
"""

import sys
from pathlib import Path

import streamlit as st

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
_STATIC_DIR = _ROOT_DIR / "static"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from lakegen_app.bootstrap import (  # noqa: E402
    bootstrap_nltk_data,
    ensure_project_paths,
    load_css,
    nltk_download_dir,
)

ensure_project_paths(_SRC_DIR, _ROOT_DIR)
_NLTK_BOOTSTRAP_ERROR = bootstrap_nltk_data()

st.set_page_config(page_title="LakeGen Interactive", page_icon="🌊", layout="wide")

if _NLTK_BOOTSTRAP_ERROR:
    st.error(_NLTK_BOOTSTRAP_ERROR)
    st.code(
        f"uv run python -m nltk.downloader -d {nltk_download_dir()} "
        "wordnet omw-1.4 stopwords"
    )
    st.stop()

from lakegen_app.resources import (  # noqa: E402
    get_all_csv_files,
    get_llms,
    get_prompt_manager,
    get_solr,
)
from lakegen_app.state import init_session_state  # noqa: E402
from lakegen_app.ui import (  # noqa: E402
    render_chat_history,
    render_header,
    render_phase_router,
    render_sidebar,
)


def main() -> None:
    load_css(_STATIC_DIR / "styles.css")
    render_header()
    init_session_state()

    ollama_url, model_name, solr_core, num_ctx = render_sidebar()
    llm_v, llm_i, _tc = get_llms(model_name, ollama_url, num_ctx)
    solr = get_solr(solr_core)
    pm = get_prompt_manager()
    all_csv = get_all_csv_files(st.session_state.csv_dir)

    render_chat_history()
    render_phase_router(llm_v, llm_i, solr, pm, all_csv)


if __name__ == "__main__":
    main()
