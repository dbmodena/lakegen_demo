from copy import deepcopy

import streamlit as st

from lakegen_app.types import SolrMetadata


DEFAULTS = {
    "phase": "idle",
    "query": "",
    "keywords": [],
    "raw_keywords": "",
    "tables": [],
    "candidates": [],
    "chat_history": [],
    "phase1_runs": [],
    "architect_reasoning": "",
    "full_trace": "",
    "solr_metadata_map": {},
    "fallback_reason": "",
    "force_execution": False,
    "tokens": {"p1": 0, "p2": 0, "p3": 0, "p5": 0},
}


def init_session_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = deepcopy(value)


def reset_conversation_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = deepcopy(value)


def get_keyword_rejection_reason(reasoning: str) -> str | None:
    if not reasoning.startswith("REJECT_KEYWORDS"):
        return None
    reason = reasoning.replace("REJECT_KEYWORDS:", "", 1)
    reason = reason.replace("REJECT_KEYWORDS", "", 1).strip()
    return reason or "The candidate tables did not match the generated keywords."


def handle_phase2_keyword_rejection(
    candidates: list[str],
    solr_metadata: SolrMetadata,
    reasoning: str,
    trace: str,
    tokens: int,
    accumulate_tokens: bool,
) -> bool:
    rejection_reason = get_keyword_rejection_reason(reasoning)
    if rejection_reason is None:
        return False

    st.session_state.tables = []
    st.session_state.candidates = candidates
    st.session_state.solr_metadata_map = solr_metadata
    st.session_state.architect_reasoning = reasoning
    st.session_state.full_trace = trace
    st.session_state.fallback_reason = rejection_reason
    if accumulate_tokens:
        st.session_state.tokens["p2"] += tokens
    else:
        st.session_state.tokens["p2"] = tokens
    st.session_state.phase = "fallback_approval_keywords"
    return True
