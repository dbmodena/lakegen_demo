from typing import Any

import streamlit as st

from lakegen_app.state import handle_phase2_keyword_rejection
from lakegen_app.ui.sections import build_phase2_section


def append_phase2_rejection_message(
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
        "phase_sections": [build_phase2_section(approval_hint=hint)],
    })
    return True
