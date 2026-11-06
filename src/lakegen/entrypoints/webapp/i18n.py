"""The Chainlit app's messages, by key, and ``t`` to format one."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MESSAGES: dict[str, str] = {
    "app.title": "# LakeGen - Data Assistant",
    "app.intro": "Ask a natural-language question about the selected Open Data portal.",
    "app.settings_updated": "Settings updated: `{model_name}` on `{solr_core}` (Agent Mode: `{agent_mode}`).",
    "settings.model": "Model",
    "settings.solr_core": "Open Data Lake",
    "hint.skip_suffix": "Reply with `skip` for no hint.",
    "phase1.keyword_stream": "Keyword stream",
    "phase1.model_reasoning": "Model reasoning",
    "phase1.step": "Phase 1 - Keyword Generation",
    "phase1.initial_generation": "Initial generation",
    "phase1.fallback_regeneration": "Fallback regeneration",
    "phase1.recalculation": "Recalculation",
    "phase1.review_keywords": "Review extracted keywords:\n\n{keywords}",
    "phase1.approve": "Approve and proceed",
    "phase1.recalculate": "Recalculate",
    "phase1.change_hint": "What should the keyword generator change?",
    "phase2.step": "Phase 2 - Table Selection",
    "phase2.keywords_rejected": "Keywords rejected: {reason}",
    "phase2.architect_rejected": (
        "The Data Architect rejected the generated keywords.\n\nFeedback: {feedback}"
    ),
    "phase2.generate_keywords": "Generate new keywords",
    "phase2.review_tables": "Review selected tables:\n\n{tables}",
    "phase2.approve": "Approve and run code",
    "phase2.recalculate": "Recalculate tables",
    "phase2.change_hint": "What should the Data Architect change?",
    "phase3.step": "Phase 3 - Code Generation & Execution",
    "phase3.code_stream": "Generated code",
    "phase3.model_reasoning": "Model reasoning",
    "phase3.success": "Script executed successfully.",
    "phase4.step": "Phase 4 - Synthesis",
    "result.final": "Final Result",
    "workflow.cancelled": "Workflow cancelled before execution.",
    "workflow.empty_question": "Ask a data question to start a LakeGen run.",
    "workflow.locked": (
        "Another LakeGen workflow is running. I will start this one as soon as it finishes."
    ),
    "workflow.tables_rejected": (
        "The Code Generator rejected the selected tables.\n\nFeedback: {feedback}"
    ),
    "workflow.reevaluate_tables": "Re-evaluate tables",
    "workflow.force_execution": "Force execution",
    "summary.none": "_none_",
    "summary.no_keywords": "_No keywords confirmed._",
    "summary.confirmed_keywords": "Confirmed keywords",
    "summary.approval_hint": "Approval hint",
    "summary.total_tokens": "Total tokens",
    "summary.attempt": "Attempt",
    "summary.error": "Error",
    "summary.feedback": "Feedback",
    "summary.generation_attempts": "Generation attempts",
    "summary.tokens": "tokens",
    "summary.keywords": "keywords",
    "summary.question_only_keywords": "_none; retrieval uses only the question_ (not searched: {keywords})",
    "summary.hint": "hint",
    "summary.candidate_tables": "Candidate tables",
    "summary.selected_tables": "Selected tables",
    "summary.no_tables": "_No tables selected._",
    "summary.architect_reasoning": "Architect reasoning",
    "summary.selector_reasoning": "Why these tables",
    "summary.alternatives": "Other selections that also answer the question",
    "summary.selector_steps": "The candidates and each tool call of the selector are the steps above; the full trace is the last one.",
    "summary.full_trace": "the full selector trace",
    "summary.no_architect_reasoning": "_No architect reasoning captured._",
    "summary.activity_log": "Activity log",
    "summary.no_activity_log": "_No activity log captured._",
    "summary.metadata": "Metadata",
    "summary.code_attempts": "Code generation and execution attempts",
    "summary.no_code_attempts": "_No code generation or execution attempts captured._",
    "summary.synthesized_answer": "Synthesized answer",
    "status.generated": "generated",
    "status.rejected_tables": "rejected tables",
    "status.success": "success",
    "status.error": "error",
}


def t(key: str, default: str | None = None, **kwargs: Any) -> str:
    """The message for ``key``, formatted with ``kwargs``.

    An unknown key falls back to ``default``, then to the key itself. A
    template that does not format with the arguments given is logged and
    returned as it is.

    Args:
        key: The message key, such as ``"phase1.step"``.
        default: The template to use when the key is unknown.
        **kwargs: The values the template's placeholders take.

    Returns:
        The formatted message, or the unformatted template when formatting fails.
    """
    template = _MESSAGES.get(key) or default or key
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError) as exc:
        logger.warning("[i18n] Error formatting key '%s': %s", key, exc)
        return template
