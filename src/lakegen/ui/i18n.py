from __future__ import annotations

from typing import Any


_MESSAGES: dict[str, str] = {
    "app.title": "# LakeGen - Data Assistant",
    "app.intro": "Ask a natural-language question about the selected Open Data portal.",
    "app.settings_updated": "Settings updated: `{model_name}` on `{solr_core}`.",
    "settings.ollama_url": "Ollama Server URL",
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
        "The Data Architect rejected the generated keywords.\n\n"
        "Feedback: {feedback}"
    ),
    "phase2.generate_keywords": "Generate new keywords",
    "phase2.review_tables": "Review selected tables:\n\n{tables}",
    "phase2.approve": "Approve and run code",
    "phase2.recalculate": "Recalculate tables",
    "phase2.change_hint": "What should the Data Architect change?",
    "phase3.step": "Phase 3 - Code Generation",
    "phase3.code_stream": "Generated code",
    "phase3.model_reasoning": "Model reasoning",
    "phase4.step": "Phase 4 - Execution",
    "phase4.success": "Script executed successfully.",
    "phase5.step": "Phase 5 - Synthesis",
    "result.final": "Final Result",
    "workflow.cancelled": "Workflow cancelled before execution.",
    "workflow.empty_question": "Ask a data question to start a LakeGen run.",
    "workflow.locked": (
        "Another LakeGen workflow is running. I will start this one as soon as "
        "it finishes."
    ),
    "workflow.tables_rejected": (
        "The Code Generator rejected the selected tables.\n\n"
        "Feedback: {feedback}"
    ),
    "workflow.reevaluate_tables": "Re-evaluate tables",
    "workflow.force_execution": "Force execution",
    "summary.none": "_none_",
    "summary.no_keywords": "_No keywords confirmed._",
    "summary.confirmed_keywords": "Confirmed keywords",
    "summary.approval_hint": "Approval hint",
    "summary.total_tokens": "Total tokens",
    "summary.attempt": "Attempt",
    "summary.feedback": "Feedback",
    "summary.generation_attempts": "Generation attempts",
    "summary.tokens": "tokens",
    "summary.keywords": "keywords",
    "summary.hint": "hint",
    "summary.candidate_tables": "Candidate tables",
    "summary.selected_tables": "Selected tables",
    "summary.no_tables": "_No tables selected._",
    "summary.architect_reasoning": "Architect reasoning",
    "summary.no_architect_reasoning": "_No architect reasoning captured._",
    "summary.activity_log": "Activity log",
    "summary.no_activity_log": "_No activity log captured._",
    "summary.metadata": "Metadata",
    "summary.code_attempts": "Code generation attempts",
    "summary.no_code_attempts": "_No code generation attempts captured._",
    "summary.execution_attempts": "Execution attempts",
    "summary.no_execution_attempts": "_No execution attempts captured._",
    "summary.synthesized_answer": "Synthesized answer",
    "status.generated": "generated",
    "status.rejected_tables": "rejected tables",
    "status.success": "success",
    "status.error": "error",
}


def t(key: str, default: str | None = None, **kwargs: Any) -> str:
    template = _MESSAGES.get(key) or default or key
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        print(f"\n\n{exc}\n\n")
        return template
