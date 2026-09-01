"""Input and output models for the non-interactive LakeGen service."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuestionSource:
    question: str
    path: str
    source_id: str | int | None = None
    source_data: dict[str, Any] = field(default_factory=dict)

    def log_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"SOURCE_JSON": self.source_data}
        for key, value in self.source_data.items():
            column = re.sub(r"[^A-Z0-9]+", "_", str(key).upper()).strip("_")
            if column:
                fields[f"SOURCE_{column}"] = value
        fields["SOURCE_PATH"] = self.path
        fields["SOURCE_ID"] = self.source_id
        return fields


@dataclass
class QueryResult:
    question: str
    status: str
    answer: str = ""
    raw_result: str = ""
    code: str = ""
    tables: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=lambda: {"p1_p2": 0, "p3": 0, "p4": 0})
    retries: int = 0
    error: str = ""
    elapsed_seconds: float = 0.0
    answer_disposition: str = ""
    pipeline_stages: dict[str, str] = field(default_factory=lambda: {
        "retrieval": "not_run", "table_selection": "not_run",
        "code_execution": "not_run", "final_answer": "not_run",
    })
    manifest: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    discovery: dict[str, Any] = field(default_factory=dict)
    ranking: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    phase_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    human_interventions: list[dict[str, Any]] = field(default_factory=list)
    execution_outcome: dict[str, Any] = field(default_factory=dict)
    code_evaluation: dict[str, Any] = field(default_factory=dict)
    coder_context_experiment: dict[str, Any] = field(default_factory=dict)
    semantic_plan_present: bool = False
    semantic_plan_initial_status: str = "missing"
    semantic_plan_final_status: str = "missing"
    semantic_plan_coder_start_status: str = "not_started"
    semantic_plan_status: str = "missing"
    semantic_plan_locked: bool = False
    semantic_plan_revised: bool = False
    semantic_plan_rejected: bool = False
    semantic_plan_validation_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    semantic_plan_evidence_count: int = 0
    validation_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    coder_started_after_verified_plan: bool = False
    coder_blocked_before_start_count: int = 0
    coder_brief_present: bool = False
    coder_brief_status: str = "missing"
    coder_brief_coder_start_status: str = "not_started"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_path(parent: str, component: str | int) -> str:
    if isinstance(component, int):
        return f"{parent}[{component}]"
    if component.isidentifier():
        return f"{parent}.{component}"
    return f"{parent}[{component!r}]"


def extract_questions(payload: Any) -> list[QuestionSource]:
    """Extract questions while preserving their source path and duplicates."""
    found: list[QuestionSource] = []

    def visit(value: Any, path: str, *, string_is_question: bool = False) -> None:
        if isinstance(value, str):
            question = value.strip()
            if string_is_question and question:
                found.append(QuestionSource(question=question, path=path))
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, _json_path(path, index), string_is_question=string_is_question)
            return
        if not isinstance(value, dict):
            return
        question = value.get("question")
        if isinstance(question, str) and question.strip():
            found.append(QuestionSource(
                question=question.strip(), path=_json_path(path, "question"),
                source_id=value.get("id"), source_data=dict(value),
            ))
        for key, item in value.items():
            if key != "question":
                visit(item, _json_path(path, str(key)), string_is_question=key in {"questions", "queries"})

    visit(payload, "$", string_is_question=isinstance(payload, list))
    return found
