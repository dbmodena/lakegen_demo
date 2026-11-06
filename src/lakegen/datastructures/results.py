"""Input and output models for the non-interactive LakeGen service."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuestionSource:
    """A question found in an input file, with where it was found.

    Attributes:
        question: The question text, stripped.
        path: Its JSON path in the input (``$[0].question``).
        source_id: The ``id`` of the object that holds it, when it has one.
        source_data: That whole object, when the question came from one.
    """

    question: str
    path: str
    source_id: str | int | None = None
    source_data: dict[str, Any] = field(default_factory=dict)

    def log_fields(self) -> dict[str, Any]:
        """The ``SOURCE_*`` log fields: one per source key, upper-cased, plus the path and id.

        Returns:
            The fields, with the whole source under ``SOURCE_JSON``.
        """
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
    """Everything one run of a question produced, as the service returns and logs it.

    Attributes:
        question: The question asked.
        status: How the run ended (``completed``, ``rejected``, ``failed``; ``running`` while
            it runs).
        answer: The final answer.
        raw_result: The executed code's raw result.
        code: The code that produced the result.
        tables: The selected tables.
        keywords: The keywords of the last discovery round.
        tokens: Tokens spent, by phase (``p1_p2`` discovery, ``p3`` code, ``p4`` answer).
        retries: How many coder attempts after the first the run made.
        error: The final error, when the run failed.
        elapsed_seconds: The run's wall-clock time.
        answer_disposition: The answer check's verdict on the result.
        pipeline_stages: The status of each stage (retrieval, table selection, code
            execution, final answer).
        manifest: The experiment manifest, for reproducibility; ``to_dict`` names it
            ``experiment_manifest``.
        configuration: The run's configuration.
        discovery: What the discovery phase recorded.
        ranking: The retrieval ranking, one entry per table.
        llm_calls: Every language-model call of the run.
        phase_metrics: Metrics by phase.
        tool_calls: Every agent tool call of the run.
        errors: The errors met along the way.
        human_interventions: What the user did during the run (in the UI).
        execution_outcome: The structured outcome of the code execution.
        code_evaluation: The evaluation of the code against the benchmark's reference.
        coder_context_experiment: The coder-context sweep's variants, when it ran.
        semantic_plan_present: Whether the coder's context audit holds a semantic plan.
        semantic_plan_initial_status: The plan's status at the first audit.
        semantic_plan_final_status: The plan's status at the last audit.
        semantic_plan_coder_start_status: The plan's status when the coder started.
        semantic_plan_status: The plan's status (the final one).
        semantic_plan_locked: Whether the plan was locked.
        semantic_plan_revised: Whether the plan was revised.
        semantic_plan_rejected: Whether the plan was rejected.
        semantic_plan_validation_diagnostics: The plan validation's diagnostics.
        semantic_plan_evidence_count: How many pieces of evidence the plan has.
        validation_diagnostics: The same diagnostics, under the older name.
        evidence_count: The same count, under the older name.
        coder_started_after_verified_plan: Whether the coder started after the plan was
            verified.
        coder_blocked_before_start_count: How many audits ended without the coder starting.
        coder_brief_present: Whether the audit holds a coder brief.
        coder_brief_status: The coder brief's status.
        coder_brief_coder_start_status: The brief's status when the coder started.
        selection_brief_status: The brief's status as the selection left it.
        effective_coder_brief_status: The status of the brief the coder worked from.
        effective_coder_brief_source: Where the brief the coder worked from came from.
    """

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
    pipeline_stages: dict[str, str] = field(
        default_factory=lambda: {
            "retrieval": "not_run",
            "table_selection": "not_run",
            "code_execution": "not_run",
            "final_answer": "not_run",
        }
    )
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
    selection_brief_status: str = "missing"
    effective_coder_brief_status: str = "missing"
    effective_coder_brief_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """The result as a JSON-ready dict, with ``manifest`` named ``experiment_manifest``."""
        payload = asdict(self)
        # This is the experiment/reproducibility manifest, not a coder analysis
        # manifest. Use an unambiguous public name while retaining the internal
        # attribute consumed by structured logging.
        payload["experiment_manifest"] = payload.pop("manifest")
        return payload


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
            found.append(
                QuestionSource(
                    question=question.strip(),
                    path=_json_path(path, "question"),
                    source_id=value.get("id"),
                    source_data=dict(value),
                )
            )
        for key, item in value.items():
            if key != "question":
                visit(
                    item,
                    _json_path(path, str(key)),
                    string_is_question=key in {"questions", "queries"},
                )

    visit(payload, "$", string_is_question=isinstance(payload, list))
    return found
