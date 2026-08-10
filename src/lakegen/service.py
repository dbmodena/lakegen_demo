"""Non-interactive LakeGen workflow used by API and batch clients."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from lakegen.core.config import BASE_DIR, LOG_DIR, resolve_portal_tables_dir
from lakegen.core.logger import save_experiment_log
from lakegen.core.resources import (
    capture_retrieval_runs,
    get_all_table_files,
    get_llm,
    get_prompt_manager,
    get_solr,
    log_retrieval_decision,
)
from lakegen.phases import (
    phase1_generate_keywords,
    phase2_select_tables,
    phase12_agent,
    phase3_generate_and_execute,
    phase4_synthesize,
)
from lakegen.ui.state import MODEL_OPTIONS, SOLR_CORE_OPTIONS, RuntimeSettings
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.output_validation import AnswerDisposition, validate_answer
from lakegen.agent_tools.tools_p12 import P12State
from lakegen.experiment_config import (
    DiscoveryArchitecture,
    ExperimentConfig,
    InteractionMode,
    load_experiment_config,
)
from lakegen.manifest import ExperimentManifest, create_manifest, persist_manifest
from lakegen.reproducibility import initialize_reproducibility
from lakegen.tracing import (
    build_llm_phase_records,
    summarize_final_ranking,
    summarize_tool_calls,
)


MAX_CODE_ATTEMPTS = 3
MAX_TABLE_ATTEMPTS = 3
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuestionSource:
    """A question together with its location in the submitted JSON document."""

    question: str
    path: str
    source_id: str | int | None = None
    source_data: dict[str, Any] = field(default_factory=dict)

    def log_fields(self) -> dict[str, Any]:
        """Expose every top-level input variable without CSV name collisions."""

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
    tokens: dict[str, int] = field(
        default_factory=lambda: {"p1_p2": 0, "p3": 0, "p4": 0}
    )
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_path(parent: str, component: str | int) -> str:
    if isinstance(component, int):
        return f"{parent}[{component}]"
    if component.isidentifier():
        return f"{parent}.{component}"
    return f"{parent}[{component!r}]"


def extract_questions(payload: Any) -> list[QuestionSource]:
    """Extract questions from both simple lists and queries_old-style documents.

    Supported examples include ``["question"]``, ``{"questions": [...]}``, and
    the historical nested ``...data.queries[*].question`` structure. Occurrences
    are preserved deliberately: benchmark files may contain intentional repeats.
    """

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
            if key == "question":
                continue
            visit(
                item,
                _json_path(path, str(key)),
                string_is_question=key in {"questions", "queries"},
            )

    visit(payload, "$", string_is_question=isinstance(payload, list))
    return found


def make_runtime_settings(
    *,
    core: str,
    model: str,
    use_unified_agent: bool = True,
    retrieval_mode: RetrievalMode | str = RetrievalMode.KEYWORD,
    top_k: int = 10,
    alpha: float = 0.5,
    candidate_multiplier: int = 5,
    interaction_mode: InteractionMode | str = InteractionMode.AUTONOMOUS,
    experiment_id: str = "default",
    seed: int = 0,
    config: ExperimentConfig | None = None,
) -> RuntimeSettings:
    if core not in SOLR_CORE_OPTIONS:
        raise ValueError(
            f"Unknown core {core!r}. Expected one of: {', '.join(SOLR_CORE_OPTIONS)}"
        )
    if model not in MODEL_OPTIONS:
        raise ValueError(
            f"Unknown model {model!r}. Expected one of: {', '.join(MODEL_OPTIONS)}"
        )

    retrieval = (
        config.retrieval.to_runtime()
        if config is not None
        else RetrievalConfig.from_env(
            mode=retrieval_mode,
            top_k=top_k,
            alpha=alpha,
            candidate_multiplier=candidate_multiplier,
        )
    )
    resolved_config = config or load_experiment_config(overrides={
        "model": model,
        "core": core,
        "discovery_architecture": (
            DiscoveryArchitecture.UNIFIED.value
            if use_unified_agent
            else DiscoveryArchitecture.DIVIDED.value
        ),
        "interaction_mode": str(interaction_mode),
        "experiment_id": experiment_id,
        "seed": seed,
        **{f"retrieval.{key}": value for key, value in retrieval.__dict__.items()},
    })
    return RuntimeSettings(
        model_name=model,
        solr_core=core,
        csv_dir=resolve_portal_tables_dir(core),
        db_path=BASE_DIR / f"data/blend_{core}.db",
        use_unified_agent=use_unified_agent,
        retrieval=retrieval,
        experiment=resolved_config,
    )


def run_question(
    question: str,
    runtime: RuntimeSettings,
    *,
    question_id: str | int | None = None,
    log_context: Mapping[str, Any] | None = None,
    manifest: ExperimentManifest | None = None,
) -> QueryResult:
    """Run the automatic unified LakeGen workflow for one question."""

    started = time.monotonic()
    result = QueryResult(question=question, status="running")
    experiment = getattr(runtime, "experiment", None)
    if experiment is None:
        # Compatibility for lightweight test/third-party RuntimeSettings-like
        # objects; public constructors still validate models and cores strictly.
        experiment = ExperimentConfig().model_copy(update={
            "model": runtime.model_name,
            "core": runtime.solr_core,
            "interaction_mode": InteractionMode.AUTONOMOUS,
        })
    source_id = log_context.get("SOURCE_ID") if log_context else None
    reproducibility = initialize_reproducibility(experiment.seed)
    manifest = manifest or create_manifest(
        experiment,
        base_dir=BASE_DIR,
        question=question,
        question_id=question_id if question_id is not None else source_id,
    )
    result.manifest = manifest.model_dump(mode="json")
    result.configuration = dict(manifest.resolved_config)
    persist_manifest(manifest, LOG_DIR / "manifests")
    llm, _token_counter = get_llm(runtime.model_name)
    solr = get_solr(runtime.solr_core)
    prompt_manager = get_prompt_manager()
    all_files = get_all_table_files(runtime.csv_dir)
    if not all_files:
        raise RuntimeError(f"No CSV or Parquet tables found in {runtime.csv_dir}")

    run_dir = BASE_DIR / "coding" / uuid.uuid4().hex
    selected: list[str] = []
    keywords: list[str] = []
    solr_meta: dict[str, dict[str, Any]] = {}
    reasoning = ""
    trace = ""
    hint = ""
    error = ""
    selection_state = P12State()
    attempted_keywords: list[str] = []
    phase_invocation_counts = {"discovery": 0, "code": 0, "result": 0}
    generated_code_instructions_applied = False

    with capture_retrieval_runs(log_context) as retrieval_runs:
        try:
            for table_attempt in range(MAX_TABLE_ATTEMPTS):
                discovery_started = time.monotonic()
                keywords_rejected = False
                if experiment.use_unified_agent:
                    (
                        selected,
                        keywords,
                        solr_meta,
                        reasoning,
                        trace,
                        architecture_tokens,
                    ) = phase12_agent(
                        query=question,
                        llm=llm,
                        pm=prompt_manager,
                        all_files=all_files,
                        solr_client=solr,
                        csv_dir=runtime.csv_dir,
                        hint=hint,
                        portal_name=runtime.portal_name,
                        retrieval_config=runtime.retrieval,
                        state=selection_state,
                    )
                    phase_invocation_counts["discovery"] += 1
                else:
                    keywords, raw_keywords, tokens_p1, reasoning_p1 = (
                        phase1_generate_keywords(
                            query=question,
                            llm=llm,
                            pm=prompt_manager,
                            hint=hint,
                            portal_name=runtime.portal_name,
                            avoid_keywords=attempted_keywords,
                        )
                    )
                    phase_invocation_counts["discovery"] += 1
                    (
                        selected,
                        candidates,
                        solr_meta,
                        reasoning,
                        trace,
                        tokens_p2,
                    ) = phase2_select_tables(
                        query=question,
                        llm=llm,
                        pm=prompt_manager,
                        all_files=all_files,
                        keywords=keywords,
                        solr_client=solr,
                        csv_dir=runtime.csv_dir,
                        hint=hint,
                        portal_name=runtime.portal_name,
                        retrieval_config=runtime.retrieval,
                    )
                    phase_invocation_counts["discovery"] += 1
                    architecture_tokens = tokens_p1 + tokens_p2
                    trace = (
                        f"--- Divided Phase 1 ---\n{reasoning_p1}\n"
                        f"--- Divided Phase 2 ---\n{trace}"
                    )
                    if reasoning.startswith("REJECT_KEYWORDS:"):
                        attempted_keywords.extend(keywords)
                        attempted_keywords = list(dict.fromkeys(attempted_keywords))
                        hint = (
                            "The previous keywords led to bad tables. "
                            f"Architect feedback: {reasoning}. "
                            "Generate completely different keywords."
                        )
                        keywords_rejected = table_attempt < MAX_TABLE_ATTEMPTS - 1
                        if not keywords_rejected:
                            selected = candidates[:3] if candidates else all_files[:3]
                result.tokens["p1_p2"] += architecture_tokens
                discovery_elapsed = round(time.monotonic() - discovery_started, 6)
                discovery_metric = result.phase_metrics.setdefault(
                    "discovery", {"latency_seconds": 0.0, "retries": 0}
                )
                discovery_metric["latency_seconds"] = round(
                    discovery_metric["latency_seconds"] + discovery_elapsed, 6
                )
                discovery_metric["retries"] = table_attempt
                result.tables = selected
                result.keywords = keywords
                result.pipeline_stages["retrieval"] = (
                    "hit" if retrieval_runs and any(run.get("hits") for run in retrieval_runs)
                    else "no_hits"
                )
                result.pipeline_stages["table_selection"] = (
                    "selected" if selected else "empty"
                )
                if keywords_rejected:
                    continue

                previous_code = ""
                for code_attempt in range(MAX_CODE_ATTEMPTS):
                    generated_code_instructions_applied = True
                    code_started = time.monotonic()
                    generated = phase3_generate_and_execute(
                        question,
                        selected,
                        selected,
                        solr_meta,
                        reasoning,
                        llm,
                        prompt_manager,
                        runtime.csv_dir,
                        retries=code_attempt,
                        error_msg=error,
                        previous_code=previous_code,
                        run_dir=run_dir,
                        seed=reproducibility.effective_seed,
                    )
                    phase_invocation_counts["code"] += 1
                    result.tokens["p3"] += generated.tokens
                    code_metric = result.phase_metrics.setdefault(
                        "code", {"latency_seconds": 0.0, "retries": 0}
                    )
                    code_metric["latency_seconds"] = round(
                        code_metric["latency_seconds"]
                        + (time.monotonic() - code_started),
                        6,
                    )
                    code_metric["retries"] = code_attempt
                    result.retries += int(code_attempt > 0)
                    previous_code = generated.clean_code or generated.code_raw
                    result.code = previous_code

                    if generated.rejected_reason:
                        result.errors.append({
                            "phase": "code",
                            "type": "tables_rejected",
                            "message": generated.rejected_reason,
                        })
                        result.pipeline_stages["code_execution"] = "tables_rejected"
                        hint = (
                            "The code generator rejected the previous tables. "
                            f"Select different tables. Feedback: {generated.rejected_reason}"
                        )
                        error = generated.rejected_reason
                        break

                    if generated.error is None and generated.raw_result is not None:
                        result.raw_result = generated.raw_result
                        raw_validation = validate_answer(generated.raw_result)
                        if raw_validation.disposition == AnswerDisposition.EMPTY:
                            result.pipeline_stages["code_execution"] = "empty"
                            error = raw_validation.reason
                            continue
                        if raw_validation.disposition == AnswerDisposition.REJECTED:
                            result.pipeline_stages["code_execution"] = "rejected"
                            result.pipeline_stages["final_answer"] = "rejected"
                            result.answer_disposition = raw_validation.disposition.value
                            result.status = "rejected"
                            result.error = raw_validation.reason
                            return result

                        result.pipeline_stages["code_execution"] = "succeeded"
                        synthesis_started = time.monotonic()
                        answer, synthesis_tokens = phase4_synthesize(
                            question, generated.raw_result, llm, prompt_manager
                        )
                        phase_invocation_counts["result"] += 1
                        result.answer = answer
                        result.tokens["p4"] = synthesis_tokens
                        result.phase_metrics["result"] = {
                            "latency_seconds": round(
                                time.monotonic() - synthesis_started, 6
                            ),
                            "retries": 0,
                        }
                        validation = validate_answer(generated.raw_result, answer)
                        result.answer_disposition = validation.disposition.value
                        result.pipeline_stages["final_answer"] = validation.disposition.value
                        if validation.disposition == AnswerDisposition.VALID:
                            result.status = "completed"
                            result.error = ""
                        elif validation.disposition == AnswerDisposition.REJECTED:
                            result.status = "rejected"
                            result.error = validation.reason
                        else:
                            result.status = "failed"
                            result.error = validation.reason
                        return result

                    error = generated.error or "Code execution returned no output."
                    result.errors.append({
                        "phase": "code",
                        "type": "execution_error",
                        "message": error,
                    })
                    result.pipeline_stages["code_execution"] = "failed"
                else:
                    break

                if table_attempt == MAX_TABLE_ATTEMPTS - 1:
                    break

            result.status = "failed"
            result.error = error or "LakeGen could not produce an executable answer."
            return result
        except Exception as exc:
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            result.errors.append({"phase": "pipeline", "message": result.error})
            return result
        finally:
            for run in retrieval_runs:
                run["selected_tables"] = list(selected)
                run["selection_reason"] = reasoning or error
            try:
                log_retrieval_decision(
                    question=question,
                    selected_tables=selected,
                    reason=reasoning or error,
                    context=log_context,
                    mode=runtime.retrieval.mode.value,
                    keywords=keywords,
                    retrieval_attempt=len(selection_state.search_attempts),
                )
            except Exception:
                logger.exception("Could not persist the retrieval selection decision")
            result.elapsed_seconds = round(time.monotonic() - started, 3)
            result.phase_metrics["total"] = {
                "latency_seconds": result.elapsed_seconds,
                "retries": result.retries,
            }
            result.discovery = {
                "outcome": result.pipeline_stages["table_selection"],
                "keywords": list(keywords),
                "selected_datasets": list(selected),
                "reasoning": reasoning,
            }
            result.ranking = summarize_final_ranking(retrieval_runs, selected)
            result.llm_calls = build_llm_phase_records(
                total_tokens={
                    "discovery": result.tokens["p1_p2"],
                    "code": result.tokens["p3"],
                    "result": result.tokens["p4"],
                },
                phase_invocations=phase_invocation_counts,
            )
            tool_counts: dict[str, int] = {}
            for run in retrieval_runs:
                tool_type = f"retrieval:{run.get('mode', 'unknown')}"
                tool_counts[tool_type] = tool_counts.get(tool_type, 0) + 1
            for item in summarize_tool_calls(trace):
                tool_name = str(item["type"])
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + int(item["count"])
            result.tool_calls = [
                {"phase": "discovery", "type": tool_type, "count": count}
                for tool_type, count in sorted(tool_counts.items())
            ]
            if result.error and not result.errors:
                result.errors.append({"phase": "pipeline", "message": result.error})
            result.execution_outcome = {
                "status": result.pipeline_stages["code_execution"],
                "raw_result": result.raw_result,
                "error": result.error,
            }
            retrieval = runtime.retrieval
            extra_fields: dict[str, Any] = {
                "CORE": runtime.solr_core,
                "PORTAL_NAME": runtime.portal_name,
                "RETRIEVAL_MODE": retrieval.mode.value,
                "TOP_K": retrieval.top_k,
                "HYBRID_ALPHA": retrieval.alpha,
                "CANDIDATE_MULTIPLIER": retrieval.candidate_multiplier,
                "REPRESENTATION_VERSION": retrieval.representation_version,
                "EMBEDDING_MODEL": retrieval.embedding_model,
                "EMBEDDING_BASE_URL": retrieval.embedding_base_url,
                "VECTOR_FIELD": retrieval.vector_field,
                "LEXICAL_QUERY_FIELDS": retrieval.lexical_query_fields,
                "MISSING_SIGNAL_POLICY": retrieval.missing_signal_policy.value,
                "FUSION_METHOD": retrieval.fusion_method.value,
                "RRF_K": retrieval.rrf_k,
                "RETRIEVAL_RUNS_JSON": retrieval_runs,
                "PIPELINE_STAGES_JSON": result.pipeline_stages,
                "ANSWER_DISPOSITION": result.answer_disposition,
                "MANIFEST_JSON": result.manifest,
                "RUN_TRACE_JSON": {
                    "architecture": experiment.architecture_name,
                    "configuration": result.configuration,
                    "reproducibility": reproducibility.telemetry(
                        generated_code_instructions_applied=(
                            generated_code_instructions_applied
                        )
                    ),
                    "discovery": result.discovery,
                    "ranking": result.ranking,
                    "llm_calls": result.llm_calls,
                    "phase_metrics": result.phase_metrics,
                    "tool_calls": result.tool_calls,
                    "errors": result.errors,
                    "human_interventions": result.human_interventions,
                    "execution_outcome": result.execution_outcome,
                },
            }
            if log_context:
                extra_fields.update(log_context)
            try:
                save_experiment_log(
                    question=question,
                    code=result.code,
                    result=result.raw_result,
                    retries=result.retries,
                    reasoning=reasoning,
                    tables=selected,
                    final_keywords=keywords,
                    final_result=result.answer,
                    full_trace=trace,
                    tokens_phase1=result.tokens["p1_p2"],
                    tokens_phase3=result.tokens["p3"],
                    tokens_phase4=result.tokens["p4"],
                    error=result.error,
                    csv_filename="api_experiments_log.csv",
                    model=runtime.model_name,
                    architecture=experiment.architecture_name,
                    status=result.status,
                    elapsed_seconds=result.elapsed_seconds,
                    extra_fields=extra_fields,
                )
            except Exception:
                logger.exception("Could not persist the LakeGen experiment log")
