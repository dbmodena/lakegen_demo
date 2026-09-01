"""Non-interactive LakeGen workflow used by API and batch clients."""

from __future__ import annotations

import logging
import re
import time
import uuid
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
    make_retrieval_run_observer,
)
from lakegen.phases import (
    phase1_generate_keywords,
    phase2_select_tables,
    phase12_agent,
    phase3_generate_and_execute,
    phase4_synthesize,
)
from lakegen.ui.state import MODEL_OPTIONS, SOLR_CORE_OPTIONS, RuntimeSettings
from lakegen.retrieval import RetrievalConfig, RetrievalMode, evaluate_ranking
from lakegen.output_validation import AnswerDisposition, validate_answer
from lakegen.code_attempts import CodeAttemptEvaluator
from lakegen.coder_experiment import run_coder_context_sweep, serialize_retry_error
from lakegen.service_models import QuestionSource, QueryResult, extract_questions
from lakegen.semantic_code_judge import judge_semantic_code_result
from lakegen.agent_tools.tools_p12 import P12State
from lakegen.experiment_config import (
    CoderContextLevel,
    DiscoveryArchitecture,
    ExperimentConfig,
    InteractionMode,
    ToolAccess,
    load_experiment_config,
)
from lakegen.orchestrated_context import prepare_discovery_context
from lakegen.phases.orchestrated_discovery import (
    OrchestratedContextPreparationError,
    OrchestratedSelectorError,
    RetrievalRequestProtocolError,
    run_unified_orchestrated_discovery,
    selector_retry_reason,
    select_from_prepared_context,
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


def _rejected_selection_signature(
    tables: list[str], details: Mapping[str, Any] | None, reason: str
) -> dict[str, Any]:
    """Build stable feedback for retry exclusion without benchmark metadata."""
    details = details or {}
    missing = sorted({
        str(item).strip() for item in details.get("missing_requirements", [])
        if str(item).strip()
    })
    columns = sorted(set(re.findall(
        r"(?:column|columns?)\s+['\"]?([A-Za-z0-9 _/-]{2,80})",
        " ".join([reason, *missing]), flags=re.IGNORECASE,
    )))
    return {
        "tables": sorted(table.casefold() for table in tables),
        "missing_requirements": missing,
        "category": str(details.get("category") or "tables_rejected"),
        "incompatible_columns": columns[:20],
    }


def _record_semantic_plan_telemetry(result: QueryResult, audit: Mapping[str, Any] | None) -> None:
    audit = audit or {}
    status = str(
        audit.get("semantic_plan_final_status")
        or audit.get("semantic_plan_status") or audit.get("status") or "missing"
    )
    initial_status = str(audit.get("semantic_plan_initial_status") or status)
    coder_start_status = str(
        audit.get("semantic_plan_coder_start_status")
        or (initial_status if audit.get("coder_started_after_verified_plan") else "not_started")
    )
    if result.semantic_plan_initial_status == "missing" and not result.semantic_plan_present:
        result.semantic_plan_initial_status = initial_status
    result.semantic_plan_final_status = status
    result.semantic_plan_present = bool(audit.get("semantic_plan_present"))
    result.semantic_plan_status = status
    result.semantic_plan_locked = bool(audit.get("semantic_plan_locked"))
    result.semantic_plan_revised = bool(audit.get("semantic_plan_revised"))
    result.semantic_plan_rejected = bool(audit.get("semantic_plan_rejected"))
    result.semantic_plan_validation_diagnostics = list(
        audit.get("validation_diagnostics") or audit.get("diagnostics") or []
    )
    result.validation_diagnostics = list(result.semantic_plan_validation_diagnostics)
    result.semantic_plan_evidence_count = int(
        audit.get("evidence_count", len(audit.get("evidence") or []))
    )
    result.evidence_count = result.semantic_plan_evidence_count
    brief = audit.get("coder_brief")
    result.coder_brief_present = isinstance(brief, Mapping) and bool(brief)
    if result.coder_brief_present or audit.get("contract_type") == "coder_brief":
        result.coder_brief_status = status
    result.coder_started_after_verified_plan = bool(
        audit.get("coder_started_after_verified_plan")
    )
    if result.coder_started_after_verified_plan:
        result.semantic_plan_coder_start_status = coder_start_status
        if result.coder_brief_present or audit.get("contract_type") == "coder_brief":
            result.coder_brief_coder_start_status = coder_start_status
    if audit and not result.coder_started_after_verified_plan:
        result.coder_blocked_before_start_count += 1


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
    expected_result_type = (
        str(log_context.get("SOURCE_EXPECTED_RESULT_TYPE"))
        if log_context and log_context.get("SOURCE_EXPECTED_RESULT_TYPE")
        else ""
    )
    reference_result = (
        log_context.get("SOURCE_REFERENCE_RESULT") if log_context else None
    )
    expected_result_description = (
        str(log_context.get("SOURCE_EXPECTED_RESULT_DESCRIPTION") or "")
        if log_context
        else ""
    )
    evaluation_contract = (
        log_context.get("SOURCE_EVALUATION_CONTRACT", {}) if log_context else {}
    )
    if not isinstance(evaluation_contract, Mapping):
        evaluation_contract = {}
    attempt_evaluator = CodeAttemptEvaluator(
        expected_result_type=expected_result_type,
        reference_result=reference_result,
        expected_description=expected_result_description,
        evaluation_contract=dict(evaluation_contract),
    )
    code_evaluation_enabled = attempt_evaluator.enabled
    result.code_evaluation = attempt_evaluator.initial_evaluation()
    discovery_phase_tokens = {"p1": 0, "p2": 0}
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
    semantic_judge_llm = None
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
    selection_states: list[P12State] = []
    selection_attempts: list[dict[str, Any]] = []
    rejected_selection_keys: set[tuple[str, ...]] = set()
    rejected_selection_signatures: list[dict[str, Any]] = []
    attempted_keywords: list[str] = []
    phase_invocation_counts = {"discovery": 0, "code": 0, "result": 0}
    generated_code_seed_instruction_provided = False
    context_telemetry: dict[str, Any] = {
        "configured_tool_access": experiment.tool_access.value,
        "execution_path": experiment.tool_access.value,
        "discovery_architecture": experiment.discovery_architecture.value,
        "agent_count": 1 if experiment.use_unified_agent else 2,
        "llm_invocations": 0,
        "retrieval_mode": runtime.retrieval.mode.value,
        "prepared_candidate_count": 0,
        "prepared_context_utf8_bytes": 0,
        "agent_direct_tools": [],
        "orchestrator_retrieval_calls": {},
        "empty_context_retries": 0,
        "request_protocol_error": None,
        "preparation_error": None,
        "selector_error": None,
    }

    def adjudicate_code_evaluation(
        evaluation: dict[str, Any], generated
    ) -> dict[str, Any]:
        nonlocal semantic_judge_llm
        if evaluation.get("evaluation_disposition") != "pending_semantic_review":
            return evaluation
        if not experiment.semantic_code_judge_enabled:
            evaluation.update({
                "evaluation_disposition": "indeterminate",
                "supported_correct": False,
                "semantic_judge_used": False,
                "semantic_judge_model": experiment.semantic_code_judge_model,
            })
            return evaluation
        if semantic_judge_llm is None:
            semantic_judge_llm = (
                llm
                if experiment.semantic_code_judge_model == runtime.model_name
                else get_llm(experiment.semantic_code_judge_model)[0]
            )
        judgment, judge_tokens = judge_semantic_code_result(
            question=question,
            expected_description=expected_result_description,
            reference_result=reference_result,
            selected_tables=selected,
            selected_metadata=solr_meta,
            generated_code=generated.clean_code or generated.code_raw,
            generated_result=getattr(generated, "structured_result", None),
            deterministic_evaluation=evaluation,
            llm=semantic_judge_llm,
            prompt_manager=prompt_manager,
        )
        disposition = judgment["disposition"]
        canonical_disposition = {
            "alternative_correct": "correct",
            "incorrect": "incorrect",
            "indeterminate": "completed_with_warnings",
        }.get(disposition, "completed_with_warnings")
        evaluation.update({
            "evaluation_disposition": canonical_disposition,
            "supported_correct": disposition == "alternative_correct",
            "semantic_judge_used": True,
            "semantic_judge_model": experiment.semantic_code_judge_model,
            "semantic_judge_tokens": judge_tokens,
            "semantic_judgment": judgment,
        })
        return evaluation

    def record_seed_instruction() -> None:
        nonlocal generated_code_seed_instruction_provided
        generated_code_seed_instruction_provided = True

    retrieval_log_context = {
        **dict(log_context or {}),
        "EXPERIMENT_ID": experiment.experiment_id,
    }
    with capture_retrieval_runs(retrieval_log_context) as retrieval_runs:
        agentic_retrieval_observer = make_retrieval_run_observer(
            retrieval_log_context,
            retrieval_runs,
        )
        try:
            for table_attempt in range(MAX_TABLE_ATTEMPTS):
                discovery_started = time.monotonic()
                keywords_rejected = False
                selection_record: dict[str, Any] = {
                    "attempt": table_attempt + 1,
                    "selected_datasets": [],
                    "keywords": [],
                    "outcome": "discovery_started",
                    "rejection_feedback": "",
                }
                selection_attempts.append(selection_record)
                if (
                    experiment.discovery_architecture == DiscoveryArchitecture.UNIFIED
                    and experiment.tool_access == ToolAccess.AGENTIC
                ):
                    # A coder rejection starts a genuinely new retrieval turn.
                    # Reusing P12State would block new searches after the first
                    # inspected candidate and could silently repeat the same set.
                    selection_state = P12State()
                    selection_state.rejected_selections = set(rejected_selection_keys)
                    selection_states.append(selection_state)
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
                        retrieval_observer=agentic_retrieval_observer,
                        planner_enabled=experiment.planner_enabled,
                        require_semantic_plan=experiment.require_semantic_plan,
                    )
                    phase_invocation_counts["discovery"] += 1
                    context_telemetry["llm_invocations"] += 1
                elif (
                    experiment.discovery_architecture == DiscoveryArchitecture.DIVIDED
                    and experiment.tool_access == ToolAccess.AGENTIC
                ):
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
                        selected, candidates, solr_meta, reasoning, trace, tokens_p2,
                    ) = phase2_select_tables(
                        query=question, llm=llm, pm=prompt_manager,
                        all_files=all_files, keywords=keywords, solr_client=solr,
                        csv_dir=runtime.csv_dir, hint=hint,
                        portal_name=runtime.portal_name,
                        retrieval_config=runtime.retrieval,
                    )
                    phase_invocation_counts["discovery"] += 1
                    context_telemetry["llm_invocations"] += 2
                    architecture_tokens = tokens_p1 + tokens_p2
                    trace = f"--- Divided Phase 1 ---\n{reasoning_p1}\n--- Divided Phase 2 ---\n{trace}"
                    if reasoning.startswith("REJECT_KEYWORDS:"):
                        attempted_keywords.extend(keywords)
                        attempted_keywords = list(dict.fromkeys(attempted_keywords))
                        hint = f"The previous keywords led to bad tables. Architect feedback: {reasoning}. Generate completely different keywords."
                        keywords_rejected = table_attempt < MAX_TABLE_ATTEMPTS - 1
                        if not keywords_rejected:
                            selected = candidates[:3] if candidates else all_files[:3]
                elif experiment.discovery_architecture == DiscoveryArchitecture.UNIFIED:
                    try:
                        discovery_result = run_unified_orchestrated_discovery(
                            query=question, llm=llm, solr_client=solr,
                            all_files=all_files, retrieval_config=runtime.retrieval,
                            table_dir=runtime.csv_dir,
                            hint=hint,
                        )
                    except RetrievalRequestProtocolError as exc:
                        context_telemetry["request_protocol_error"] = f"{type(exc).__name__}: {exc}"
                        context_telemetry["llm_invocations"] += 1
                        phase_invocation_counts["discovery"] += 1
                        raise
                    except OrchestratedContextPreparationError as exc:
                        context_telemetry["preparation_error"] = f"{type(exc).__name__}: {exc}"
                        context_telemetry["llm_invocations"] += 1
                        phase_invocation_counts["discovery"] += 1
                        calls = context_telemetry["orchestrator_retrieval_calls"]
                        mode = runtime.retrieval.mode.value
                        calls[mode] = calls.get(mode, 0) + 1
                        raise
                    except OrchestratedSelectorError as exc:
                        context_telemetry["selector_error"] = f"{type(exc).__name__}: {exc}"
                        context_telemetry["llm_invocations"] += 2
                        phase_invocation_counts["discovery"] += 2
                        calls = context_telemetry["orchestrator_retrieval_calls"]
                        mode = runtime.retrieval.mode.value
                        calls[mode] = calls.get(mode, 0) + 1
                        raise
                    selected = discovery_result.selected_datasets
                    keywords = discovery_result.keywords
                    solr_meta = discovery_result.metadata
                    reasoning = discovery_result.reasoning
                    trace = discovery_result.trace
                    architecture_tokens = discovery_result.tokens
                    phase_invocation_counts["discovery"] += discovery_result.llm_invocations
                    context_telemetry["llm_invocations"] += discovery_result.llm_invocations
                    prepared = discovery_result.prepared_context
                    keywords_rejected = discovery_result.retry_keywords
                else:
                    keywords, _raw_keywords, tokens_p1, reasoning_p1 = (
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
                    try:
                        prepared, solr_meta = prepare_discovery_context(
                            query=question, keywords=keywords, solr_client=solr,
                            all_files=all_files, retrieval_config=runtime.retrieval,
                            table_dir=runtime.csv_dir,
                        )
                    except Exception as preparation_error:
                        exc = OrchestratedContextPreparationError(str(preparation_error))
                        context_telemetry["preparation_error"] = f"{type(exc).__name__}: {exc}"
                        context_telemetry["llm_invocations"] += 1
                        calls = context_telemetry["orchestrator_retrieval_calls"]
                        mode = runtime.retrieval.mode.value
                        calls[mode] = calls.get(mode, 0) + 1
                        raise exc from preparation_error
                    candidates = [item.dataset for item in prepared.candidates]
                    if not candidates:
                        selected = []
                        reasoning = "REJECT_KEYWORDS: No datasets found in the prepared context"
                        trace = "--- Divided Orchestrated Retrieval ---\nNo datasets found."
                        tokens_p2 = 0
                        keywords_rejected = True
                    else:
                        try:
                            selected, reasoning, trace, tokens_p2 = select_from_prepared_context(
                                query=question, llm=llm, context=prepared,
                                all_files=all_files, architecture=experiment.discovery_architecture,
                                hint=hint,
                            )
                        except OrchestratedSelectorError as exc:
                            context_telemetry["selector_error"] = f"{type(exc).__name__}: {exc}"
                            context_telemetry["llm_invocations"] += 2
                            phase_invocation_counts["discovery"] += 1
                            raise
                        phase_invocation_counts["discovery"] += 1
                        retry_reason = selector_retry_reason(selected, reasoning)
                        if retry_reason is not None:
                            selected = []
                            reasoning = retry_reason
                            keywords_rejected = True
                    context_telemetry["llm_invocations"] += 1 + int(bool(candidates))
                    architecture_tokens = tokens_p1 + tokens_p2
                if experiment.tool_access == ToolAccess.ORCHESTRATED_CONTEXT:
                    calls = context_telemetry["orchestrator_retrieval_calls"]
                    mode = runtime.retrieval.mode.value
                    calls[mode] = calls.get(mode, 0) + 1
                    if prepared is not None:
                        context_telemetry.update({
                            "prepared_candidate_count": prepared.prepared_candidate_count,
                            "retrieved_hit_count": prepared.retrieved_hit_count,
                            "prepared_context_utf8_bytes": len(prepared.stable_json().encode("utf-8")),
                            "preparation_error": None,
                        })
                    if keywords_rejected:
                        attempted_keywords.extend(keywords)
                        attempted_keywords = list(dict.fromkeys(attempted_keywords))
                        context_telemetry["empty_context_retries"] += 1
                        hint = f"The previous keywords returned no datasets. Generate different keywords. Attempted: {attempted_keywords}"
                result.tokens["p1_p2"] += architecture_tokens
                selection_record.update({
                    "selected_datasets": list(selected),
                    "keywords": list(keywords),
                    "reasoning_available_to_coder": False,
                    "selection_plan": dict(selection_state.selection_plan),
                    "selection_advisories": list(selection_state.selection_advisories),
                    "semantic_plan_failure": selection_state.semantic_failure,
                    "initial_stall_reason": selection_state.initial_stall_reason,
                    "recovery_started": selection_state.recovery_started,
                    "recovery_stop_reason": selection_state.recovery_stop_reason,
                    "selection_plan_source": selection_state.selection_plan_source,
                    "semantic_planner_attempts": selection_state.semantic_planner_attempts,
                    "semantic_draft_present": bool(selection_state.semantic_draft),
                    "outcome": "keywords_rejected" if keywords_rejected else "selected",
                })
                if experiment.discovery_architecture == DiscoveryArchitecture.DIVIDED:
                    discovery_phase_tokens["p1"] += tokens_p1
                    discovery_phase_tokens["p2"] += tokens_p2
                else:
                    # A unified agent owns both logical phases, so its usage is
                    # intentionally kept together in the Phase 1 log column.
                    discovery_phase_tokens["p1"] += architecture_tokens
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

                selection_key = tuple(sorted(table.casefold() for table in selected))
                if selected and selection_key in rejected_selection_keys:
                    selection_record["outcome"] = (
                        "rejected_selection_excluded"
                        if table_attempt < MAX_TABLE_ATTEMPTS - 1
                        else "tables_rejected_no_alternative"
                    )
                    prior = next((
                        item for item in reversed(rejected_selection_signatures)
                        if tuple(item["tables"]) == selection_key
                    ), {})
                    error = "No verified alternative selection remains. " + str(prior)
                    hint = (
                        error
                        + " Select at least one different table. Use these missing "
                        f"requirements as retrieval concepts: {hint}"
                    )
                    if table_attempt < MAX_TABLE_ATTEMPTS - 1:
                        continue
                    result.status = "rejected"
                    result.pipeline_stages["code_execution"] = "tables_rejected"
                    break

                if experiment.automatic_test_coder:
                    # Full context is the only reliable table-rejection gate.
                    # Run it first so rejected selections can return to discovery
                    # without wasting calls on the reduced-context variants.
                    variants = run_coder_context_sweep(
                        question=question, selected=selected,
                        solr_meta=solr_meta, reasoning=reasoning, llm=llm,
                        prompt_manager=prompt_manager, csv_dir=runtime.csv_dir,
                        run_dir=run_dir, seed=reproducibility.effective_seed,
                        record_seed_instruction=record_seed_instruction,
                        expected_result_type=expected_result_type,
                        evaluation_enabled=code_evaluation_enabled,
                        evaluator=attempt_evaluator,
                        adjudicate=adjudicate_code_evaluation,
                        generate_and_execute=phase3_generate_and_execute,
                        result=result,
                        phase_invocation_counts=phase_invocation_counts,
                        max_attempts=MAX_CODE_ATTEMPTS,
                        context_levels=[CoderContextLevel.FULL],
                        selection_plan=dict(selection_state.selection_plan),
                        source_field_names=list((log_context or {}).keys()),
                        require_semantic_plan=experiment.require_semantic_plan,
                    )
                    primary = variants[CoderContextLevel.FULL.value]
                    _record_semantic_plan_telemetry(
                        result, primary.get("coder_context_audit")
                    )
                    coder_audit = primary.get("coder_context_audit") or {}
                    selection_record["coder_brief_status"] = str(
                        coder_audit.get("status") or "missing"
                    )
                    selection_record["coder_started"] = bool(
                        coder_audit.get("coder_started_after_verified_plan")
                    )
                    selection_record["coder_outcome"] = primary.get("status")
                    if (
                        experiment.require_semantic_plan
                        and (primary.get("execution_error") or {}).get("stage") == "selection_contract_gate"
                    ):
                        selection_record["outcome"] = "selection_contract_blocked"
                        result.pipeline_stages["code_execution"] = "blocked_by_selection"
                        result.code_evaluation.update({
                            "evaluation_disposition": "blocked",
                            "error_category": "selection_contract_gate",
                        })
                        error = primary["error"]
                        hint = error + (
                            " Retry discovery with exact inspected column choices "
                            "and explicit join keys when joining tables."
                        )
                        if table_attempt < MAX_TABLE_ATTEMPTS - 1:
                            continue
                        break
                    if primary["status"] == "tables_rejected":
                        rejection_reason = primary["error"]
                        rejected_selection_keys.add(selection_key)
                        rejected_selection_signatures.append(
                            _rejected_selection_signature(
                                selected, primary.get("rejection_details"), rejection_reason
                            )
                        )
                        selection_record["outcome"] = "tables_rejected"
                        selection_record["rejection_feedback"] = rejection_reason
                        result.errors.append({
                            "phase": "code",
                            "type": "tables_rejected",
                            "message": rejection_reason,
                            "coder_context_level": CoderContextLevel.FULL.value,
                        })
                        result.pipeline_stages["code_execution"] = "tables_rejected"
                        previous_visible = [
                            table for table in selection_state.all_candidates[
                                : selection_state.visible_candidate_count
                            ]
                            if table not in selected
                        ][:5]
                        hint = (
                            "The full-context code generator rejected the previous "
                            "table combination. First reassess these previously visible "
                            "alternatives if retrieval returns them again: "
                            f"{previous_visible}. Select a different combination. Start "
                            "a genuinely new search only if none covers the missing "
                            f"requirement. Feedback: {rejection_reason}"
                        )
                        error = rejection_reason
                        if table_attempt < MAX_TABLE_ATTEMPTS - 1:
                            continue
                        break

                    variants.update(run_coder_context_sweep(
                        question=question, selected=selected,
                        solr_meta=solr_meta, reasoning=reasoning, llm=llm,
                        prompt_manager=prompt_manager, csv_dir=runtime.csv_dir,
                        run_dir=run_dir, seed=reproducibility.effective_seed,
                        record_seed_instruction=record_seed_instruction,
                        expected_result_type=expected_result_type,
                        evaluation_enabled=code_evaluation_enabled,
                        evaluator=attempt_evaluator,
                        adjudicate=adjudicate_code_evaluation,
                        generate_and_execute=phase3_generate_and_execute,
                        result=result,
                        phase_invocation_counts=phase_invocation_counts,
                        max_attempts=MAX_CODE_ATTEMPTS,
                        context_levels=[
                            CoderContextLevel.SCHEMA_ONLY,
                            CoderContextLevel.MINIMAL,
                        ],
                        selection_plan=dict(selection_state.selection_plan),
                        source_field_names=list((log_context or {}).keys()),
                        require_semantic_plan=experiment.require_semantic_plan,
                    ))
                    result.coder_context_experiment = {
                        "shared_retrieval": True,
                        "shared_tables": list(selected),
                        "shared_keywords": list(keywords),
                        "shared_reasoning": None,
                        "primary_level": CoderContextLevel.FULL.value,
                        "variants": variants,
                    }
                    selection_record["outcome"] = "accepted"
                    result.code = primary["code"]
                    result.raw_result = primary["raw_result"]
                    result.code_evaluation = primary["code_evaluation"]
                    primary_status = primary["status"]
                    if primary_status == "completed":
                        result.pipeline_stages["code_execution"] = "succeeded"
                        synthesis_started = time.monotonic()
                        answer, synthesis_tokens = phase4_synthesize(
                            question, result.raw_result, llm, prompt_manager
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
                        validation = validate_answer(result.raw_result, answer)
                        result.answer_disposition = validation.disposition.value
                        result.pipeline_stages["final_answer"] = (
                            validation.disposition.value
                        )
                        if validation.disposition == AnswerDisposition.VALID:
                            result.status = "completed"
                            result.error = ""
                        elif validation.disposition == AnswerDisposition.REJECTED:
                            result.status = "rejected"
                            result.error = validation.reason
                        else:
                            result.status = "failed"
                            result.error = validation.reason
                    else:
                        result.pipeline_stages["code_execution"] = primary_status
                        result.status = (
                            "rejected" if primary_status == "rejected" else "failed"
                        )
                        result.error = primary["error"]
                    return result

                previous_code = ""
                for code_attempt in range(MAX_CODE_ATTEMPTS):
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
                        seed_instruction_recorder=record_seed_instruction,
                        coder_context_level=experiment.coder_context_level,
                        evaluation_result_type=None,
                        selection_plan=dict(selection_state.selection_plan),
                        source_field_names=list((log_context or {}).keys()),
                        require_semantic_plan=experiment.require_semantic_plan,
                    )
                    phase_invocation_counts["code"] += 1
                    _record_semantic_plan_telemetry(
                        result, getattr(generated, "coder_context_audit", None)
                    )
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

                    if (
                        experiment.require_semantic_plan
                        and (getattr(generated, "execution_error", None) or {}).get("stage") == "selection_contract_gate"
                    ):
                        selection_record["outcome"] = "selection_contract_blocked"
                        result.pipeline_stages["code_execution"] = "blocked_by_selection"
                        result.code_evaluation.update({
                            "evaluation_disposition": "blocked",
                            "error_category": "selection_contract_gate",
                        })
                        error = generated.error or "The selection contract was not verified."
                        hint = error + " Retry discovery with exact inspected columns and join keys."
                        break

                    if code_evaluation_enabled:
                        attempts = result.code_evaluation["attempts"]
                        attempts.append(attempt_evaluator.evaluate(
                            generated, code_attempt + 1
                        ))
                        result.code_evaluation = attempt_evaluator.summarize(attempts)

                    if generated.rejected_reason:
                        rejected_selection_keys.add(selection_key)
                        rejected_selection_signatures.append(
                            _rejected_selection_signature(
                                selected, generated.rejection_details,
                                generated.rejected_reason,
                            )
                        )
                        result.errors.append({
                            "phase": "code",
                            "type": "tables_rejected",
                            "message": generated.rejected_reason,
                        })
                        result.pipeline_stages["code_execution"] = "tables_rejected"
                        previous_visible = [
                            table for table in selection_state.all_candidates[
                                : selection_state.visible_candidate_count
                            ]
                            if table not in selected
                        ][:5]
                        hint = (
                            "The code generator rejected the previous table combination. "
                            "First reassess these previously visible alternatives if "
                            f"retrieval returns them again: {previous_visible}. Select a "
                            "different combination. Start a genuinely new search only if "
                            "none covers the missing requirement. Feedback: "
                            f"{generated.rejected_reason}"
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

                        if code_evaluation_enabled:
                            result.code_evaluation = adjudicate_code_evaluation(
                                result.code_evaluation, generated
                            )
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

                    error = serialize_retry_error(generated)
                    error_record = {
                        "phase": "code",
                        "type": "execution_error",
                        "message": error,
                    }
                    structured_execution_error = getattr(
                        generated, "execution_error", None
                    )
                    if structured_execution_error:
                        error_record["details"] = structured_execution_error
                        error_record["type"] = structured_execution_error.get(
                            "category", "execution_error"
                        )
                    result.errors.append(error_record)
                    result.pipeline_stages["code_execution"] = "failed"
                    if getattr(generated, "coder_runs", 0):
                        break
                else:
                    break

                if table_attempt == MAX_TABLE_ATTEMPTS - 1:
                    break

            result.status = (
                "blocked_by_selection"
                if result.pipeline_stages["code_execution"] == "blocked_by_selection"
                else "failed"
            )
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
                    context=retrieval_log_context,
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
            p12_states = selection_states or [selection_state]
            result.discovery = {
                "outcome": result.pipeline_stages["table_selection"],
                "keywords": list(keywords),
                "selected_datasets": list(selected),
                "reasoning_available_to_coder": False,
                "search_attempt_count": sum(
                    len(state.search_attempts) for state in p12_states
                ),
                "expansion_used": any(state.expansion_count for state in p12_states),
                "expansion_requirements": list(dict.fromkeys(
                    requirement
                    for state in p12_states
                    for requirement in state.expansion_requirements
                )),
                "selection_attempts": selection_attempts,
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
            result.tool_calls = []
            for tool_type, count in sorted(tool_counts.items()):
                is_retrieval = tool_type.startswith("retrieval:")
                if experiment.tool_access == ToolAccess.ORCHESTRATED_CONTEXT:
                    actor = "orchestrator" if is_retrieval else "agent"
                elif experiment.use_unified_agent:
                    actor = "agent"
                else:
                    actor = "orchestrator" if is_retrieval else "agent"
                result.tool_calls.append({
                    "phase": "discovery",
                    "type": tool_type,
                    "count": count,
                    "actor": actor,
                })
            context_telemetry["agent_direct_tools"] = [
                item["type"]
                for item in result.tool_calls
                if item["actor"] == "agent"
            ]
            if result.coder_context_experiment:
                context_telemetry["coder_context_audit"] = {
                    level: variant.get("coder_context_audit")
                    for level, variant in result.coder_context_experiment.get(
                        "variants", {}
                    ).items()
                }
            else:
                context_telemetry["coder_context_audit"] = getattr(
                    locals().get("generated"), "coder_context_audit", None
                )
            if experiment.tool_access != ToolAccess.ORCHESTRATED_CONTEXT:
                orchestrator_calls: dict[str, int] = {}
                for item in result.tool_calls:
                    if item["actor"] != "orchestrator":
                        continue
                    tool_type = str(item["type"])
                    if not tool_type.startswith("retrieval:"):
                        continue
                    mode = tool_type.split(":", 1)[1]
                    orchestrator_calls[mode] = (
                        orchestrator_calls.get(mode, 0) + int(item["count"])
                    )
                context_telemetry["orchestrator_retrieval_calls"] = orchestrator_calls
            if result.error and not result.errors:
                result.errors.append({"phase": "pipeline", "message": result.error})
            result.execution_outcome = {
                "status": result.pipeline_stages["code_execution"],
                "raw_result": result.raw_result,
                "error": result.error,
                "code_evaluation": result.code_evaluation,
                "coder_context_experiment": result.coder_context_experiment,
                "coder_runs": getattr(locals().get("generated"), "coder_runs", 0),
                "coder_review": getattr(
                    locals().get("generated"), "coder_review", None
                ),
            }
            retrieval = runtime.retrieval
            extra_fields: dict[str, Any] = {
                "EXPERIMENT_ID": experiment.experiment_id,
                "MANIFEST_ID": manifest.run_id,
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
                    "tool_access": context_telemetry,
                    "configuration": result.configuration,
                    "reproducibility": reproducibility.telemetry(
                        generated_code_seed_instruction_provided=(
                            generated_code_seed_instruction_provided
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
                    "code_evaluation": result.code_evaluation,
                    "coder_context_experiment": result.coder_context_experiment,
                },
            }
            if log_context:
                extra_fields.update(log_context)
            gold_tables = extra_fields.get("SOURCE_RELEVANT_TABLE_IDS")
            if isinstance(gold_tables, list) and gold_tables:
                ranking = [
                    re.sub(r"\.(?:parquet|pq|csv)$", "", table, flags=re.I)
                    for table in selected
                ]
                metrics = evaluate_ranking(ranking, gold_tables, k_values=(1, 5, 10))
                extra_fields.update({
                    "HIT_AT_1": metrics["Hit@1"],
                    "HIT_AT_5": metrics["Hit@5"],
                    "HIT_AT_10": metrics["Hit@10"],
                    "RECALL_AT_1": metrics["Recall@1"],
                    "RECALL_AT_5": metrics["Recall@5"],
                    "RECALL_AT_10": metrics["Recall@10"],
                    "MRR": metrics["MRR"],
                    "NDCG_AT_1": metrics["nDCG@1"],
                    "NDCG_AT_5": metrics["nDCG@5"],
                    "NDCG_AT_10": metrics["nDCG@10"],
                })
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
                    tokens_phase1=discovery_phase_tokens["p1"],
                    tokens_phase2=discovery_phase_tokens["p2"],
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
