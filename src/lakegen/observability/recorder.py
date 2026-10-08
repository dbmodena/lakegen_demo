"""What a run reports, assembled into its result and its logs.

The ``Recorder`` holds the ``QueryResult`` of a run and the telemetry the
stages report as they go: LLM invocations per phase, discovery tokens per
phase, the tool-access telemetry, the coder's semantic-plan audit, the seed
instruction. When the run ends, ``finish`` assembles the result's record
(discovery, ranking, LLM calls, tool calls, execution outcome), writes the
retrieval decision and the CSV log row, the way the service's ``finally``
block did. Nothing else in the engine writes telemetry.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Mapping
from typing import Any

from lakegen.experiment_config import ToolAccess
from lakegen.retrieval import evaluate_ranking
from lakegen.service_models import QueryResult
from lakegen.tracing import build_llm_phase_records, summarize_final_ranking, summarize_tool_calls

logger = logging.getLogger(__name__)


def record_semantic_plan_telemetry(result: QueryResult, audit: Mapping[str, Any] | None) -> None:
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
    result.selection_brief_status = str(
        audit.get("selection_brief_status") or initial_status
    )
    result.effective_coder_brief_status = str(
        audit.get("effective_coder_brief_status") or status
    )
    result.effective_coder_brief_source = str(
        audit.get("effective_coder_brief_source") or ""
    )
    if result.coder_brief_present or audit.get("contract_type") == "coder_brief":
        result.coder_brief_status = result.effective_coder_brief_status
    result.coder_started_after_verified_plan = bool(
        audit.get("coder_started_after_verified_plan")
    )
    if result.coder_started_after_verified_plan:
        result.semantic_plan_coder_start_status = coder_start_status
        if result.coder_brief_present or audit.get("contract_type") == "coder_brief":
            result.coder_brief_coder_start_status = coder_start_status
    if audit and not result.coder_started_after_verified_plan:
        result.coder_blocked_before_start_count += 1



class Recorder:
    def __init__(self, question: str, ctx: Any, deps: Any) -> None:
        self.question = question
        self.ctx = ctx
        self.experiment = ctx.experiment
        self.deps = deps
        self.started = time.monotonic()
        self.result = QueryResult(question=question, status="running")
        self.phase_invocation_counts = {"discovery": 0, "code": 0, "result": 0}
        self.discovery_phase_tokens = {"p1": 0, "p2": 0}
        self.seed_instruction_provided = False
        experiment = self.experiment
        self.context_telemetry: dict[str, Any] = {
            "configured_tool_access": experiment.tool_access.value,
            "execution_path": experiment.tool_access.value,
            "discovery_architecture": experiment.discovery_architecture.value,
            "agent_count": 1 if experiment.use_unified_agent else 2,
            "llm_invocations": 0,
            "retrieval_mode": ctx.retrieval.mode.value,
            "prepared_candidate_count": 0,
            "prepared_context_utf8_bytes": 0,
            "agent_direct_tools": [],
            "orchestrator_retrieval_calls": {},
            "empty_context_retries": 0,
            "request_protocol_error": None,
            "preparation_error": None,
            "selector_error": None,
        }

    def record_seed_instruction(self) -> None:
        self.seed_instruction_provided = True

    def semantic_plan(self, audit: Mapping[str, Any] | None) -> None:
        record_semantic_plan_telemetry(self.result, audit)

    def finish(  # noqa: C901, PLR0915  (the service's finally block, moved whole)
        self,
        *,
        retrieval_runs: list[dict[str, Any]],
        retrieval_log_context: Mapping[str, Any],
        selected: list[str],
        keywords: list[str],
        reasoning: str,
        trace: str,
        error: str,
        selection_state: Any,
        selection_states: list[Any],
        selection_attempts: list[dict[str, Any]],
        last_generated: Any,
        manifest: Any,
        reproducibility: Any,
        benchmark: Any,
    ) -> None:
        """Close the run: its record, the retrieval decision and the log row."""
        result, ctx, experiment, deps, question = self.result, self.ctx, self.experiment, self.deps, self.question
        for run in retrieval_runs:
            run["selected_tables"] = list(selected)
            run["selection_reason"] = reasoning or error
        try:
            deps.log_retrieval_decision(
                question=question,
                selected_tables=selected,
                reason=reasoning or error,
                context=retrieval_log_context,
                mode=ctx.retrieval.mode.value,
                keywords=keywords,
                retrieval_attempt=len(selection_state.search_attempts),
            )
        except Exception:
            logger.exception("Could not persist the retrieval selection decision")
        result.elapsed_seconds = round(time.monotonic() - self.started, 3)
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
            phase_invocations=self.phase_invocation_counts,
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
        self.context_telemetry["agent_direct_tools"] = [
            item["type"]
            for item in result.tool_calls
            if item["actor"] == "agent"
        ]
        if result.coder_context_experiment:
            self.context_telemetry["coder_context_audit"] = {
                level: variant.get("coder_context_audit")
                for level, variant in result.coder_context_experiment.get(
                    "variants", {}
                ).items()
            }
        else:
            self.context_telemetry["coder_context_audit"] = getattr(last_generated, "coder_context_audit", None)
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
            self.context_telemetry["orchestrator_retrieval_calls"] = orchestrator_calls
        if result.error and not result.errors:
            result.errors.append({"phase": "pipeline", "message": result.error})
        result.execution_outcome = {
            "status": result.pipeline_stages["code_execution"],
            "raw_result": result.raw_result,
            "error": result.error,
            "code_evaluation": result.code_evaluation,
            "coder_context_experiment": result.coder_context_experiment,
            "coder_runs": getattr(last_generated, "coder_runs", 0),
            "coder_review": getattr(last_generated, "coder_review", None),
        }
        retrieval = ctx.retrieval
        extra_fields: dict[str, Any] = {
            "EXPERIMENT_ID": experiment.experiment_id,
            "MANIFEST_ID": manifest.run_id,
            "CORE": ctx.lake,
            "PORTAL_NAME": ctx.portal_name,
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
                "tool_access": self.context_telemetry,
                "configuration": result.configuration,
                "reproducibility": reproducibility.telemetry(
                    generated_code_seed_instruction_provided=(
                        self.seed_instruction_provided
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
        extra_fields.update(benchmark.to_log_fields())
        gold_tables = benchmark.relevant_tables
        if gold_tables:
            ranking_entries = [
                entry for entry in getattr(result, "ranking", [])
                if isinstance(entry, dict)
            ]
            if ranking_entries:
                final_attempt = max(
                    int(entry.get("attempt", 1)) for entry in ranking_entries
                )
                ranking = [
                    str(entry["resource_id"])
                    for entry in sorted(
                        (
                            entry for entry in ranking_entries
                            if int(entry.get("attempt", 1)) == final_attempt
                        ),
                        key=lambda entry: int(entry.get("rank", 0)),
                    )
                    if entry.get("resource_id")
                ]
            else:
                ranking = [
                    re.sub(r"\.(?:parquet|pq|csv)$", "", table, flags=re.I)
                    .rsplit("___", 1)[-1]
                    for table in selected
                ]
            normalized_gold = [str(table).rsplit("___", 1)[-1] for table in gold_tables]
            metrics = evaluate_ranking(
                ranking, normalized_gold, k_values=(1, 5, 10, 15, 20)
            )
            extra_fields.update({
                "HIT_AT_1": metrics["Hit@1"],
                "HIT_AT_5": metrics["Hit@5"],
                "HIT_AT_10": metrics["Hit@10"],
                "HIT_AT_15": metrics["Hit@15"],
                "HIT_AT_20": metrics["Hit@20"],
                "RECALL_AT_1": metrics["Recall@1"],
                "RECALL_AT_5": metrics["Recall@5"],
                "RECALL_AT_10": metrics["Recall@10"],
                "RECALL_AT_15": metrics["Recall@15"],
                "RECALL_AT_20": metrics["Recall@20"],
                "MRR": metrics["MRR"],
                "NDCG_AT_1": metrics["nDCG@1"],
                "NDCG_AT_5": metrics["nDCG@5"],
                "NDCG_AT_10": metrics["nDCG@10"],
                "NDCG_AT_15": metrics["nDCG@15"],
                "NDCG_AT_20": metrics["nDCG@20"],
            })
        searched_keywords, unused_concepts = retrieval.mode.split_keywords(keywords)
        coder_variants = (
            result.coder_context_experiment.get("variants", {})
            if isinstance(result.coder_context_experiment, dict)
            else {}
        )
        coder_variant_tokens = {
            name: int(coder_variants.get(name, {}).get("tokens", 0) or 0)
            for name in ("full", "schema_only", "minimal")
        }
        extra_fields.update({
            "TOKENS_CODE_FULL": coder_variant_tokens["full"],
            "TOKENS_CODE_SCHEMA_ONLY": coder_variant_tokens["schema_only"],
            "TOKENS_CODE_MINIMAL": coder_variant_tokens["minimal"],
            "TOKENS_CODE_UNATTRIBUTED": max(
                0, int(result.tokens.get("p3", 0) or 0) - sum(coder_variant_tokens.values())
            ),
        })
        try:
            deps.save_experiment_log(
                question=question,
                code=result.code,
                result=result.raw_result,
                retries=result.retries,
                reasoning=reasoning,
                tables=selected,
                final_keywords=searched_keywords,
                unused_concepts=unused_concepts,
                final_result=result.answer,
                full_trace=trace,
                tokens_phase1=self.discovery_phase_tokens["p1"],
                tokens_phase2=self.discovery_phase_tokens["p2"],
                tokens_phase3=result.tokens["p3"],
                tokens_phase4=result.tokens["p4"],
                error=result.error,
                csv_filename="api_experiments_log.csv",
                model=ctx.model_name,
                architecture=experiment.architecture_name,
                status=result.status,
                elapsed_seconds=result.elapsed_seconds,
                extra_fields=extra_fields,
            )
        except Exception:
            logger.exception("Could not persist the LakeGen experiment log")
