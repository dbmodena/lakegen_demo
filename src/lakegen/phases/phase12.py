import io
import json
import os
import sys
import asyncio
import re
from pathlib import Path
from collections.abc import Callable

from llama_index.core import Settings
from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from lakegen.phases.logging import (
    Phase2AgentStall,
    detect_phase2_agent_stall,
    format_phase2_tool_args,
    format_phase2_tool_call,
    format_phase2_tool_output,
    format_phase2_tool_result,
)
from lakegen.core.types import SolrMetadata, StreamCallback
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.core.table_io import read_table
from lakegen.ui.state import WorkflowCancelled
from lakegen.agents.instrumentation import ThinkingCapture
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient
from lakegen.agent_tools.tools_p12 import P12State, Phase12ToolsManager
from lakegen.retrieval import RetrievalConfig
from lakegen.retrieval.models import RetrievalRun


def _reasoning_with_selection_plan(
    reasoning: str,
    plan: dict[str, object] | None,
    advisories: list[str] | None,
) -> str:
    """Add the architect's agentic plan to the existing coder context."""
    if not plan:
        return reasoning
    lines = [reasoning, "", "AGENTIC SELECTION PLAN"]
    lines.append(f"Combination strategy: {plan.get('combination_strategy', 'unspecified')}")
    roles = plan.get("table_roles", {})
    if isinstance(roles, dict) and roles:
        lines.append("Table roles:")
        lines.extend(f"- {table}: {role}" for table, role in roles.items())
    coverage = plan.get("requirement_coverage", {})
    if isinstance(coverage, dict) and coverage:
        lines.append("Requirement coverage:")
        for requirement, evidence in coverage.items():
            if isinstance(evidence, dict):
                table = evidence.get("table", "unspecified")
                columns = evidence.get("columns", [])
                columns_text = ", ".join(map(str, columns)) if isinstance(columns, list) else str(columns)
                lines.append(f"- {requirement}: {table} [{columns_text}]")
    uncovered = plan.get("uncovered_requirements", [])
    if isinstance(uncovered, list) and uncovered:
        lines.append("Uncovered requirements: " + "; ".join(map(str, uncovered)))
    semantic_plan = plan.get("semantic_plan")
    if isinstance(semantic_plan, dict):
        lines.append("Semantic analysis plan (authoritative, non-oracle JSON):")
        lines.append(json.dumps(semantic_plan, ensure_ascii=False, sort_keys=True))
    alternatives = plan.get("alternatives_rejected", {})
    if isinstance(alternatives, dict) and alternatives:
        lines.append("Inspected alternatives rejected:")
        for table, evidence in list(alternatives.items())[:2]:
            if isinstance(evidence, dict):
                matched = ", ".join(map(str, evidence.get("matched_requirements", [])))
                missing = evidence.get("missing_requirement", "not specified")
                lines.append(f"- {table}: matches [{matched}]; lacks {missing}")
            else:
                lines.append(f"- {table}: lacks {evidence}")
    if advisories:
        lines.append("Non-blocking architect advisories:")
        lines.extend(f"- {advisory}" for advisory in advisories)
    return "\n".join(lines).strip()


def _recover_minimal_selection_plan(
    selected: list[str], reasoning: str
) -> tuple[dict[str, object], list[str]]:
    """Recover non-blocking coder guidance when discovery exits without a plan."""
    if not selected:
        return {}, []
    lowered = reasoning.casefold()
    if len(selected) == 1:
        strategy = "single_table"
    elif any(term in lowered for term in ("concat", "partition", "append", "union")):
        strategy = "concat_partitions"
    elif any(term in lowered for term in ("lookup", "mapping", "reference table")):
        strategy = "lookup"
    elif any(term in lowered for term in ("compare", "comparison", "versus", " vs ")):
        strategy = "compare"
    elif any(term in lowered for term in ("join", "merge", "shared key")):
        strategy = "join"
    else:
        strategy = "aggregate_separately"
    roles = {
        table: ("primary selected source" if index == 0 else "supporting selected source")
        for index, table in enumerate(selected)
    }
    plan = {
        "requirement_coverage": {},
        "table_roles": roles,
        "combination_strategy": strategy,
        "uncovered_requirements": [],
        "alternatives_rejected": {},
        "recovered_from_existing_discovery_context": True,
    }
    return plan, [
        "The structured selection plan was recovered from the existing discovery "
        "decision; treat it as guidance, not as a blocking constraint."
    ]


def _inspected_runtime_evidence(state: P12State, max_chars: int = 24000) -> str:
    """Serialize benchmark-blind cached inspections for a fresh recovery agent."""
    evidence: dict[str, str] = {}
    remaining = max_chars
    for table in state.inspected_candidates():
        inspection = str(state.inspection_cache.get(table.casefold(), "")).strip()
        if not inspection or remaining <= 0:
            continue
        excerpt = inspection[:remaining]
        evidence[table] = excerpt
        remaining -= len(excerpt)
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True)


def _extract_plausible_json(text: str) -> dict[str, object] | None:
    """Recover one complete JSON object from assistant text without accepting it."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _conservative_draft_from_requirements(
    requirements: dict[str, object], selected: list[str], schema: set[str]
) -> dict[str, object] | None:
    """Build only a single-table draft whose choices are explicit and unambiguous."""
    if len(selected) != 1:
        return None
    canonical = {column.casefold(): column for column in schema}
    grouping: list[list[object]] = []
    for raw in requirements.get("grouping", []) if isinstance(requirements.get("grouping"), list) else []:
        column = canonical.get(str(raw).strip().casefold())
        if not column:
            return None
        grouping.append([column, column])
    measures: list[list[object]] = []
    for index, raw in enumerate(
        requirements.get("measures", []) if isinstance(requirements.get("measures"), list) else []
    ):
        text = str(raw).strip()
        lowered = text.casefold()
        if lowered in {"count rows", "row count", "count_rows"}:
            measures.append(["row_count", "count_rows", []])
            continue
        operation = next((
            normalized for term, normalized in (
                ("average", "mean"), ("mean", "mean"), ("sum", "sum"),
                ("distinct", "count_distinct"), ("minimum", "min"),
                ("maximum", "max"),
            ) if term in lowered
        ), "")
        matched = [column for folded, column in canonical.items() if folded in lowered]
        if not operation or len(matched) != 1:
            return None
        measures.append([f"{operation}_{matched[0]}", operation, matched])
    if not measures:
        return None
    filters: list[list[object]] = []
    for raw in requirements.get("filters", []) if isinstance(requirements.get("filters"), list) else []:
        match = re.fullmatch(r"\s*([^=<>]+?)\s*(=|contains)\s*(.+?)\s*", str(raw), re.I)
        if not match:
            return None
        column = canonical.get(match.group(1).strip().casefold())
        if not column:
            return None
        filters.append([
            column, "equals" if match.group(2) == "=" else "contains",
            match.group(3).strip(),
        ])
    ordering: list[list[object]] = []
    ordering_text = str(requirements.get("ordering") or "").casefold()
    if ordering_text:
        direction = "descending" if "desc" in ordering_text else (
            "ascending" if "asc" in ordering_text else ""
        )
        if not direction:
            return None
        output = next((str(item[0]) for item in measures if str(item[0]).casefold() in ordering_text), str(measures[0][0]))
        ordering.append([output, direction])
    return {
        "filters": filters, "temporal_filters": [], "dimensions": grouping,
        "measures": measures, "joins": [], "ordering": ordering,
        "limit": requirements.get("limit"),
    }


def _semantic_planner_prompt(query: str, state: P12State) -> str:
    """Build the complete planner context from runtime-only allowlisted inputs."""
    return (
        "Create a compact SemanticPlanDraft for the confirmed selection. "
        "You cannot search or change tables. Call submit_semantic_plan_draft. "
        "Use only these forms: filters [[column, operator, value]], "
        "dimensions [[output, column]], measures [[output, operation, [columns]]], "
        "ordering [[output, direction]], limit, and structured joins. "
        "Allowed operations: count_rows, count_distinct, sum, mean, min, max, "
        "ratio, difference, custom. Correct only fields reported by validation; "
        "at most two tool attempts. Never use benchmark, gold, reference, expected "
        "answer, evaluator, judge, or prior results.\n\n"
        f"QUESTION:\n{query}\n\n"
        "CONFIRMED TABLES:\n"
        + json.dumps(state.confirmed_tables, ensure_ascii=False)
        + "\n\nDECLARED REQUIREMENTS:\n"
        + json.dumps(state.selection_requirements, ensure_ascii=False)
        + "\n\nINSPECTED RUNTIME EVIDENCE:\n"
        + _inspected_runtime_evidence(state)
    )

def phase12_agent(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    solr_client: LocalSolrClient,
    csv_dir: Path,
    hint: str = "",
    portal_name: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check: Callable[[], None] | None = None,
    retrieval_config: RetrievalConfig | None = None,
    state: P12State | None = None,
    retrieval_observer: Callable[[RetrievalRun], None] | None = None,
    planner_enabled: bool = False,
    require_semantic_plan: bool = True,
) -> tuple[list[str], list[str], SolrMetadata, str, str, int]:

    state = state or P12State()
    tools_manager = Phase12ToolsManager(
        state,
        solr_client,
        all_files,
        csv_dir,
        question=query,
        retrieval_config=retrieval_config,
        retrieval_observer=retrieval_observer,
    )
    agent_tools = tools_manager.get_tools()

    system_prompt = pm.render(
        "unified_architect",
        "system_prompt",
        portal_name=portal_name,
        hint=hint
    )

    token_counter = next(
        (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
        None,
    )
    if token_counter:
        token_counter.reset_counts()
    reset_llm_token_usage(llm)

    agent_prompt = pm.render(
        "unified_architect",
        "user_prompt",
        question=query
    )

    stream_trace = io.StringIO()

    def emit_stream(delta: str) -> None:
        if not delta:
            return
        stream_trace.write(delta)
        print(delta, end="", flush=True)
        if stream_callback is not None:
            stream_callback(delta)

    thinking_capture = ThinkingCapture()
    dispatcher = get_dispatcher()
    dispatcher.add_event_handler(thinking_capture)
    emit_stream(
        "\n**Unified Architect & Search agent started**\n"
        "- Streaming model output and tool inspections below.\n"
    )

    from lakegen.agents.agent_runner import run_agent_workflow
    
    try:
        agent_resp = run_agent_workflow(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=agent_prompt,
            agent_name="unified_explorer",
            emit_stream=emit_stream,
            cancel_check=cancel_check,
            tools=agent_tools,
            max_iterations=16 if planner_enabled else 10,
            max_repeats=3,
            max_tool_calls=12 if planner_enabled else 8,
            timeout_seconds=300,
        )
    except Phase2AgentStall as stall_err:
        state.initial_stall_reason = str(stall_err)
        if planner_enabled and require_semantic_plan and state.inspected_candidates():
            state.recovery_started = True
            emit_stream(
                "\n\n**Semantic planner recovery started**\n"
                "- Using only the question and already inspected runtime evidence.\n"
            )
            planner_prompt = (
                "The discovery exploration stopped before producing its required typed "
                "semantic plan. Do not search again. Using only the user question and "
                "the schemas/values already obtained through prior tool calls, inspect "
                "a selected table only if essential evidence is still missing, then call "
                "confirm_unified_selection. The semantic_plan must explicitly contain "
                "filters, temporal_filters, dimensions, measures, joins, ordering, limit, "
                "output_columns, null_policy, and table_roles. If runtime evidence proves "
                "the selection cannot answer the question, return REJECT_KEYWORDS with a "
                "concrete missing requirement. Never use benchmark, gold, expected answer, "
                "reference output, or prior evaluator output.\n\n"
                "INSPECTED RUNTIME EVIDENCE (authoritative, benchmark-blind):\n"
                + _inspected_runtime_evidence(state)
            )
            try:
                agent_resp = run_agent_workflow(
                    llm=llm, system_prompt=system_prompt, user_prompt=planner_prompt,
                    agent_name="semantic_planner_recovery", emit_stream=emit_stream,
                    cancel_check=cancel_check, tools=agent_tools,
                    max_iterations=6, max_repeats=2, max_tool_calls=4,
                    timeout_seconds=180,
                )
            except Phase2AgentStall as planner_stall:
                state.recovery_stop_reason = str(planner_stall)
                agent_resp = "FINAL_PAYLOAD: " + json.dumps({
                    "tables": "",
                    "reasoning": (
                        "REJECT_KEYWORDS: semantic planner could not establish a "
                        "complete runtime-grounded plan after bounded recovery: "
                        f"{planner_stall}"
                    ),
                })
        else:
            inspected_fallback = state.inspected_candidates()[:2]
            fallback_payload = {
                "tables": ", ".join(inspected_fallback),
                "reasoning": (
                    f"Phase loop guard triggered: {stall_err}. "
                    "Fallback restricted to inspected candidates."
                ),
            }
            emit_stream(
                "\n\n**Loop guard triggered**\n"
                f"- Reason: `{str(stall_err)}`\n"
                "- Action: using only inspected candidates.\n"
            )
            agent_resp = f"FINAL_PAYLOAD: {json.dumps(fallback_payload)}"
    except WorkflowCancelled:
        raise
    except Exception as agent_err:
        err_msg = str(agent_err)
        state.initial_stall_reason = f"{type(agent_err).__name__}: {err_msg}"
        if planner_enabled and require_semantic_plan and state.inspected_candidates():
            state.recovery_started = True
            emit_stream(
                "\n\n**Semantic planner recovery started after discovery error**\n"
                "- Using only the question and already inspected runtime evidence.\n"
            )
            planner_prompt = (
                "Discovery ended without its required semantic plan. Do not search "
                "again. Using only the user question and already inspected runtime "
                "schemas/values, call confirm_unified_selection now. Supply every "
                "semantic_plan field: filters, temporal_filters, dimensions, measures, "
                "joins, ordering, limit, output_columns, null_policy, and table_roles. "
                "Never use benchmark, reference, gold, expected-answer, evaluator, or "
                "prior-judge data. If the inspected tables are insufficient, return "
                "REJECT_KEYWORDS with the concrete missing runtime requirement.\n\n"
                "INSPECTED RUNTIME EVIDENCE (authoritative, benchmark-blind):\n"
                + _inspected_runtime_evidence(state)
            )
            try:
                agent_resp = run_agent_workflow(
                    llm=llm, system_prompt=system_prompt, user_prompt=planner_prompt,
                    agent_name="semantic_planner_recovery", emit_stream=emit_stream,
                    cancel_check=cancel_check, tools=agent_tools,
                    max_iterations=6, max_repeats=2, max_tool_calls=4,
                    timeout_seconds=180,
                )
                err_msg = ""
            except Phase2AgentStall as planner_stall:
                state.recovery_stop_reason = str(planner_stall)
                agent_resp = "FINAL_PAYLOAD: " + json.dumps({
                    "tables": "",
                    "reasoning": (
                        "REJECT_KEYWORDS: semantic planner could not establish a "
                        "complete runtime-grounded plan after bounded recovery: "
                        f"{planner_stall}"
                    ),
                })
                err_msg = ""
        if not err_msg:
            pass
        elif "Max iterations" in err_msg:
            reason = "Agent exceeded maximum iterations without a final decision. Fallback to top candidates."
        else:
            reason = f"Agent error: {err_msg[:120]}. Fallback to top 2."
        if err_msg:
            inspected_fallback = state.inspected_candidates()[:2]
            fallback_payload = {
                "tables": ", ".join(inspected_fallback),
                "reasoning": reason,
            }
            emit_stream(f"\n[agent error] {str(agent_err)[:160]}\n")
            agent_resp = f"FINAL_PAYLOAD: {json.dumps(fallback_payload)}"
    finally:
        if (
            planner_enabled and require_semantic_plan
            and state.confirmed_tables
            and not state.selection_plan.get("coder_brief")
            and not state.selection_plan.get("semantic_plan")
        ):
            state.recovery_started = True
            emit_stream(
                "\n\n**Dedicated semantic planner started**\n"
                "- Search tools are unavailable; using only confirmed selection and runtime evidence.\n"
            )
            planner_prompt = _semantic_planner_prompt(query, state)
            try:
                planner_response = run_agent_workflow(
                    llm=llm, system_prompt=(
                        "You are a benchmark-blind semantic planner. Compile explicit "
                        "question semantics against inspected runtime schemas."
                    ),
                    user_prompt=planner_prompt,
                    agent_name="dedicated_semantic_planner", emit_stream=emit_stream,
                    cancel_check=cancel_check,
                    tools=tools_manager.get_semantic_planner_tools(),
                    max_iterations=5, max_repeats=3, max_tool_calls=2,
                    timeout_seconds=180,
                )
                if state.selection_plan.get("semantic_plan"):
                    agent_resp = planner_response
                elif state.semantic_planner_attempts < 2:
                    candidate = _extract_plausible_json(planner_response)
                    if isinstance(candidate, dict):
                        draft = candidate.get(
                            "draft", candidate.get("semantic_plan", candidate)
                        )
                        if isinstance(draft, dict):
                            agent_resp = tools_manager.submit_semantic_plan_draft(draft)
            except (Phase2AgentStall, ValueError) as planner_error:
                state.recovery_stop_reason = str(planner_error)

            if not state.selection_plan.get("semantic_plan"):
                try:
                    table = state.confirmed_tables[0] if len(state.confirmed_tables) == 1 else ""
                    schema = (
                        {str(column) for column in read_table(csv_dir / table, nrows=0).columns}
                        if table else set()
                    )
                    fallback_draft = _conservative_draft_from_requirements(
                        state.selection_requirements, state.confirmed_tables, schema
                    )
                    if fallback_draft and state.semantic_planner_attempts < 2:
                        agent_resp = tools_manager.submit_semantic_plan_draft(fallback_draft)
                        state.selection_plan_source = "semantic_requirements_fallback"
                except ValueError as fallback_error:
                    state.recovery_stop_reason = str(fallback_error)
        agent_stream_trace = stream_trace.getvalue()
        full_trace = "--- Unified Phase Activity Log ---\n" + agent_stream_trace
        stream_trace.close()
        dispatcher.event_handlers.remove(thinking_capture)

    tokens = 0
    if token_counter:
        tokens = (token_counter.prompt_llm_token_count +
                     token_counter.completion_llm_token_count)
        token_counter.reset_counts()
    tokens = max(tokens, get_llm_token_usage(llm))
    
    # Parse agent_resp
    selected = []
    reasoning = ""
    parsed_plan: dict[str, object] | None = None
    parsed_advisories: list[str] = []
    try:
        match = re.search(r"FINAL_PAYLOAD:\s*(\{.*\})", agent_resp, re.DOTALL)
        if match:
            payload = json.loads(match.group(1))
            tables_raw = payload.get("tables", "")
            reasoning = payload.get("reasoning", "")
            parsed_plan = payload.get("selection_plan")
            parsed_advisories = payload.get("advisories") or []
            for t in [x.strip() for x in tables_raw.split(",")]:
                if t in all_files and t not in selected:
                    selected.append(t)
        else:
            reasoning = agent_resp
    except json.JSONDecodeError:
        pass

    if not selected:
        selected = state.inspected_candidates()[:3]

    # confirm_unified_selection persists the validated contract before returning
    # its terminal payload. Prefer that authoritative state if the surrounding
    # agent response was truncated or its FINAL_PAYLOAD wrapper was malformed.
    if not parsed_plan and state.selection_plan:
        parsed_plan = dict(state.selection_plan)
        parsed_advisories.extend(state.selection_advisories)
        if state.selection_plan_source == "none":
            state.selection_plan_source = "confirmed_state_recovery"

    if not parsed_plan:
        parsed_plan, recovered_advisories = _recover_minimal_selection_plan(
            selected, reasoning
        )
        parsed_advisories.extend(recovered_advisories)
        state.selection_plan_source = "minimal_fallback"
    if parsed_plan:
        state.selection_plan = parsed_plan
        state.selection_advisories = parsed_advisories
        if state.selection_plan_source == "none":
            state.selection_plan_source = "final_payload"
    reasoning = _reasoning_with_selection_plan(
        reasoning, parsed_plan, parsed_advisories
    )

    return selected, state.used_keywords, state.solr_meta, reasoning, full_trace, tokens
