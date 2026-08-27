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
from lakegen.ui.state import WorkflowCancelled
from lakegen.agents.instrumentation import ThinkingCapture
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient
from lakegen.agent_tools.tools_p12 import P12State, make_p12_tools
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
        "recovered_from_existing_discovery_context": True,
    }
    return plan, [
        "The structured selection plan was recovered from the existing discovery "
        "decision; treat it as guidance, not as a blocking constraint."
    ]

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
) -> tuple[list[str], list[str], SolrMetadata, str, str, int]:

    state = state or P12State()
    agent_tools = make_p12_tools(
        state,
        solr_client,
        all_files,
        csv_dir,
        question=query,
        retrieval_config=retrieval_config,
        retrieval_observer=retrieval_observer,
    )

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
            max_iterations=10,
            max_repeats=3,
            max_tool_calls=8,
            timeout_seconds=300,
        )
    except Phase2AgentStall as stall_err:
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
        if "Max iterations" in err_msg:
            reason = "Agent exceeded maximum iterations without a final decision. Fallback to top candidates."
        else:
            reason = f"Agent error: {err_msg[:120]}. Fallback to top 2."
            
        inspected_fallback = state.inspected_candidates()[:2]
        fallback_payload = {
            "tables": ", ".join(inspected_fallback),
            "reasoning": reason,
        }
        emit_stream(f"\n[agent error] {str(agent_err)[:160]}\n")
        agent_resp = f"FINAL_PAYLOAD: {json.dumps(fallback_payload)}"
    finally:
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

    if not parsed_plan:
        parsed_plan, recovered_advisories = _recover_minimal_selection_plan(
            selected, reasoning
        )
        parsed_advisories.extend(recovered_advisories)
    if parsed_plan:
        state.selection_plan = parsed_plan
        state.selection_advisories = parsed_advisories
    reasoning = _reasoning_with_selection_plan(
        reasoning, parsed_plan, parsed_advisories
    )

    return selected, state.used_keywords, state.solr_meta, reasoning, full_trace, tokens
