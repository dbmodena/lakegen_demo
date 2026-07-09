import io
import json
import os
import re
import sys
import asyncio
from collections.abc import Callable
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import LLM

from lakegen.phases.logging import (
    Phase2AgentStall,
    detect_phase2_agent_stall,
    format_phase2_tool_args,
    format_phase2_tool_call,
    format_phase2_tool_output,
    format_phase2_tool_result,
)
from lakegen.core.types import Phase2SelectionResult, SolrMetadata, StreamCallback
from lakegen.ui.state import WorkflowCancelled
from lakegen.agents.instrumentation import ThinkingCapture
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient
from lakegen.agent_tools.tools_p2 import make_p2_judge_tools

from lakegen.phases.utils import (
    format_candidate_context,
    match_local_csv,
    parse_table_selector_response,
    solr_metadata_from_doc,
)


def _solr_and_search(
    keywords: list[str],
    solr_client: LocalSolrClient,
    all_files: list[str],
) -> tuple[list[str], SolrMetadata]:
    """Execute a Solr AND query and return matched candidates + metadata."""
    candidates: list[str] = []
    metadata: SolrMetadata = {}

    try:
        solr_response = solr_client.select(tokens=keywords, q_op="AND", rows=15)
        docs = solr_response.get("response", {}).get("docs", [])
        print(
            f"[phase2] Solr AND search keywords={keywords} "
            f"numFound={solr_response.get('response', {}).get('numFound', '?')} "
            f"docs_returned={len(docs)}",
            flush=True,
        )

        for doc in docs:
            matched = match_local_csv(doc, all_files)
            if matched is None or matched in candidates:
                continue
            candidates.append(matched)
            metadata[matched] = solr_metadata_from_doc(doc)
            if len(candidates) >= 10:
                break

    except Exception as solr_err:
        print(
            f"[phase2] Solr error {type(solr_err).__name__}: {solr_err}",
            flush=True,
        )

    return candidates, metadata


def phase2_select_tables(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    keywords: list[str],
    solr_client: LocalSolrClient,
    csv_dir: Path,
    hint: str = "",
    portal_name: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> Phase2SelectionResult:
    """Search Solr with AND logic, then run a judge agent on the results.

    Flow:
    1. Execute Solr AND query programmatically with the provided keywords
    2. If 0 results → immediately return REJECT_KEYWORDS
    3. If results found → run the FunctionAgent to inspect and judge them
    4. Agent accepts (confirm_table_selection) or rejects (REJECT_KEYWORDS)

    Returns:
        (selected, all_candidates, solr_meta, reasoning, full_trace, tokens)
    """

    # ── Step 1: Solr AND search (programmatic, not agent-driven) ──────
    candidates, solr_meta = _solr_and_search(keywords, solr_client, all_files)

    # ── Step 2: No results → reject keywords back to Phase 1 ─────────
    if not candidates:
        no_result_msg = (
            f"REJECT_KEYWORDS: No tables found with AND keywords "
            f"[{', '.join(keywords)}]. Try broader or different terms."
        )
        trace = (
            f"[phase2] Solr AND search with keywords={keywords} returned 0 results.\n"
            f"Rejecting keywords back to Phase 1."
        )
        if stream_callback:
            stream_callback(
                f"\n⚠️ **No tables found** with keywords: "
                f"`{' '.join(keywords)}`\n"
                f"Sending feedback to Phase 1 for better keywords.\n"
            )
        return [], [], {}, no_result_msg, trace, 0

    # ── Step 3: Prepare agent with judge-only tools (no search_solr) ──
    agent_tools = make_p2_judge_tools(candidates, csv_dir)

    system_prompt = pm.render(
        "data_architect",
        "system_prompt",
        portal_name=portal_name,
        hint=hint,
    )

    token_counter = next(
        (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
        None,
    )
    if token_counter:
        token_counter.reset_counts()

    candidate_context = format_candidate_context(candidates, solr_meta)
    agent_prompt = pm.render(
        "data_architect",
        "user_prompt",
        question=query,
        keywords_str=" ".join(keywords),
        enriched_candidates_info=candidate_context,
        table_hint=hint,
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

    candidates_summary = ", ".join(f"`{c}`" for c in candidates)
    emit_stream(
        "\n**Phase 2 – Table Judge agent started**\n"
        f"- Keywords from Phase 1: `{' '.join(keywords)}`\n"
        f"- Solr AND returned {len(candidates)} candidates: {candidates_summary}\n"
        "- Agent inspecting and judging tables below.\n"
    )

    # ── Step 4: Run the judge agent ──────────────────────────────────
    from lakegen.agents.agent_runner import run_agent_workflow
    
    try:
        agent_resp = run_agent_workflow(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=agent_prompt,
            agent_name="table_judge",
            emit_stream=emit_stream,
            cancel_check=cancel_check,
            tools=agent_tools,
            max_iterations=10,
            max_repeats=3,
        )
    except Phase2AgentStall as stall_err:
        fallback_payload = {
            "tables": ", ".join(candidates[:2]),
            "reasoning": (
                f"Phase 2 loop guard triggered: {stall_err}. "
                "Fallback to top Solr candidates."
            ),
        }
        emit_stream(
            "\n\n**Phase 2 loop guard triggered**\n"
            f"- Reason: `{str(stall_err)}`\n"
            "- Action: using the top Solr candidates as a fallback.\n"
        )
        agent_resp = f"FINAL_PAYLOAD: {json.dumps(fallback_payload)}"
    except WorkflowCancelled:
        raise
    except Exception as agent_err:
        err_msg = str(agent_err)
        if "Max iterations" in err_msg:
            reason = "Agent exceeded maximum iterations. Fallback to top candidates."
        else:
            reason = f"Agent error: {err_msg[:120]}. Fallback to top 2."

        fallback_payload = {
            "tables": ", ".join(candidates[:2]),
            "reasoning": reason,
        }
        emit_stream(f"\n[phase2 agent error] {str(agent_err)[:160]}\n")
        agent_resp = f"FINAL_PAYLOAD: {json.dumps(fallback_payload)}"
    finally:
        agent_stream_trace = stream_trace.getvalue()
        full_trace = "--- Phase 2 Activity Log ---\n" + agent_stream_trace
        stream_trace.close()
        dispatcher.event_handlers.remove(thinking_capture)

    tokens_p2 = 0
    if token_counter:
        tokens_p2 = (token_counter.prompt_llm_token_count +
                     token_counter.completion_llm_token_count)
        token_counter.reset_counts()

    Settings.callback_manager = CallbackManager([])

    selected, reasoning = parse_table_selector_response(
        agent_resp,
        all_files,
        candidates,
    )

    return selected, candidates, solr_meta, reasoning, full_trace, tokens_p2
