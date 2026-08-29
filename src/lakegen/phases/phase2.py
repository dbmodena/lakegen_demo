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
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.ui.state import WorkflowCancelled
from lakegen.agents.instrumentation import ThinkingCapture
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient
from lakegen.agent_tools.tools_p2 import Phase2JudgeToolsManager

from lakegen.phases.utils import (
    format_candidate_context,
    match_local_csv,
    parse_table_selector_response,
    solr_metadata_from_doc,
)
from lakegen.core.resources import get_table_retrieval_service
from lakegen.retrieval import RetrievalConfig, RetrievalMode


def _solr_and_search(
    query: str,
    keywords: list[str],
    solr_client: LocalSolrClient,
    all_files: list[str],
    retrieval_config: RetrievalConfig | None = None,
    table_dir: Path | None = None,
) -> tuple[list[str], SolrMetadata]:
    """Execute the configured retriever and return candidates + metadata."""
    candidates: list[str] = []
    metadata: SolrMetadata = {}
    config = retrieval_config or RetrievalConfig()

    try:
        retriever = get_table_retrieval_service(
            solr_client, config,
            **({"table_dir": table_dir} if config.mode == RetrievalMode.DUCKDB_AGENTIC else {}),
        )
        hits = retriever.retrieve(
            question=query,
            keywords=keywords,
            top_k=config.top_k,
            lexical_fetch_k=max(15, config.top_k),
            q_op="AND",
        )
        print(
            f"[phase2] {config.mode} retrieval keywords={keywords} "
            f"docs_returned={len(hits)}",
            flush=True,
        )

        for hit in hits:
            doc = hit.document
            matched = match_local_csv(doc, all_files)
            if matched is None or matched in candidates:
                continue
            candidates.append(matched)
            metadata[matched] = solr_metadata_from_doc(doc)
            metadata[matched]["retrieval"] = hit.to_log_dict()
            if len(candidates) >= config.top_k:
                break

    except Exception as solr_err:
        print(
            f"[phase2] Solr error {type(solr_err).__name__}: {solr_err}",
            flush=True,
        )
        # Dense setup errors (missing model/vector field/provenance) must not be
        # disguised as a failed Phase 1 keyword query.
        if config.mode != RetrievalMode.KEYWORD:
            raise

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
    retrieval_config: RetrievalConfig | None = None,
) -> Phase2SelectionResult:
    """Run the configured retriever, then judge the retrieved tables.

    Flow:
    1. Execute keyword, semantic, or hybrid retrieval programmatically
    2. If 0 results → immediately return REJECT_KEYWORDS
    3. If results found → run the FunctionAgent to inspect and judge them
    4. Agent accepts (confirm_table_selection) or rejects (REJECT_KEYWORDS)

    Returns:
        (selected, all_candidates, solr_meta, reasoning, full_trace, tokens)
    """

    # ── Step 1: table retrieval (programmatic, not agent-driven) ─────
    config = retrieval_config or RetrievalConfig()
    candidates, solr_meta = _solr_and_search(
        query,
        keywords,
        solr_client,
        all_files,
        config,
        csv_dir,
    )

    # ── Step 2: No results → reject keywords back to Phase 1 ─────────
    if not candidates:
        no_result_msg = (
            f"REJECT_KEYWORDS: No tables found with {config.mode} retrieval "
            f"for [{', '.join(keywords)}]. Try broader or different terms."
        )
        trace = (
            f"[phase2] {config.mode} retrieval with keywords={keywords} returned 0 results.\n"
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
    tools_manager = Phase2JudgeToolsManager(
        candidates,
        csv_dir,
        question=query,
        metadata=solr_meta,
    )
    agent_tools = tools_manager.get_tools()

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
    reset_llm_token_usage(llm)

    candidate_context = format_candidate_context(
        tools_manager.visible_candidates(),
        solr_meta,
    )
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

    visible_candidates = tools_manager.visible_candidates()
    candidates_summary = ", ".join(f"`{c}`" for c in visible_candidates)
    emit_stream(
        "\n**Phase 2 – Table Judge agent started**\n"
        f"- Keywords from Phase 1: `{' '.join(keywords)}`\n"
        f"- {config.mode} retrieval built a pool of {len(candidates)} candidates; "
        f"showing {len(visible_candidates)} initially: "
        f"{candidates_summary}\n"
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
            max_tool_calls=8,
            timeout_seconds=300,
        )
    except Phase2AgentStall as stall_err:
        inspected_fallback = tools_manager.inspected_candidates()[:2]
        fallback_payload = {
            "tables": ", ".join(inspected_fallback),
            "reasoning": (
                f"Phase 2 loop guard triggered: {stall_err}. "
                "Fallback restricted to inspected candidates."
            ),
        }
        emit_stream(
            "\n\n**Phase 2 loop guard triggered**\n"
            f"- Reason: `{str(stall_err)}`\n"
            "- Action: using only inspected candidates as a fallback.\n"
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

        inspected_fallback = tools_manager.inspected_candidates()[:2]
        fallback_payload = {
            "tables": ", ".join(inspected_fallback),
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
    tokens_p2 = max(tokens_p2, get_llm_token_usage(llm))

    selected, reasoning = parse_table_selector_response(
        agent_resp,
        all_files,
        candidates,
    )

    return selected, candidates, solr_meta, reasoning, full_trace, tokens_p2
