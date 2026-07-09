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
from llama_index.core.callbacks import CallbackManager
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from lakegen.phase2_logging import (
    Phase2AgentStall,
    detect_phase2_agent_stall,
    format_phase2_tool_args,
    format_phase2_tool_call,
    format_phase2_tool_output,
    format_phase2_tool_result,
)
from lakegen.types import SolrMetadata, StreamCallback
from lakegen.ui.state import WorkflowCancelled
from lakegen.instrumentation import ThinkingCapture
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient
from lakegen.tools_p12 import P12State, make_p12_tools

def phase1_updated_agent(
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
) -> tuple[list[str], list[str], SolrMetadata, str, str, int]:
    
    state = P12State()
    agent_tools = make_p12_tools(state, solr_client, all_files, csv_dir)

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

    try:
        async def _run_agent():
            explorer = FunctionAgent(
                name="unified_explorer", 
                tool_retriever=agent_tools, 
                llm=llm,
                system_prompt=system_prompt,
            )

            handler = explorer.run(
                user_msg=agent_prompt,
                max_iterations=16,
            )

            tool_call_count = 0
            tool_result_count = 0
            tool_call_signatures: dict[str, int] = {}
            async for event in handler.stream_events():
                if cancel_check is not None:
                    cancel_check()
                if isinstance(event, AgentStream):
                    emit_stream(event.delta or "")
                    if tool_call_count > 0:
                        stall_reason = detect_phase2_agent_stall(
                            stream_trace.getvalue()
                        )
                        if stall_reason:
                            raise Phase2AgentStall(stall_reason)
                elif isinstance(event, ToolCall):
                    tool_call_count += 1
                    tool_name = getattr(event, 'tool_name', 'unknown_tool')
                    tool_signature = (
                        f"{tool_name}:"
                        f"{format_phase2_tool_args(event)}"
                    )
                    tool_call_signatures[tool_signature] = (
                        tool_call_signatures.get(tool_signature, 0) + 1
                    )

                    max_repeats = 4
                    
                    if tool_call_signatures[tool_signature] >= max_repeats:
                        raise Phase2AgentStall(
                            "repeated identical tool call: "
                            f"{tool_name}"
                        )
                    emit_stream(format_phase2_tool_call(event, tool_call_count))
                elif isinstance(event, ToolCallResult):
                    tool_result_count += 1
                    emit_stream(format_phase2_tool_result(event, tool_result_count))
                    tool_output = getattr(event, "tool_output", None)
                    output = format_phase2_tool_output(tool_output).lower()
                    if "missing in active dataset" in output:
                        emit_stream(
                            "\n⚠️ **File not found** – the requested table "
                            "is not in the active dataset. "
                            "The agent will try an alternative.\n"
                        )

            return await handler

        if hasattr(llm, "_async_client"):
            llm._async_client = None

        try:
            res = asyncio.run(_run_agent())
        except Exception:
            raise
        agent_resp = str(getattr(res, "response", res)).strip()
    except Phase2AgentStall as stall_err:
        fallback_payload = {
            "tables": ", ".join(state.all_candidates[:2]),
            "reasoning": (
                f"Phase loop guard triggered: {stall_err}. "
                "Fallback to top candidates."
            ),
        }
        emit_stream(
            "\n\n**Loop guard triggered**\n"
            f"- Reason: `{str(stall_err)}`\n"
            "- Action: using top candidates.\n"
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
            
        fallback_payload = {
            "tables": ", ".join(state.all_candidates[:2]),
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

    Settings.callback_manager = CallbackManager([])
    
    # Parse agent_resp
    selected = []
    reasoning = ""
    try:
        match = re.search(r"FINAL_PAYLOAD:\s*(\{.*\})", agent_resp, re.DOTALL)
        if match:
            payload = json.loads(match.group(1))
            tables_raw = payload.get("tables", "")
            reasoning = payload.get("reasoning", "")
            for t in [x.strip() for x in tables_raw.split(",")]:
                if t in all_files and t not in selected:
                    selected.append(t)
        else:
            reasoning = agent_resp
    except json.JSONDecodeError:
        pass

    if not selected:
        selected = state.all_candidates[:3]

    return selected, state.used_keywords, state.solr_meta, reasoning, full_trace, tokens
