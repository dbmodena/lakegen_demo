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
from lakegen.tools import make_agent_tools
from lakegen.utils import ThinkingCapture
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient
from .utils import match_local_csv, solr_metadata_from_doc, format_candidate_context

def phase1_updated_agent(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    solr_client: LocalSolrClient,
    csv_dir: Path,
    blend_db: Path,
    hint: str = "",
    portal_name: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check: Callable[[], None] | None = None,
) -> tuple[list[str], list[str], SolrMetadata, str, str, int]:
    
    # We will track all tables retrieved from solr and their metadata here
    all_candidates: list[str] = []
    solr_meta: SolrMetadata = {}
    used_keywords: list[str] = []

    def search_solr(keywords_str: str) -> str:
        """
        Search for relevant tables in Solr using a space-separated string of keywords.
        Because this uses AND logic, use ONLY 2-3 essential keywords at most to avoid getting zero results.
        Example: "sales 2024"
        Returns the top matching table names and their schema descriptions.
        """
        try:
            keywords = [k.strip() for k in keywords_str.split(" ") if k.strip()]
            nonlocal used_keywords
            used_keywords = keywords
            
            solr_response = solr_client.select(tokens=keywords, q_op="AND", rows=15)
            docs = solr_response.get("response", {}).get("docs", [])
            
            candidates: list[str] = []
            for doc in docs:
                matched = match_local_csv(doc, all_files)
                if matched is None or matched in candidates:
                    continue
                candidates.append(matched)
                solr_meta[matched] = solr_metadata_from_doc(doc)
                if matched not in all_candidates:
                    all_candidates.append(matched)
                if len(candidates) >= 10:
                    break
            
            if not candidates:
                return f"Keywords used: {keywords}\nNo tables found. Try with fewer or different keywords."
            
            return f"Keywords used: {keywords}\n\n" + format_candidate_context(candidates, solr_meta)
        except Exception as e:
            return f"Error querying Solr: {str(e)}"

    def confirm_unified_selection(selected_files: str, reasoning: str) -> str:
        """
        CRITICAL: Use this tool ONLY when you have identified the required files after searching solr and inspecting them.
        - selected_files: A comma-separated string of the exact file names needed (e.g., "sales.csv, dates.csv").
        - reasoning: Write a brief explanation IN ENGLISH.
        Calling this tool means you have successfully finished the task.
        """
        dati_uscita = {
            "tables": selected_files,
            "reasoning": reasoning
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

    base_tools = make_agent_tools(blend_db, csv_dir=csv_dir)
    # Remove the standard confirm_table_selection to replace it with ours
    base_tools = [t for t in base_tools if t.metadata.name != "confirm_table_selection"]
    
    agent_tools = [
        FunctionTool.from_defaults(fn=search_solr),
        *base_tools,
        FunctionTool.from_defaults(fn=confirm_unified_selection, return_direct=True),
    ]

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

    old_stdout = sys.stdout
    capture = io.StringIO()
    stream_trace = io.StringIO()
    sys.stdout = capture

    def emit_stream(delta: str) -> None:
        if not delta:
            return
        stream_trace.write(delta)
        print(delta, end="", flush=True, file=old_stdout)
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
                tools=agent_tools, 
                llm=llm,
                system_prompt=system_prompt,
            )

            handler = explorer.run(
                user_msg=agent_prompt,
                max_iterations=15,
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
                    tool_signature = (
                        f"{getattr(event, 'tool_name', 'unknown_tool')}:"
                        f"{format_phase2_tool_args(event)}"
                    )
                    tool_call_signatures[tool_signature] = (
                        tool_call_signatures.get(tool_signature, 0) + 1
                    )
                    if tool_call_signatures[tool_signature] >= 2:
                        raise Phase2AgentStall(
                            "repeated identical tool call: "
                            f"{getattr(event, 'tool_name', 'unknown_tool')}"
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

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(_run_agent())
        finally:
            loop.close()
        agent_resp = str(getattr(res, "response", res)).strip()
    except Phase2AgentStall as stall_err:
        fallback_payload = {
            "tables": ", ".join(all_candidates[:2]),
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
        fallback_payload = {
            "tables": ", ".join(all_candidates[:2]),
            "reasoning": (
                f"Agent error: {str(agent_err)[:80]}. Fallback to top 2."
            ),
        }
        emit_stream(f"\n[agent error] {str(agent_err)[:160]}\n")
        agent_resp = f"FINAL_PAYLOAD: {json.dumps(fallback_payload)}"
    finally:
        sys.stdout = old_stdout
        stdout_trace = capture.getvalue()
        agent_stream_trace = stream_trace.getvalue()
        full_trace = stdout_trace + "\n\n--- Unified Phase Activity Log ---\n" + agent_stream_trace
        capture.close()
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
        selected = all_candidates[:3]

    return selected, used_keywords, solr_meta, reasoning, full_trace, tokens
