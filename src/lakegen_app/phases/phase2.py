import io
import os
import sys
from pathlib import Path

import anyio
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

from lakegen_app.phase2_logging import (
    Phase2AgentStall,
    detect_phase2_agent_stall,
    format_phase2_tool_args,
    format_phase2_tool_call,
    format_phase2_tool_result,
)
from lakegen_app.types import Phase2SelectionResult, SolrMetadata, StreamCallback
from prompts.prompt_manager import PromptManager
from src.tools import make_agent_tools
from src.utils import ThinkingCapture

from .utils import (
    format_candidate_context,
    parse_table_selector_response,
    prepare_candidate_index,
)


def phase2_table_selector_agent(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    candidates: list[str],
    solr_meta: SolrMetadata,
    csv_dir: Path,
    blend_db: Path,
    activity_log_parts: list[str],
    hint: str = "",
    stream_callback: StreamCallback | None = None,
) -> tuple[list[str], str, str, int]:
    """Inspect retrieved candidates and choose or reject tables."""
    try:
        architect_system_prompt = pm.render("data_architect", "system_prompt")
        agent_tools = make_agent_tools(blend_db, csv_dir=csv_dir)

        token_counter = next(
            (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
            None,
        )
        if token_counter:
            token_counter.reset_counts()

        agent_prompt = pm.render("data_architect", "user_prompt",
                                 question=query,
                                 enriched_candidates_info=format_candidate_context(
                                     candidates, solr_meta
                                 ),
                                 table_hint=hint)

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
            "\n**Data Architect agent started**\n"
            "- Streaming model output and tool inspections below.\n"
        )

        try:
            async def _run_agent():
                explorer = FunctionAgent(
                    name="data_explorer", 
                    tools=agent_tools, 
                    llm=llm,
                    system_prompt=architect_system_prompt,
                    # verbose=False, 
                    # streaming=True,
                    # early_stopping_method="generate",
                )

                handler = explorer.run(
                    user_msg=agent_prompt,
                    max_iterations=10,
                    # early_stopping_method="generate",
                )

                tool_call_count = 0
                tool_result_count = 0
                tool_call_signatures: dict[str, int] = {}
                async for event in handler.stream_events():
                    if isinstance(event, AgentStream):
                        emit_stream(event.delta or "")
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
                            continue
                            raise Phase2AgentStall(
                                "repeated identical tool call: "
                                f"{getattr(event, 'tool_name', 'unknown_tool')}"
                            )
                        emit_stream(format_phase2_tool_call(event, tool_call_count))
                    elif isinstance(event, ToolCallResult):
                        tool_result_count += 1
                        emit_stream(format_phase2_tool_result(event, tool_result_count))

                return await handler

            res = anyio.run(_run_agent)
            agent_resp = str(getattr(res, "response", res)).strip()
        except Phase2AgentStall as stall_err:
            emit_stream(
                "\n\n**Phase 2 loop guard triggered**\n"
                f"- Reason: `{str(stall_err)}`\n"
                "- Action: using the top Solr candidates as a fallback.\n"
            )
            agent_resp = (
                f'FINAL_PAYLOAD: {{"tables": "{", ".join(candidates[:2])}",'
                ' "reasoning": "Stopped a repeated Data Architect thought loop. '
                'Fallback to top Solr candidates."}}'
            )
        except Exception as agent_err:
            emit_stream(f"\n[phase2 agent error] {str(agent_err)[:160]}\n")
            agent_resp = (
                f'FINAL_PAYLOAD: {{"tables": "{", ".join(candidates[:2])}",'
                f' "reasoning": "Agent error: {str(agent_err)[:80]}. Fallback to top 2."}}'
            )
        finally:
            sys.stdout = old_stdout
            stdout_trace = capture.getvalue()
            agent_stream_trace = stream_trace.getvalue()
            full_trace = stdout_trace
            phase2_activity_trace = "".join(activity_log_parts) + agent_stream_trace
            if phase2_activity_trace:
                full_trace += (
                    "\n\n--- Phase 2 Activity Log ---\n"
                    f"{phase2_activity_trace}"
                )
            capture.close()
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
        return selected, reasoning, full_trace, tokens_p2

    finally:
        if blend_db.exists():
            try:
                os.remove(blend_db)
            except Exception:
                pass


def phase2_select_tables(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    candidates: list[str],
    solr_meta: SolrMetadata,
    csv_dir: Path,
    db_path: Path,
    activity_log_parts: list[str] | None = None,
    hint: str = "",
    stream_callback: StreamCallback | None = None,
) -> Phase2SelectionResult:
    activity_log = activity_log_parts or []
    blend_db = prepare_candidate_index(
        candidates=candidates,
        csv_dir=csv_dir,
        db_path=db_path,
        activity_log_parts=activity_log,
        stream_callback=stream_callback,
    )
    selected, reasoning, full_trace, tokens_p2 = phase2_table_selector_agent(
        query=query,
        llm=llm,
        pm=pm,
        all_files=all_files,
        candidates=candidates,
        solr_meta=solr_meta,
        csv_dir=csv_dir,
        blend_db=blend_db,
        activity_log_parts=activity_log,
        hint=hint,
        stream_callback=stream_callback,
    )
    return selected, candidates, solr_meta, reasoning, full_trace, tokens_p2
