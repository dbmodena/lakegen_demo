import asyncio
from typing import Callable, Any

from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.llms import LLM

from lakegen.phase2_logging import (
    Phase2AgentStall,
    detect_phase2_agent_stall,
    format_phase2_tool_args,
    format_phase2_tool_call,
    format_phase2_tool_output,
    format_phase2_tool_result,
)


def run_agent_workflow(
    llm: LLM,
    system_prompt: str,
    user_prompt: str,
    agent_name: str,
    emit_stream: Callable[[str], None],
    cancel_check: Callable[[], None] | None = None,
    tools: list | None = None,
    tool_retriever: Any | None = None,
    max_iterations: int = 10,
    max_repeats: int = 3,
) -> str:
    """
    Run a LlamaIndex FunctionAgent and safely yield events/handle stalls.
    Abstracts the boilerplate async loops for phase1 and phase2.
    """
    async def _run_agent():
        kwargs = {
            "name": agent_name,
            "llm": llm,
            "system_prompt": system_prompt,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_retriever is not None:
            kwargs["tool_retriever"] = tool_retriever

        explorer = FunctionAgent(**kwargs)

        handler = explorer.run(
            user_msg=user_prompt,
            max_iterations=max_iterations,
        )

        tool_call_count = 0
        tool_result_count = 0
        tool_call_signatures: dict[str, int] = {}
        stream_content = ""

        async for event in handler.stream_events():
            if cancel_check is not None:
                cancel_check()
                
            if isinstance(event, AgentStream):
                delta = event.delta or ""
                emit_stream(delta)
                stream_content += delta
                if tool_call_count > 0:
                    stall_reason = detect_phase2_agent_stall(stream_content)
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

                if tool_call_signatures[tool_signature] >= max_repeats:
                    raise Phase2AgentStall(
                        f"repeated identical tool call: {tool_name}"
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
    return str(getattr(res, "response", res)).strip()
