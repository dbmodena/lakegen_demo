import asyncio
from typing import Callable, Any

from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.llms import LLM, ChatMessage
from pydantic import Field

from lakegen.phases.logging import (
    Phase2AgentStall,
    detect_phase2_agent_stall,
    format_phase2_tool_args,
    format_phase2_tool_call,
    format_phase2_tool_output,
    format_phase2_tool_result,
)


class _ToolPruningFunctionAgent(FunctionAgent):
    """A FunctionAgent that offers the model only the tools still worth calling.

    Every call to a spent tool costs a full model turn and one of the agent's
    capped tool calls just to hear a refusal, so the model is simply not shown
    it. ``get_tools`` is left alone: a stray call to a withdrawn tool still
    reaches that tool's own refusal message rather than a generic "not found".

    ``context_editor``, when set, rewrites the in-run history (the scratchpad
    of assistant tool calls and tool results) right before each model call,
    e.g. to shorten results later calls superseded. The rewrite is stored, so
    it is also what the next edit starts from.
    """

    tool_available: Callable[[str], bool] | None = Field(default=None, exclude=True)
    context_editor: Callable[[list[ChatMessage]], list[ChatMessage]] | None = Field(
        default=None, exclude=True
    )

    async def take_step(self, ctx, llm_input, tools, memory):
        if self.tool_available is not None:
            tools = [tool for tool in tools if self.tool_available(tool.metadata.name)]
        if self.context_editor is not None:
            scratchpad = await ctx.store.get(self.scratchpad_key, default=[])
            await ctx.store.set(self.scratchpad_key, self.context_editor(list(scratchpad)))
        return await super().take_step(ctx, llm_input, tools, memory)


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
    max_tool_calls: int | None = None,
    timeout_seconds: float | None = None,
    chat_history: list | None = None,
    tool_available: Callable[[str], bool] | None = None,
    context_editor: Callable[[list[ChatMessage]], list[ChatMessage]] | None = None,
) -> str:
    """
    Run a LlamaIndex FunctionAgent and safely yield events/handle stalls.
    Abstracts the boilerplate async loops for phase1 and phase2.

    ``tool_available`` is asked, before every model turn, whether each tool is
    still worth offering; tools it rejects are left out of that turn's request.
    ``context_editor`` rewrites the in-run history before every model turn.
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

        if tool_available is not None or context_editor is not None:
            explorer = _ToolPruningFunctionAgent(
                tool_available=tool_available, context_editor=context_editor, **kwargs
            )
        else:
            explorer = FunctionAgent(**kwargs)

        handler = explorer.run(
            user_msg=user_prompt,
            chat_history=chat_history,
            max_iterations=max_iterations,
        )

        tool_call_count = 0
        tool_result_count = 0
        tool_call_signatures: dict[str, int] = {}
        stream_content = ""
        last_stall_check_len = 0

        async for event in handler.stream_events():
            if cancel_check is not None:
                cancel_check()
                
            if isinstance(event, AgentStream):
                delta = event.delta or ""
                emit_stream(delta)
                stream_content += delta
                if tool_call_count > 0 and (len(stream_content) - last_stall_check_len > 100):
                    stall_reason = detect_phase2_agent_stall(stream_content)
                    if stall_reason:
                        raise Phase2AgentStall(stall_reason)
                    last_stall_check_len = len(stream_content)
                        
            elif isinstance(event, ToolCall):
                tool_call_count += 1
                tool_name = getattr(event, 'tool_name', 'unknown_tool')
                if max_tool_calls is not None and tool_call_count > max_tool_calls:
                    raise Phase2AgentStall(
                        f"tool-call limit reached ({max_tool_calls})"
                    )
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
        workflow = _run_agent()
        if timeout_seconds is not None:
            workflow = asyncio.wait_for(workflow, timeout=timeout_seconds)
        res = asyncio.run(workflow)
    except TimeoutError as exc:
        raise Phase2AgentStall(
            f"workflow timeout reached ({timeout_seconds:g}s)"
        ) from exc
    except Exception:
        raise
    return str(getattr(res, "response", res)).strip()
