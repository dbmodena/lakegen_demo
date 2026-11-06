"""The agent runner: a LlamaIndex FunctionAgent run to its answer, its events streamed, its stalls stopped."""

import asyncio
from collections.abc import Callable
from typing import Any

from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.llms import LLM

from lakegen.llm.agent_activity import (
    AgentStall,
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
    max_tool_calls: int | None = None,
    timeout_seconds: float | None = None,
    chat_history: list | None = None,
) -> str:
    """Run a LlamaIndex FunctionAgent to its answer, streaming its events and stopping it when it stalls.

    Abstracts the boilerplate async loops of the coder, the synthesiser and the judge.

    Args:
        llm: The chat model.
        system_prompt: The agent's system prompt.
        user_prompt: The user message the run starts from.
        agent_name: The agent's name.
        emit_stream: Receives the streamed text and the tool-call and result blocks.
        cancel_check: Called before every event; raises to cancel the run.
        tools: The agent's tools.
        tool_retriever: A retriever of tools, instead of or beside ``tools``.
        max_iterations: The agent's iteration cap.
        max_repeats: How many identical tool calls stop the run.
        max_tool_calls: How many tool calls stop the run; None for no cap.
        timeout_seconds: How long the run may take; None for no limit.
        chat_history: Earlier messages of the conversation.

    Returns:
        The agent's final response, stripped.

    Raises:
        AgentStall: The run timed out, or (from the event loop) it stalled,
            repeated a tool call or reached the tool-call cap.
    """

    async def _run_agent():
        explorer = _build_agent(llm, system_prompt, agent_name, tools, tool_retriever)
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
                last_stall_check_len = _check_stream_stall(
                    stream_content, last_stall_check_len, tool_call_count
                )

            elif isinstance(event, ToolCall):
                tool_call_count += 1
                _check_tool_call(
                    event,
                    tool_call_count,
                    tool_call_signatures,
                    max_tool_calls=max_tool_calls,
                    max_repeats=max_repeats,
                )
                emit_stream(format_phase2_tool_call(event, tool_call_count))

            elif isinstance(event, ToolCallResult):
                tool_result_count += 1
                _emit_tool_result(event, tool_result_count, emit_stream)

        return await handler

    if hasattr(llm, "_async_client"):
        llm._async_client = None

    try:
        workflow = _run_agent()
        if timeout_seconds is not None:
            workflow = asyncio.wait_for(workflow, timeout=timeout_seconds)
        result = asyncio.run(workflow)
    except TimeoutError as exc:
        raise AgentStall(f"workflow timeout reached ({timeout_seconds:g}s)") from exc
    return str(getattr(result, "response", result)).strip()


def _build_agent(
    llm: LLM,
    system_prompt: str,
    agent_name: str,
    tools: list | None,
    tool_retriever: Any | None,
) -> FunctionAgent:
    """The agent of a run.

    Args:
        llm: The chat model.
        system_prompt: The agent's system prompt.
        agent_name: The agent's name.
        tools: The agent's tools, when given.
        tool_retriever: A retriever of tools, when given.

    Returns:
        The agent.
    """
    kwargs = {
        "name": agent_name,
        "llm": llm,
        "system_prompt": system_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_retriever is not None:
        kwargs["tool_retriever"] = tool_retriever
    return FunctionAgent(**kwargs)


def _check_stream_stall(
    stream_content: str, last_stall_check_len: int, tool_call_count: int
) -> int:
    """Check the streamed text for a stall once a tool was called and 100 more characters arrived.

    Args:
        stream_content: Everything streamed so far.
        last_stall_check_len: The stream's length at the last check.
        tool_call_count: The tool calls so far.

    Returns:
        The stream's length at the latest check.

    Raises:
        AgentStall: The text shows a stall.
    """
    if tool_call_count > 0 and (len(stream_content) - last_stall_check_len > 100):
        stall_reason = detect_phase2_agent_stall(stream_content)
        if stall_reason:
            raise AgentStall(stall_reason)
        return len(stream_content)
    return last_stall_check_len


def _check_tool_call(
    event: ToolCall,
    tool_call_count: int,
    tool_call_signatures: dict[str, int],
    *,
    max_tool_calls: int | None,
    max_repeats: int,
) -> None:
    """Count the call's signature and stop the run at the call cap or a repeated call.

    Args:
        event: The tool call.
        tool_call_count: The tool calls so far, this one included.
        tool_call_signatures: Calls per tool-and-arguments signature; updated here.
        max_tool_calls: The call cap; None for none.
        max_repeats: How many identical calls stop the run.

    Raises:
        AgentStall: The cap is passed or the call was repeated ``max_repeats`` times.
    """
    tool_name = getattr(event, "tool_name", "unknown_tool")
    if max_tool_calls is not None and tool_call_count > max_tool_calls:
        raise AgentStall(f"tool-call limit reached ({max_tool_calls})")
    tool_signature = f"{tool_name}:{format_phase2_tool_args(event)}"
    tool_call_signatures[tool_signature] = tool_call_signatures.get(tool_signature, 0) + 1

    if tool_call_signatures[tool_signature] >= max_repeats:
        raise AgentStall(f"repeated identical tool call: {tool_name}")


def _emit_tool_result(
    event: ToolCallResult, tool_result_count: int, emit_stream: Callable[[str], None]
) -> None:
    """Stream a tool's result block, and a warning when it reports a missing table file."""
    emit_stream(format_phase2_tool_result(event, tool_result_count))
    tool_output = getattr(event, "tool_output", None)
    output = format_phase2_tool_output(tool_output).lower()
    if "missing in active dataset" in output:
        emit_stream(
            "\n⚠️ **File not found** – the requested table "
            "is not in the active dataset. "
            "The agent will try an alternative.\n"
        )
