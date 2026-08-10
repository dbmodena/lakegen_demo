"""Tool-free dataset selection over orchestrator-prepared context."""

from __future__ import annotations

import io

from llama_index.core import Settings
from llama_index.core.llms import LLM

from lakegen.agents.agent_runner import run_agent_workflow
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.core.types import StreamCallback
from lakegen.experiment_config import DiscoveryArchitecture
from lakegen.orchestrated_context import PreparedDiscoveryContext
from lakegen.phases.utils import parse_table_selector_response


def select_from_prepared_context(
    *,
    query: str,
    llm: LLM,
    context: PreparedDiscoveryContext,
    all_files: list[str],
    architecture: DiscoveryArchitecture,
    hint: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check=None,
) -> tuple[list[str], str, str, int]:
    """Ask an agent with an empty tool set to select only supplied candidates."""

    context_json = context.stable_json()
    system_prompt = (
        "You are a dataset selection agent. You have no tools and must reason only "
        "from the prepared JSON context supplied by the orchestrator. Never search, "
        "inspect files, or invent metadata. Return exactly FINAL_PAYLOAD: followed by "
        "a JSON object with string fields 'tables' (comma-separated exact dataset "
        "names) and 'reasoning'. Select no dataset outside the context. "
        f"Discovery architecture: {architecture.value}."
    )
    user_prompt = (
        f"Prepared discovery context:\n{context_json}\n"
        + (f"Previous-attempt constraint: {hint}\n" if hint else "")
        + "Choose the minimal set of datasets needed to answer the question."
    )
    token_counter = next(
        (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
        None,
    )
    if token_counter:
        token_counter.reset_counts()
    reset_llm_token_usage(llm)
    stream = io.StringIO()

    def emit(delta: str) -> None:
        stream.write(delta or "")
        if stream_callback is not None and delta:
            stream_callback(delta)

    response = run_agent_workflow(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        agent_name=f"{architecture.value}_context_selector",
        emit_stream=emit,
        cancel_check=cancel_check,
        tools=[],
        max_iterations=1,
        max_repeats=1,
    )
    trace = "--- Orchestrated Context Discovery ---\n" + stream.getvalue()
    stream.close()
    candidate_names = [item.dataset for item in context.candidates]
    selected, reasoning = parse_table_selector_response(
        response, all_files, candidate_names
    )
    tokens = 0
    if token_counter:
        tokens = (
            token_counter.prompt_llm_token_count
            + token_counter.completion_llm_token_count
        )
        token_counter.reset_counts()
    return selected, reasoning, trace, max(tokens, get_llm_token_usage(llm))
