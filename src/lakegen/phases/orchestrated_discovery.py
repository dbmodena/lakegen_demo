"""Typed, tool-free discovery paths using orchestrator-prepared retrieval context."""

from __future__ import annotations

import io
import json
import re
from typing import Any

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole, LLM
from pydantic import BaseModel, ConfigDict, Field

from lakegen.agents.agent_runner import run_agent_workflow
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.core.types import SolrMetadata, StreamCallback
from lakegen.experiment_config import DiscoveryArchitecture
from lakegen.orchestrated_context import (
    PreparedDiscoveryContext,
    prepare_discovery_context,
)
from lakegen.retrieval import RetrievalConfig
from lakegen.retrieval.intent import RetrievalIntent, parse_retrieval_intent
from prompts.prompt_manager import PromptManager
from lakegen.ui.state import WorkflowCancelled


class RetrievalRequestProtocolError(ValueError):
    """The first tool-free turn did not produce a valid retrieval request."""


class OrchestratedContextPreparationError(RuntimeError):
    """The configured retriever or context construction failed."""


class OrchestratedSelectorError(RuntimeError):
    """The tool-free selector invocation or response protocol failed."""


RetrievalRequest = RetrievalIntent


class DiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    selected_datasets: list[str]
    candidates: list[str]
    keywords: list[str]
    metadata: SolrMetadata
    reasoning: str
    trace: str
    tokens: int = Field(ge=0)
    llm_invocations: int = Field(ge=0)
    agent_count: int = Field(ge=1)
    retry_keywords: bool = False
    retry_reason: str | None = None
    prepared_context: PreparedDiscoveryContext | None = None


def parse_retrieval_request(response: str) -> RetrievalRequest:
    try:
        return parse_retrieval_intent(response)
    except ValueError as exc:
        raise RetrievalRequestProtocolError(str(exc)) from exc


def parse_orchestrated_selection(
    response: str, candidates: list[str]
) -> tuple[list[str], str]:
    if response.strip().startswith("REJECT_KEYWORDS"):
        reason = response.strip()
        if not reason.startswith("REJECT_KEYWORDS:"):
            reason = "REJECT_KEYWORDS: " + reason.removeprefix("REJECT_KEYWORDS").strip()
        return [], reason
    match = re.search(r"\bFINAL_PAYLOAD:\s*", response)
    if match is None:
        raise OrchestratedSelectorError("invalid FINAL_PAYLOAD envelope")
    try:
        payload, end = json.JSONDecoder().raw_decode(response, match.end())
        trailing = response[end:].strip()
        if trailing and trailing not in {"```", "`"}:
            raise ValueError("unexpected content after FINAL_PAYLOAD object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise OrchestratedSelectorError(f"invalid FINAL_PAYLOAD JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"tables", "reasoning"}:
        raise OrchestratedSelectorError(
            "FINAL_PAYLOAD requires exactly fields 'tables' and 'reasoning'"
        )
    tables_value = payload["tables"]
    if isinstance(tables_value, str):
        table_names = [item.strip() for item in tables_value.split(",")]
    elif isinstance(tables_value, list):
        table_names = []
        for item in tables_value:
            if isinstance(item, str):
                table_names.append(item.strip())
                continue
            if isinstance(item, dict):
                names = [item.get(key) for key in ("dataset", "table", "name") if item.get(key)]
                if len(names) == 1 and isinstance(names[0], str):
                    table_names.append(names[0].strip())
                    continue
            raise OrchestratedSelectorError(
                "FINAL_PAYLOAD 'tables' entries must be dataset strings or named objects"
            )
    else:
        raise OrchestratedSelectorError(
            "FINAL_PAYLOAD 'tables' must be a string or an array of strings"
        )
    if not isinstance(payload["reasoning"], str):
        raise OrchestratedSelectorError("FINAL_PAYLOAD 'reasoning' must be a string")
    allowed = set(candidates)
    selected = []
    for name in table_names:
        if name and name in allowed and name not in selected:
            selected.append(name)
    return selected, payload["reasoning"].strip()


def selector_retry_reason(selected: list[str], reasoning: str) -> str | None:
    if selected and not reasoning.startswith("REJECT_KEYWORDS"):
        return None
    if reasoning.startswith("REJECT_KEYWORDS"):
        return reasoning
    return "REJECT_KEYWORDS: The orchestrated selector returned no valid datasets"


def _run_tool_free_turn(
    *, llm: LLM, system_prompt: str, user_prompt: str, agent_name: str,
    stream_callback: StreamCallback | None = None, cancel_check=None,
    chat_history: list[ChatMessage] | None = None,
) -> tuple[str, str, int]:
    token_counter = next(
        (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")), None
    )
    if token_counter:
        token_counter.reset_counts()
    reset_llm_token_usage(llm)
    stream = io.StringIO()

    def emit(delta: str) -> None:
        stream.write(delta or "")
        if not delta:
            return
        # Every phase owns the terminal; stream_callback is the extra UI channel.
        print(delta, end="", flush=True)
        if stream_callback is not None:
            stream_callback(delta)

    response = run_agent_workflow(
        llm=llm, system_prompt=system_prompt, user_prompt=user_prompt,
        agent_name=agent_name, emit_stream=emit, cancel_check=cancel_check,
        # FunctionAgent counts the final response hand-off as another workflow
        # iteration even when no tools are available.  A limit of one lets the
        # model emit its answer but raises "Max iterations of 1 reached" before
        # the handler can return it.
        tools=[], chat_history=chat_history, max_iterations=2, max_repeats=1,
    )
    tokens = 0
    if token_counter:
        tokens = token_counter.prompt_llm_token_count + token_counter.completion_llm_token_count
        token_counter.reset_counts()
    trace = stream.getvalue()
    stream.close()
    # AgentStream is the model's verbatim answer.  Some FunctionAgent versions
    # return an empty/wrapper final response for tool-free runs even though the
    # complete protocol payload was streamed successfully.
    protocol_response = trace.strip() or response
    return protocol_response, trace, max(tokens, get_llm_token_usage(llm))


def _selector_prompts(context: PreparedDiscoveryContext, hint: str) -> tuple[str, str]:
    system = (
        "You are a dataset selection agent with no tools. Reason only from the "
        "orchestrator context. Return exactly FINAL_PAYLOAD: followed by JSON with "
        "a 'tables' array of dataset-name strings and a string field 'reasoning'. "
        "Never invent datasets or metadata."
    )
    user = f"Prepared discovery context:\n{context.agent_json()}\n"
    if hint:
        user += f"Previous-attempt constraint: {hint}\n"
    return system, user + "Select the minimal sufficient dataset set."


def select_from_prepared_context(
    *, query: str, llm: LLM, context: PreparedDiscoveryContext,
    all_files: list[str], architecture: DiscoveryArchitecture, hint: str = "",
    stream_callback: StreamCallback | None = None, cancel_check=None,
) -> tuple[list[str], str, str, int]:
    """Divided architecture's second, distinct tool-free agent."""
    system, user = _selector_prompts(context, hint)
    try:
        response, stream, tokens = _run_tool_free_turn(
            llm=llm, system_prompt=system, user_prompt=user,
            agent_name="divided_context_selector", stream_callback=stream_callback,
            cancel_check=cancel_check,
        )
    except WorkflowCancelled:
        raise
    except Exception as exc:
        if isinstance(exc, OrchestratedSelectorError):
            raise
        raise OrchestratedSelectorError(str(exc)) from exc
    candidates = [item.dataset for item in context.candidates]
    try:
        selected, reasoning = parse_orchestrated_selection(response, candidates)
    except OrchestratedSelectorError as first_error:
        correction = (
            "Your previous selection payload was rejected by the protocol: "
            f"{first_error}. Return only FINAL_PAYLOAD JSON with a 'tables' array "
            "of exact dataset names and a string 'reasoning'. Previous response:\n"
            + response
        )
        try:
            response, correction_stream, correction_tokens = _run_tool_free_turn(
                llm=llm, system_prompt=system, user_prompt=correction,
                agent_name="divided_context_selector",
                stream_callback=stream_callback, cancel_check=cancel_check,
            )
            selected, reasoning = parse_orchestrated_selection(response, candidates)
        except WorkflowCancelled:
            raise
        except Exception as exc:
            if isinstance(exc, OrchestratedSelectorError):
                raise
            raise OrchestratedSelectorError(str(exc)) from exc
        stream += "\n--- Protocol correction ---\n" + correction_stream
        tokens += correction_tokens
    return selected, reasoning, "--- Divided Orchestrated Selector ---\n" + stream, tokens


def run_unified_orchestrated_discovery(
    *, query: str, llm: LLM, solr_client, all_files: list[str],
    retrieval_config: RetrievalConfig, hint: str = "",
    stream_callback: StreamCallback | None = None, cancel_check=None,
    table_dir=None, portal_name: str = "",
) -> DiscoveryResult:
    """Run two turns of one logical tool-free agent with explicit chat history."""
    pm = PromptManager()
    value_search = retrieval_config.mode.value_keywords
    system = pm.render(
        "retrieval_intent", "system_prompt", value_search=value_search,
        verbatim_entities=retrieval_config.mode.verbatim_entities,
    )
    first_user = pm.render(
        "retrieval_intent", "user_prompt", question=query,
        catalog=portal_name, schema="not supplied", hint=hint,
    )
    request_invocations = 1
    try:
        request_text, first_trace, first_tokens = _run_tool_free_turn(
            llm=llm, system_prompt=system, user_prompt=first_user,
            agent_name="unified_orchestrated_discovery", stream_callback=stream_callback,
            cancel_check=cancel_check,
        )
        try:
            request = parse_retrieval_request(request_text)
        except RetrievalRequestProtocolError as first_error:
            correction = (
                "Your previous retrieval intent was rejected by the protocol: "
                f"{first_error}. Return only one corrected RETRIEVAL_INTENT JSON "
                "object matching the system schema. Previous response:\n"
                + request_text
            )
            corrected_text, corrected_trace, corrected_tokens = _run_tool_free_turn(
                llm=llm, system_prompt=system, user_prompt=correction,
                agent_name="unified_orchestrated_discovery",
                stream_callback=stream_callback, cancel_check=cancel_check,
            )
            request = parse_retrieval_request(corrected_text)
            request_text = corrected_text
            first_trace += "\n--- Protocol correction ---\n" + corrected_trace
            first_tokens += corrected_tokens
            request_invocations = 2
    except WorkflowCancelled:
        raise
    except Exception as exc:
        if isinstance(exc, RetrievalRequestProtocolError):
            raise
        raise RetrievalRequestProtocolError(str(exc)) from exc
    if request.status == "unresolved":
        reason = "UNRESOLVED_RETRIEVAL_INTENT: " + "; ".join(request.missing_evidence)
        return DiscoveryResult(
            selected_datasets=[], candidates=[], keywords=[], metadata={},
            reasoning=reason, trace="--- Unified Orchestrated Turn 1 ---\n" + first_trace,
            tokens=first_tokens, llm_invocations=request_invocations, agent_count=1,
            retry_keywords=False, retry_reason=reason,
        )
    keywords = request.search_terms(value_search)
    if not keywords:
        # A cell-value search has nothing to look for unless values were listed;
        # falling back to concepts would search dataset topics against cells.
        raise RetrievalRequestProtocolError(
            "resolved retrieval intent lists no search_values"
        )
    try:
        prepared, metadata = prepare_discovery_context(
            query=query, keywords=keywords, solr_client=solr_client,
            all_files=all_files, retrieval_config=retrieval_config,
            table_dir=table_dir, entities=request.entities,
        )
        prepared.agent_json()
    except WorkflowCancelled:
        raise
    except Exception as exc:
        raise OrchestratedContextPreparationError(str(exc)) from exc
    candidates = [item.dataset for item in prepared.candidates]
    if not candidates:
        reason = "REJECT_KEYWORDS: No datasets found in the prepared context"
        return DiscoveryResult(
            selected_datasets=[], candidates=[], keywords=keywords,
            metadata=metadata, reasoning=reason,
            trace="--- Unified Orchestrated Turn 1 ---\n" + first_trace,
            tokens=first_tokens, llm_invocations=request_invocations, agent_count=1,
            retry_keywords=True, retry_reason=reason, prepared_context=prepared,
        )
    selection_system, second_user = _selector_prompts(prepared, "")
    history = [
        ChatMessage(role=MessageRole.USER, content=first_user),
        ChatMessage(role=MessageRole.ASSISTANT, content=request_text),
    ]
    selector_invocations = 1
    try:
        final_text, second_trace, second_tokens = _run_tool_free_turn(
            llm=llm, system_prompt=selection_system, user_prompt=second_user,
            agent_name="unified_orchestrated_discovery", stream_callback=stream_callback,
            cancel_check=cancel_check, chat_history=history,
        )
        try:
            selected, reasoning = parse_orchestrated_selection(final_text, candidates)
        except OrchestratedSelectorError as first_error:
            correction = (
                "Your previous selection payload was rejected by the protocol: "
                f"{first_error}. Return only FINAL_PAYLOAD JSON with a 'tables' "
                "array containing exact dataset names from the supplied context and "
                "a string 'reasoning'. Previous response:\n" + final_text
            )
            corrected_text, corrected_trace, corrected_tokens = _run_tool_free_turn(
                llm=llm, system_prompt=selection_system, user_prompt=correction,
                agent_name="unified_orchestrated_discovery",
                stream_callback=stream_callback, cancel_check=cancel_check,
                chat_history=history,
            )
            selected, reasoning = parse_orchestrated_selection(
                corrected_text, candidates
            )
            final_text = corrected_text
            second_trace += "\n--- Protocol correction ---\n" + corrected_trace
            second_tokens += corrected_tokens
            selector_invocations = 2
    except WorkflowCancelled:
        raise
    except Exception as exc:
        if isinstance(exc, OrchestratedSelectorError):
            raise
        raise OrchestratedSelectorError(str(exc)) from exc
    retry_reason = selector_retry_reason(selected, reasoning)
    return DiscoveryResult(
        selected_datasets=selected, candidates=candidates, keywords=keywords,
        metadata=metadata,
        trace=("--- Unified Orchestrated Turn 1 ---\n" + first_trace
               + "\n--- Unified Orchestrated Turn 2 ---\n" + second_trace),
        tokens=first_tokens + second_tokens,
        llm_invocations=request_invocations + selector_invocations, agent_count=1,
        retry_keywords=retry_reason is not None, retry_reason=retry_reason,
        reasoning=retry_reason or reasoning, prepared_context=prepared,
    )
