"""Typed, tool-free discovery paths using orchestrator-prepared retrieval context."""

from __future__ import annotations

import io
import json
import re
from typing import Any

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole, LLM
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lakegen.agents.agent_runner import run_agent_workflow
from lakegen.core.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.core.types import SolrMetadata, StreamCallback
from lakegen.experiment_config import DiscoveryArchitecture
from lakegen.orchestrated_context import (
    PreparedDiscoveryContext,
    prepare_discovery_context,
)
from lakegen.retrieval import RetrievalConfig
from lakegen.ui.state import WorkflowCancelled


class RetrievalRequestProtocolError(ValueError):
    """The first tool-free turn did not produce a valid retrieval request."""


class OrchestratedContextPreparationError(RuntimeError):
    """The configured retriever or context construction failed."""


class OrchestratedSelectorError(RuntimeError):
    """The tool-free selector invocation or response protocol failed."""


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    concepts: list[str] = Field(min_length=1, max_length=2)

    @field_validator("concepts", mode="before")
    @classmethod
    def normalize_concepts(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("concepts must be a list")
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("each concept must be a string")
            concept = " ".join(item.split())
            if not concept:
                continue
            if len(concept) > 100:
                raise ValueError("concepts must be at most 100 characters")
            identity = concept.casefold()
            if identity not in seen:
                seen.add(identity)
                normalized.append(concept)
        if not normalized:
            raise ValueError("at least one non-empty concept is required")
        return normalized

    @property
    def keywords(self) -> list[str]:
        """Internal compatibility name used by existing retriever calls."""
        return list(self.concepts)


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
    match = re.fullmatch(r"\s*RETRIEVAL_REQUEST:\s*(\{.*\})\s*", response, re.DOTALL)
    if match is None:
        raise RetrievalRequestProtocolError("invalid RETRIEVAL_REQUEST envelope")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RetrievalRequestProtocolError(f"invalid RETRIEVAL_REQUEST JSON: {exc}") from exc
    try:
        return RetrievalRequest.model_validate(payload)
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
    match = re.fullmatch(r"\s*FINAL_PAYLOAD:\s*(\{.*\})\s*", response, re.DOTALL)
    if match is None:
        raise OrchestratedSelectorError("invalid FINAL_PAYLOAD envelope")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise OrchestratedSelectorError(f"invalid FINAL_PAYLOAD JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"tables", "reasoning"}:
        raise OrchestratedSelectorError(
            "FINAL_PAYLOAD requires exactly string fields 'tables' and 'reasoning'"
        )
    if not isinstance(payload["tables"], str) or not isinstance(payload["reasoning"], str):
        raise OrchestratedSelectorError("FINAL_PAYLOAD fields must be strings")
    allowed = set(candidates)
    selected = []
    for name in (item.strip() for item in payload["tables"].split(",")):
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
        if stream_callback is not None and delta:
            stream_callback(delta)

    response = run_agent_workflow(
        llm=llm, system_prompt=system_prompt, user_prompt=user_prompt,
        agent_name=agent_name, emit_stream=emit, cancel_check=cancel_check,
        tools=[], chat_history=chat_history, max_iterations=1, max_repeats=1,
    )
    tokens = 0
    if token_counter:
        tokens = token_counter.prompt_llm_token_count + token_counter.completion_llm_token_count
        token_counter.reset_counts()
    trace = stream.getvalue()
    stream.close()
    return response, trace, max(tokens, get_llm_token_usage(llm))


def _selector_prompts(context: PreparedDiscoveryContext, hint: str) -> tuple[str, str]:
    system = (
        "You are a dataset selection agent with no tools. Reason only from the "
        "orchestrator context. Return exactly FINAL_PAYLOAD: followed by JSON with "
        "string fields 'tables' and 'reasoning'. Never invent datasets or metadata."
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
    selected, reasoning = parse_orchestrated_selection(response, candidates)
    return selected, reasoning, "--- Divided Orchestrated Selector ---\n" + stream, tokens


def run_unified_orchestrated_discovery(
    *, query: str, llm: LLM, solr_client, all_files: list[str],
    retrieval_config: RetrievalConfig, hint: str = "",
    stream_callback: StreamCallback | None = None, cancel_check=None,
    table_dir=None,
) -> DiscoveryResult:
    """Run two turns of one logical tool-free agent with explicit chat history."""
    system = (
        "You are the unified LakeGen discovery agent. You have no tools. First request "
        "retrieval using exactly RETRIEVAL_REQUEST: {\"concepts\":[...]}, containing "
        "one or two concise dataset concepts. Do not choose system parameters. After "
        "context arrives, return exactly "
        "FINAL_PAYLOAD: {\"tables\":\"comma-separated exact names\",\"reasoning\":\"...\"}."
    )
    first_user = f"Question: {query}\nFormulate the metadata retrieval request."
    if hint:
        first_user += f"\nPrevious-attempt constraint: {hint}"
    try:
        request_text, first_trace, first_tokens = _run_tool_free_turn(
            llm=llm, system_prompt=system, user_prompt=first_user,
            agent_name="unified_orchestrated_discovery", stream_callback=stream_callback,
            cancel_check=cancel_check,
        )
        request = parse_retrieval_request(request_text)
    except WorkflowCancelled:
        raise
    except Exception as exc:
        if isinstance(exc, RetrievalRequestProtocolError):
            raise
        raise RetrievalRequestProtocolError(str(exc)) from exc
    try:
        prepared, metadata = prepare_discovery_context(
            query=query, keywords=request.keywords, solr_client=solr_client,
            all_files=all_files, retrieval_config=retrieval_config,
            table_dir=table_dir,
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
            selected_datasets=[], candidates=[], keywords=request.keywords,
            metadata=metadata, reasoning=reason,
            trace="--- Unified Orchestrated Turn 1 ---\n" + first_trace,
            tokens=first_tokens, llm_invocations=1, agent_count=1,
            retry_keywords=True, retry_reason=reason, prepared_context=prepared,
        )
    second_user = "Orchestrator-prepared context:\n" + prepared.agent_json()
    history = [
        ChatMessage(role=MessageRole.USER, content=first_user),
        ChatMessage(role=MessageRole.ASSISTANT, content=request_text),
    ]
    try:
        final_text, second_trace, second_tokens = _run_tool_free_turn(
            llm=llm, system_prompt=system, user_prompt=second_user,
            agent_name="unified_orchestrated_discovery", stream_callback=stream_callback,
            cancel_check=cancel_check, chat_history=history,
        )
        selected, reasoning = parse_orchestrated_selection(final_text, candidates)
    except WorkflowCancelled:
        raise
    except Exception as exc:
        if isinstance(exc, OrchestratedSelectorError):
            raise
        raise OrchestratedSelectorError(str(exc)) from exc
    retry_reason = selector_retry_reason(selected, reasoning)
    return DiscoveryResult(
        selected_datasets=selected, candidates=candidates, keywords=request.keywords,
        metadata=metadata,
        trace=("--- Unified Orchestrated Turn 1 ---\n" + first_trace
               + "\n--- Unified Orchestrated Turn 2 ---\n" + second_trace),
        tokens=first_tokens + second_tokens, llm_invocations=2, agent_count=1,
        retry_keywords=retry_reason is not None, retry_reason=retry_reason,
        reasoning=retry_reason or reasoning, prepared_context=prepared,
    )
