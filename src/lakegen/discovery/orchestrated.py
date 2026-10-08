"""Typed, tool-free discovery paths using orchestrator-prepared retrieval context."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, MessageRole, LLM
from pydantic import BaseModel, ConfigDict, Field

from lakegen.agents.agent_runner import run_agent_workflow
from lakegen.llm.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.core.types import SolrMetadata, StreamCallback
from lakegen.discovery.selectors import SelectionRequest, SelectorUnavailable
from lakegen.discovery.selectors import choose as choose_selector
from lakegen.experiment_config import DiscoveryArchitecture, DiscoveryConfig
from lakegen.keyword_terms import keyword_terms
from lakegen.orchestrated_context import (
    PreparedDiscoveryContext,
    prepare_discovery_context,
)
from lakegen.retrieval import RetrievalConfig
from lakegen.retrieval.intent import RetrievalIntent, parse_retrieval_intent
from lakegen.prompts.prompt_manager import PromptManager
from lakegen.errors import WorkflowCancelled


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
    failed_keyword_combinations: list[list[str]] = Field(default_factory=list)
    retrieval_memory_events: list[dict[str, object]] = Field(default_factory=list)


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
    discovery: DiscoveryConfig | None = None, table_dir: Path | None = None,
) -> tuple[list[str], str, str, int]:
    """Divided architecture's second, distinct tool-free agent."""
    jev_fallback_note = ""
    selector = choose_selector(discovery)
    if selector is not None:
        try:
            selection = selector.select(SelectionRequest(
                query,
                [item.dataset for item in context.candidates],
                {item.dataset: item.metadata for item in context.candidates},
                table_dir,
                discovery,
            ))
        except SelectorUnavailable as exc:
            jev_fallback_note = f"[{selector.kind.value} selector unavailable, using the LLM selector] {exc}\n"
            print(jev_fallback_note, flush=True)
        else:
            print(selection.trace, flush=True)
            if stream_callback is not None:
                stream_callback(f"\n**{selector.title}**\n{selection.reasoning}\n")
            return selection.selected, selection.reasoning, selection.trace, selection.tokens
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
    return selected, reasoning, jev_fallback_note + "--- Divided Orchestrated Selector ---\n" + stream, tokens


TURN_1_HEADER = "--- Unified Orchestrated Turn 1 ---\n"
TURN_2_HEADER = "\n--- Unified Orchestrated Turn 2 ---\n"
AGENT_NAME = "unified_orchestrated_discovery"


class _UnifiedOrchestratedRun:
    """Two turns of one tool-free agent: the retrieval intent, then the selection from the prepared context."""

    def __init__(
        self, *, query: str, llm: LLM, solr_client, all_files: list[str], retrieval_config: RetrievalConfig,
        hint: str, stream_callback: StreamCallback | None, cancel_check, table_dir, portal_name: str,
        retrieval_memory: str, failed_keyword_combinations: list[frozenset[str]] | None,
        max_zero_result_retries: int, memory_enabled: bool,
    ) -> None:
        self.query, self.llm, self.solr_client, self.all_files = query, llm, solr_client, all_files
        self.retrieval_config, self.stream_callback, self.cancel_check = retrieval_config, stream_callback, cancel_check
        self.table_dir, self.max_zero_result_retries, self.memory_enabled = table_dir, max_zero_result_retries, memory_enabled
        self.value_search = retrieval_config.mode.value_keywords
        self.lexical_mode = retrieval_config.mode.value in {"keyword", "hybrid"}
        pm = PromptManager()
        self.system = pm.render(
            "retrieval_intent", "system_prompt", value_search=self.value_search,
            verbatim_entities=retrieval_config.mode.verbatim_entities,
        )
        self.first_user = pm.render(
            "retrieval_intent", "user_prompt", question=query,
            catalog=portal_name, schema="not supplied", hint=hint,
            retrieval_memory=retrieval_memory,
        )
        self.hard_bans = [frozenset(item) for item in (failed_keyword_combinations or []) if item]
        self.memory_events: list[dict[str, object]] = []
        # What the first agent said and spent, corrections included.
        self.request_text = ""
        self.first_trace = ""
        self.first_tokens = 0
        self.request_invocations = 0

    def _turn(self, user_prompt: str, *, system: str | None = None, chat_history=None) -> tuple[str, str, int]:
        return _run_tool_free_turn(
            llm=self.llm, system_prompt=self.system if system is None else system, user_prompt=user_prompt,
            agent_name=AGENT_NAME, stream_callback=self.stream_callback, cancel_check=self.cancel_check,
            **({"chat_history": chat_history} if chat_history is not None else {}),
        )

    def _first_turn(self, user_prompt: str, correction_header: str | None = None) -> RetrievalRequest:
        """One turn of the first agent, folded into its trace and tokens."""
        text, trace, tokens = self._turn(user_prompt)
        self.first_trace = (self.first_trace + correction_header + trace) if correction_header else trace
        self.first_tokens += tokens
        self.request_invocations += 1
        request = parse_retrieval_request(text)
        self.request_text = text
        return request

    def intent(self) -> RetrievalRequest:
        """The retrieval intent, corrected once when it breaks the protocol."""
        try:
            try:
                return self._first_turn(self.first_user)
            except RetrievalRequestProtocolError as first_error:
                correction = (
                    "Your previous retrieval intent was rejected by the protocol: "
                    f"{first_error}. Return only one corrected RETRIEVAL_INTENT JSON "
                    "object matching the system schema. Previous response:\n"
                    + self.request_text
                )
                return self._first_turn(correction, "\n--- Protocol correction ---\n")
        except WorkflowCancelled:
            raise
        except Exception as exc:
            if isinstance(exc, RetrievalRequestProtocolError):
                raise
            raise RetrievalRequestProtocolError(str(exc)) from exc

    def _remember_failed_keywords(self, values: list[str]) -> None:
        failed = keyword_terms(values)
        if not failed or any(known <= failed for known in self.hard_bans):
            return
        self.hard_bans[:] = [known for known in self.hard_bans if not failed < known]
        self.hard_bans.append(failed)
        self.hard_bans.sort(key=lambda item: (len(item), sorted(item)))

    def _banned_subset(self, values: list[str]) -> frozenset[str] | None:
        proposed = keyword_terms(values)
        matches = [known for known in self.hard_bans if known <= proposed]
        return min(matches, key=lambda item: (len(item), sorted(item)), default=None)

    def _corrected_request(self, constraint: str) -> RetrievalRequest:
        return self._first_turn(
            self.first_user + "\nOrchestrator feedback: " + constraint, "\n--- Orchestrator retrieval correction ---\n",
        )

    def prepare(self, request: RetrievalRequest) -> tuple[list[str], PreparedDiscoveryContext, SolrMetadata]:
        """The prepared context of the intent's keywords, after the corrections a known or a new zero result earns."""
        zero_retries = 0
        blocked_corrections = 0
        while True:
            keywords = request.search_terms(self.value_search)
            if not keywords:
                # A cell-value search has nothing to look for unless values were listed;
                # falling back to concepts would search dataset topics against cells.
                raise RetrievalRequestProtocolError("resolved retrieval intent lists no search_values")
            blocked = self._banned_subset(keywords) if self.lexical_mode and self.memory_enabled else None
            if blocked is not None:
                blocked_corrections += 1
                if blocked_corrections > 3:
                    raise RetrievalRequestProtocolError(
                        "retrieval intent repeatedly contains a known zero-result subset"
                    )
                request = self._corrected_request(
                    "This strict-AND query contains the known zero-result subset "
                    f"{{{', '.join(sorted(blocked))}}}. Return a different intent that excludes it."
                )
                continue
            try:
                prepared, metadata = prepare_discovery_context(
                    query=self.query, keywords=keywords, solr_client=self.solr_client,
                    all_files=self.all_files, retrieval_config=self.retrieval_config,
                    table_dir=self.table_dir, entities=request.entities,
                )
                prepared.agent_json()
            except WorkflowCancelled:
                raise
            except Exception as exc:
                raise OrchestratedContextPreparationError(str(exc)) from exc
            if prepared.candidates or not self.lexical_mode:
                return keywords, prepared, metadata
            if self.memory_enabled:
                self._remember_failed_keywords(keywords)
                self.memory_events.append({"outcome": "zero_results", "terms": list(keywords)})
            if zero_retries >= self.max_zero_result_retries:
                return keywords, prepared, metadata
            zero_retries += 1
            request = self._corrected_request(
                "The previous strict-AND retrieval returned zero local candidates. "
                "Return a genuinely different intent; do not reuse its terms."
            )

    def select(self, prepared: PreparedDiscoveryContext, candidates: list[str]) -> tuple[list[str], str, str, int, int]:
        """The second turn over the prepared context: the selection, its reasoning, trace, tokens and calls."""
        selection_system, second_user = _selector_prompts(prepared, "")
        history = [
            ChatMessage(role=MessageRole.USER, content=self.first_user),
            ChatMessage(role=MessageRole.ASSISTANT, content=self.request_text),
        ]
        invocations = 1
        try:
            final_text, trace, tokens = self._turn(second_user, system=selection_system, chat_history=history)
            try:
                selected, reasoning = parse_orchestrated_selection(final_text, candidates)
            except OrchestratedSelectorError as first_error:
                correction = (
                    "Your previous selection payload was rejected by the protocol: "
                    f"{first_error}. Return only FINAL_PAYLOAD JSON with a 'tables' "
                    "array containing exact dataset names from the supplied context and "
                    "a string 'reasoning'. Previous response:\n" + final_text
                )
                corrected_text, corrected_trace, corrected_tokens = self._turn(
                    correction, system=selection_system, chat_history=history,
                )
                selected, reasoning = parse_orchestrated_selection(corrected_text, candidates)
                trace += "\n--- Protocol correction ---\n" + corrected_trace
                tokens += corrected_tokens
                invocations = 2
        except WorkflowCancelled:
            raise
        except Exception as exc:
            if isinstance(exc, OrchestratedSelectorError):
                raise
            raise OrchestratedSelectorError(str(exc)) from exc
        return selected, reasoning, trace, tokens, invocations

    def _first_turn_result(
        self, reason: str, *, keywords: list[str] | None = None, metadata: SolrMetadata | None = None, **fields,
    ) -> DiscoveryResult:
        """A result that ends after the first turn, with the given reason."""
        return DiscoveryResult(
            selected_datasets=[], candidates=[], keywords=list(keywords or []), metadata=metadata or {},
            reasoning=reason, trace=TURN_1_HEADER + self.first_trace,
            tokens=self.first_tokens, llm_invocations=self.request_invocations, agent_count=1,
            retry_reason=reason, **fields,
        )

    def run(self) -> DiscoveryResult:
        request = self.intent()
        if request.status == "unresolved":
            reason = "UNRESOLVED_RETRIEVAL_INTENT: " + "; ".join(request.missing_evidence)
            return self._first_turn_result(reason, retry_keywords=False)
        keywords, prepared, metadata = self.prepare(request)
        candidates = [item.dataset for item in prepared.candidates]
        if not candidates:
            return self._first_turn_result(
                "REJECT_KEYWORDS: No datasets found in the prepared context", retry_keywords=True,
                keywords=keywords, metadata=metadata, prepared_context=prepared,
                failed_keyword_combinations=[sorted(item) for item in self.hard_bans],
                retrieval_memory_events=self.memory_events,
            )
        selected, reasoning, second_trace, second_tokens, selector_invocations = self.select(prepared, candidates)
        retry_reason = selector_retry_reason(selected, reasoning)
        if retry_reason is not None and self.memory_enabled:
            self.memory_events.append({
                "outcome": "insufficient_coverage",
                "terms": list(keywords),
                "tables": list(candidates),
                "reason": retry_reason,
            })
        return DiscoveryResult(
            selected_datasets=selected, candidates=candidates, keywords=keywords,
            metadata=metadata,
            trace=TURN_1_HEADER + self.first_trace + TURN_2_HEADER + second_trace,
            tokens=self.first_tokens + second_tokens,
            llm_invocations=self.request_invocations + selector_invocations, agent_count=1,
            retry_keywords=retry_reason is not None, retry_reason=retry_reason,
            reasoning=retry_reason or reasoning, prepared_context=prepared,
            failed_keyword_combinations=[sorted(item) for item in self.hard_bans],
            retrieval_memory_events=self.memory_events,
        )


def run_unified_orchestrated_discovery(
    *, query: str, llm: LLM, solr_client, all_files: list[str],
    retrieval_config: RetrievalConfig, hint: str = "",
    stream_callback: StreamCallback | None = None, cancel_check=None,
    table_dir=None, portal_name: str = "",
    retrieval_memory: str = "",
    failed_keyword_combinations: list[frozenset[str]] | None = None,
    max_zero_result_retries: int = 0,
    memory_enabled: bool = False,
) -> DiscoveryResult:
    """Run two turns of one logical tool-free agent with explicit chat history."""
    return _UnifiedOrchestratedRun(
        query=query, llm=llm, solr_client=solr_client, all_files=all_files, retrieval_config=retrieval_config,
        hint=hint, stream_callback=stream_callback, cancel_check=cancel_check, table_dir=table_dir,
        portal_name=portal_name, retrieval_memory=retrieval_memory,
        failed_keyword_combinations=failed_keyword_combinations,
        max_zero_result_retries=max_zero_result_retries, memory_enabled=memory_enabled,
    ).run()
