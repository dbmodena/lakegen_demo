import io
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import LLM

from lakegen.agent_tools.requirement_ledger import build_minimal_selection_fallback
from lakegen.agent_tools.tools_p2 import Phase2JudgeToolsManager
from lakegen.config.experiment import DiscoveryConfig
from lakegen.discovery.candidates import (
    format_candidate_context,
    match_local_csv,
    parse_table_selector_response,
    solr_metadata_from_doc,
)
from lakegen.discovery.selectors import SelectionRequest, SelectorUnavailable
from lakegen.discovery.selectors import choose as choose_selector
from lakegen.domain.errors import WorkflowCancelled
from lakegen.domain.types import Phase2SelectionResult, SolrMetadata, StreamCallback
from lakegen.llm.agents.activity import (
    Phase2AgentStall,
)
from lakegen.llm.agents.instrumentation import ThinkingCapture
from lakegen.llm.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.prompts.prompt_manager import PromptManager
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.factory import get_table_retrieval_service
from lakegen.retrieval.solr_client import LocalSolrClient


def _solr_and_search(
    query: str,
    keywords: list[str],
    solr_client: LocalSolrClient,
    all_files: list[str],
    retrieval_config: RetrievalConfig | None = None,
    table_dir: Path | None = None,
    entities: Sequence[str] | None = None,
    carried_tables: Sequence[str] = (),
    carried_metadata: SolrMetadata | None = None,
    excluded_tables: Iterable[str] = (),
) -> tuple[list[str], SolrMetadata]:
    """Execute the configured retriever and return candidates + metadata.

    ``carried_tables`` are candidates a previous, rejected attempt already
    verified as partially satisfying the question; they are re-surfaced here
    regardless of whether this round's retrieval finds them again.
    ``excluded_tables`` were inspected and ruled out by a previous attempt and
    must not be re-proposed.
    """
    candidates: list[str] = []
    metadata: SolrMetadata = {}
    config = retrieval_config or RetrievalConfig()
    excluded = {str(table).casefold() for table in excluded_tables}
    # Retrieval itself truncates to top_k, so asking for exactly top_k would
    # let a skip-listed hit shrink the final pool instead of being replaced by
    # the next-ranked table just outside the previous cutoff.
    fetch_k = config.top_k + len(excluded)

    try:
        retriever = get_table_retrieval_service(
            solr_client, config,
            **({"table_dir": table_dir} if config.mode.requires_table_dir else {}),
        )
        hits = retriever.retrieve(
            question=query,
            keywords=keywords,
            top_k=fetch_k,
            lexical_fetch_k=max(15, fetch_k),
            q_op="AND",
            entities=entities,
        )
        print(
            f"[phase2] {config.mode} retrieval keywords={keywords} "
            f"docs_returned={len(hits)}",
            flush=True,
        )

        for hit in hits:
            doc = hit.document
            matched = match_local_csv(doc, all_files)
            if (
                matched is None
                or matched in candidates
                or matched.casefold() in excluded
            ):
                continue
            candidates.append(matched)
            metadata[matched] = solr_metadata_from_doc(doc)
            metadata[matched]["retrieval"] = hit.to_log_dict()
            if len(candidates) >= config.top_k:
                break

    except Exception as solr_err:
        print(
            f"[phase2] Solr error {type(solr_err).__name__}: {solr_err}",
            flush=True,
        )
        # Dense setup errors (missing model/vector field/provenance) must not be
        # disguised as a failed Phase 1 keyword query.
        if config.mode != RetrievalMode.KEYWORD:
            raise

    for table in carried_tables:
        if table.casefold() in excluded:
            continue
        if table in candidates:
            candidates.remove(table)
        candidates.insert(0, table)
        if carried_metadata and table in carried_metadata:
            metadata[table] = carried_metadata[table]
    if len(candidates) > config.top_k:
        candidates = candidates[: config.top_k]

    return candidates, metadata


TRACE_HEADER = "--- Phase 2 Activity Log ---\n"


def _no_result(config: RetrievalConfig, keywords: list[str], stream_callback: StreamCallback | None) -> Phase2SelectionResult:
    """Retrieval found nothing: the keywords go back to the writer, before any agent runs."""
    message = (
        f"REJECT_KEYWORDS: No tables found with {config.mode} retrieval "
        f"for [{', '.join(keywords)}]. Try broader or different terms."
    )
    trace = (
        f"[phase2] {config.mode} retrieval with keywords={keywords} returned 0 results.\n"
        f"Rejecting keywords back to Phase 1."
    )
    if stream_callback:
        stream_callback(
            f"\n⚠️ **No tables found** with keywords: "
            f"`{' '.join(keywords)}`\n"
            f"Sending feedback to Phase 1 for better keywords.\n"
        )
    return [], [], {}, message, trace, 0


def _configured_selector(
    discovery: DiscoveryConfig | None, query: str, candidates: list[str], solr_meta: SolrMetadata,
    csv_dir: Path, selection_state: object | None, stream_callback: StreamCallback | None,
) -> tuple[Phase2SelectionResult | None, str]:
    """The configured selector's answer, or the note the judge's trace starts with when it could not run."""
    selector = choose_selector(discovery)
    if selector is None:
        return None, ""
    try:
        selection = selector.select(SelectionRequest(query, candidates, solr_meta, csv_dir, discovery))
    except SelectorUnavailable as exc:
        note = f"[{selector.kind.value} selector unavailable, using the LLM judge] {exc}\n"
        print(f"[phase2] {note}", flush=True)
        return None, note
    print(f"[phase2] {selection.trace}", flush=True)
    if stream_callback:
        stream_callback(f"\n**Phase 2 – {selector.title}**\n{selection.reasoning}\n")
    if selection_state is not None and selection.selected:
        selection_state.selection_plan = dict(selection.plan or {})
        selection_state.selection_advisories = list(selection.advisories)
        selection_state.selection_plan_source = selection.source
        selection_state.confirmed_tables = list(selection.selected)
        selection_state.selection_reasoning = selection.reasoning
    return (selection.selected, candidates, solr_meta, selection.reasoning, selection.trace, selection.tokens), ""


class _JudgeRun:
    """One run of the table judge over the retrieved candidates: the agent, its fallbacks, the trace."""

    def __init__(
        self, *, query: str, llm: LLM, pm: PromptManager, keywords: list[str], candidates: list[str],
        solr_meta: SolrMetadata, csv_dir: Path, config: RetrievalConfig, hint: str, portal_name: str,
        stream_callback: StreamCallback | None, cancel_check: Callable[[], None] | None,
    ) -> None:
        self.llm, self.keywords, self.candidates = llm, keywords, candidates
        self.stream_callback, self.cancel_check = stream_callback, cancel_check
        self.manager = Phase2JudgeToolsManager(
            candidates, csv_dir, question=query, metadata=solr_meta, value_search=config.mode.value_keywords,
        )
        self.system_prompt = pm.render(
            "data_architect", "system_prompt", portal_name=portal_name, hint=hint,
            value_search=config.mode.value_keywords,
        )
        self.token_counter = next(
            (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")), None,
        )
        if self.token_counter:
            self.token_counter.reset_counts()
        reset_llm_token_usage(llm)
        self.agent_prompt = pm.render(
            "data_architect", "user_prompt", question=query, keywords_str=" ".join(keywords),
            enriched_candidates_info=format_candidate_context(self.manager.visible_candidates(), solr_meta),
            table_hint=hint, value_search=config.mode.value_keywords,
            ranks_question_only=config.mode.ranks_question_only,
        )
        self.config = config
        self.stream_trace = io.StringIO()
        self.thinking_capture = ThinkingCapture()
        self.dispatcher = get_dispatcher()
        self.dispatcher.add_event_handler(self.thinking_capture)
        self.response = ""

    def emit(self, delta: str) -> None:
        if not delta:
            return
        self.stream_trace.write(delta)
        print(delta, end="", flush=True)
        if self.stream_callback is not None:
            self.stream_callback(delta)

    def judge(self) -> None:
        from lakegen.llm.agents.runner import run_agent_workflow

        visible = self.manager.visible_candidates()
        self.emit(
            "\n**Phase 2 – Table Judge agent started**\n"
            f"- Keywords from Phase 1: `{' '.join(self.keywords)}`\n"
            f"- {self.config.mode} retrieval built a pool of {len(self.candidates)} candidates; "
            f"showing {len(visible)} initially: "
            f"{', '.join(f'`{c}`' for c in visible)}\n"
            "- Agent inspecting and judging tables below.\n"
        )
        self.response = run_agent_workflow(
            llm=self.llm, system_prompt=self.system_prompt, user_prompt=self.agent_prompt,
            agent_name="table_judge", emit_stream=self.emit, cancel_check=self.cancel_check,
            tools=self.manager.get_tools(), max_iterations=10, max_repeats=3, max_tool_calls=15,
            timeout_seconds=300,
        )

    def _fallback(self, reason: str) -> None:
        """The inspected candidates with a minimal plan, when the judge did not decide."""
        inspected = self.manager.inspected_candidates()[:2]
        self.manager.selection_plan, _ = build_minimal_selection_fallback(inspected, reason)
        payload = {"tables": ", ".join(inspected), "reasoning": reason, "selection_plan": self.manager.selection_plan}
        self.response = f"FINAL_PAYLOAD: {json.dumps(payload)}"

    def after_stall(self, stall: Phase2AgentStall) -> None:
        self._fallback(f"Phase 2 loop guard triggered: {stall}. Fallback restricted to inspected candidates.")
        self.emit(
            "\n\n**Phase 2 loop guard triggered**\n"
            f"- Reason: `{stall}`\n"
            "- Action: using only inspected candidates as a fallback.\n"
        )

    def after_error(self, error: Exception) -> None:
        message = str(error)
        if "Max iterations" in message:
            reason = "Agent exceeded maximum iterations. Fallback to top candidates."
        else:
            reason = f"Agent error: {message[:120]}. Fallback to top 2."
        self._fallback(reason)
        self.emit(f"\n[phase2 agent error] {message[:160]}\n")

    def close(self, note: str) -> str:
        """The judge's activity log, behind the note of a selector that could not run."""
        full_trace = note + TRACE_HEADER + self.stream_trace.getvalue()
        self.stream_trace.close()
        self.dispatcher.event_handlers.remove(self.thinking_capture)
        return full_trace

    def tokens(self) -> int:
        tokens = 0
        if self.token_counter:
            tokens = self.token_counter.prompt_llm_token_count + self.token_counter.completion_llm_token_count
            self.token_counter.reset_counts()
        return max(tokens, get_llm_token_usage(self.llm))

    def settle(self, selection_state: object | None, all_files: list[str]) -> tuple[list[str], str]:
        """The judge's selection and reasoning, recorded on the round's state when there is one."""
        selected, reasoning = parse_table_selector_response(self.response, all_files, self.candidates)
        manager = self.manager
        if selection_state is not None:
            selection_state.rejection_keep_tables = list(manager.rejection_keep_tables)
            selection_state.rejection_skip_tables = list(manager.rejection_skip_tables)
        if selection_state is not None and manager.selection_plan:
            selection_state.selection_plan = dict(manager.selection_plan)
            selection_state.selection_plan_source = (
                "minimal_fallback"
                if manager.selection_plan.get("recovered_from_existing_discovery_context")
                else "confirm_divided_selection"
            )
            selection_state.confirmed_tables = list(selected)
            selection_state.selection_reasoning = reasoning
            selection_state.selection_requirements = dict(manager.selection_plan.get("requirements") or {})
        return selected, reasoning


def phase2_select_tables(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    keywords: list[str],
    solr_client: LocalSolrClient,
    csv_dir: Path,
    hint: str = "",
    portal_name: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check: Callable[[], None] | None = None,
    retrieval_config: RetrievalConfig | None = None,
    selection_state: object | None = None,
    entities: Sequence[str] | None = None,
    carried_tables: Sequence[str] = (),
    carried_metadata: SolrMetadata | None = None,
    excluded_tables: Iterable[str] = (),
    discovery: DiscoveryConfig | None = None,
) -> Phase2SelectionResult:
    """Run the configured retriever, then judge the retrieved tables.

    Flow:
    1. Execute keyword, semantic, or hybrid retrieval programmatically
    2. If 0 results → immediately return REJECT_KEYWORDS
    3. If results found → the configured selector, or the FunctionAgent judge
    4. Agent accepts (confirm_table_selection) or rejects (REJECT_KEYWORDS)

    Returns:
        (selected, all_candidates, solr_meta, reasoning, full_trace, tokens)
    """
    if selection_state is not None:
        selection_state.rejection_keep_tables = []
        selection_state.rejection_skip_tables = []
    config = retrieval_config or RetrievalConfig()
    candidates, solr_meta = _solr_and_search(
        query, keywords, solr_client, all_files, config, csv_dir, entities,
        carried_tables=carried_tables, carried_metadata=carried_metadata, excluded_tables=excluded_tables,
    )
    if not candidates:
        return _no_result(config, keywords, stream_callback)
    selected_by_selector, note = _configured_selector(
        discovery, query, candidates, solr_meta, csv_dir, selection_state, stream_callback,
    )
    if selected_by_selector is not None:
        return selected_by_selector
    run = _JudgeRun(
        query=query, llm=llm, pm=pm, keywords=keywords, candidates=candidates, solr_meta=solr_meta,
        csv_dir=csv_dir, config=config, hint=hint, portal_name=portal_name,
        stream_callback=stream_callback, cancel_check=cancel_check,
    )
    try:
        run.judge()
    except Phase2AgentStall as stall:
        run.after_stall(stall)
    except WorkflowCancelled:
        raise
    except Exception as error:
        run.after_error(error)
    finally:
        full_trace = run.close(note)
    tokens = run.tokens()
    selected, reasoning = run.settle(selection_state, all_files)
    return selected, candidates, solr_meta, reasoning, full_trace, tokens
