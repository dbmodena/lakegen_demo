"""The unified agent's tool manager: the tools it offers, which are still worth offering, and the record of each call."""

import functools
import inspect
from pathlib import Path
from typing import Callable

from llama_index.core.llms import LLM
from llama_index.core.tools import FunctionTool

from lakegen.config.experiment import DiscoveryConfig
from lakegen.domain.types import StreamCallback
from lakegen.integrations.solr.client import LocalSolrClient
from lakegen.retrieval import (
    RetrievalConfig,
    RetrievalRun,
)
from lakegen.tools.shared.context_editor import REPEATED_BLOCK, DiscoveryContext
from lakegen.tools.unified.brief import BriefMixin
from lakegen.tools.unified.inspection import InspectionMixin
from lakegen.tools.unified.plan_models import (
    ConfirmUnifiedSelectionSchema,
    RejectUnifiedSelectionSchema,
    SubmitSemanticPlanDraftSchema,
)
from lakegen.tools.unified.search import SearchMixin
from lakegen.tools.unified.selection import SelectionMixin
from lakegen.tools.unified.state import P12State


class Phase12ToolsManager(SearchMixin, InspectionMixin, SelectionMixin, BriefMixin):
    """Manager for Phase 1 & 2 unified tools to avoid closures and improve testability."""

    def __init__(
        self,
        state: P12State,
        solr_client: LocalSolrClient,
        all_files: list[str],
        csv_dir: Path,
        question: str = "",
        retrieval_config: RetrievalConfig | None = None,
        retrieval_observer: Callable[[RetrievalRun], None] | None = None,
        discovery_config: DiscoveryConfig | None = None,
        notice_callback: StreamCallback | None = None,
        llm: LLM | None = None,
    ):
        self.state = state
        self.solr_client = solr_client
        self.all_files = all_files
        self.csv_dir = Path(csv_dir)
        self.question = question
        self.retrieval_config = retrieval_config or RetrievalConfig()
        self.retrieval_observer = retrieval_observer
        self.discovery = discovery_config or DiscoveryConfig()
        self.notice_callback = notice_callback
        # Only used by the split keyword searches (see keyword_search in
        # DiscoveryConfig): the same LLM instance driving the agent, reused for
        # their internal calls rather than constructing a second, separate one.
        self.llm = llm
        # keyword_search="decompose_preview" only: per-word Solr match counts, memoized
        # for the life of this manager so sub-questions that share a word don't
        # re-query it.
        self._word_freq_cache: dict[str, int] = {}
        # Records every tool result and edits the agent's history before each
        # model call (state_board / compact_history in DiscoveryConfig).
        self.context = DiscoveryContext(self)

    def _emit_notice(self, text: str) -> None:
        """Report to the operator, never to the agent.

        This is deliberately not part of any tool's return value: the agent must
        choose concepts on the merits of the question, not adapt to whichever
        retriever happens to be configured. Falls back to stdout when no channel
        is wired, so a standalone run still says what searched.
        """
        if self.notice_callback is not None:
            self.notice_callback(text)
        else:
            print(text, end="", flush=True)

    def _record_retrieval_memory_event(self, **event: object) -> None:
        """Collect bounded, factual retrieval feedback for persistence."""
        self.state.retrieval_memory_events.append(dict(event))

    def get_semantic_planner_tools(self) -> list[FunctionTool]:
        return [FunctionTool.from_defaults(
            fn=self.submit_semantic_plan_draft,
            fn_schema=SubmitSemanticPlanDraftSchema,
            return_direct=True,
        )]

    def _search_tool(self) -> FunctionTool:
        if self.retrieval_config.mode.verbatim_entities:
            # Entities are matched whole, so the agent passes a real list, as for
            # cell values. The tool keeps its name for prompts and activity logs.
            return FunctionTool.from_defaults(
                fn=self.search_table_entities,
                name="search_tables",
                description=self._search_tool_description(),
            )
        if self.retrieval_config.mode.value_keywords:
            # Values are searched whole, so the agent passes a real list; splitting
            # a string on spaces would turn "East River" into two searches. The
            # tool keeps its name, so prompts and activity logs are unchanged.
            return FunctionTool.from_defaults(
                fn=self.search_table_values,
                name="search_tables",
                description=self._search_tool_description(),
            )
        return FunctionTool.from_defaults(
            fn=self.search_keyword_concepts,
            name="search_tables",
            description=self._search_tool_description(),
        )

    def is_tool_available(self, tool_name: str) -> bool:
        """Report whether a tool is still worth offering to the model.

        Every tool already refuses gracefully once spent, but that refusal
        still costs a full model turn and one of the run's capped tool calls.
        This mirrors each tool's own exhaustion checks -- never a precondition
        the tool just hasn't met *yet* -- so the pruning agent can drop a
        genuinely spent tool from a turn's offering instead of paying for a
        refusal that only repeats what the last one already said. A tool
        blocked only by something else that can still change (an expansion
        that hasn't run, a carried schema not yet served) stays offered.
        """
        if tool_name == "search_tables":
            mode = self.retrieval_config.mode
            if (
                mode.uses_embeddings
                and self.state.semantic_failure is not None
            ):
                return False
            # inspection_counts, not inspection_cache: see _search's own note --
            # a carried-forward schema must not look like an in-round inspection.
            if self.state.inspection_counts and not self.discovery.search_after_inspection:
                return False
            return not self._search_limit_reached()

        if tool_name == "expand_candidates":
            # Only the spendable resource -- the expansion budget -- withdraws
            # this tool. "Nothing to inspect yet" and "nothing hidden right now"
            # are the tool's own, still-changeable refusals, not exhaustion.
            return self.state.expansion_count < self.discovery.max_expansions

        if tool_name == "inspect_columns":
            visible_keys = {
                candidate.casefold()
                for candidate in self.state.all_candidates[: self.state.visible_candidate_count]
            }
            attempted_candidates = len(self.state.inspection_counts)
            current_limit = (
                self.discovery.initial_shortlist_size
                if self.state.expansion_count == 0
                else self.discovery.max_inspected_candidates
            )
            if attempted_candidates < current_limit:
                return True
            # A carried schema not yet served this round bypasses the limit in
            # inspect_columns itself, so it must not look spent here either.
            has_unserved_carried = any(
                key in self.state.inspection_cache and key not in self.state.inspection_counts
                for key in visible_keys
            )
            if has_unserved_carried:
                return True
            return (
                self.state.expansion_count == 0
                and self.is_tool_available("expand_candidates")
                and self.state.visible_candidate_count < len(self.state.all_candidates)
            )

        return True

    @staticmethod
    def tool_names() -> tuple[str, ...]:
        return (
            "search_tables", "inspect_columns", "expand_candidates",
            "check_join_union", "confirm_unified_selection", "reject_unified_selection",
        )

    def _tracked(self, fn: Callable[..., str], tool_name: str) -> Callable[..., str]:
        """Record each result for the context editor and, with state_board,
        flag a call that can only repeat a refusal the model already has."""

        @functools.wraps(fn)
        def call(*args, **kwargs):
            withdrawn = (
                self.discovery.state_board
                and tool_name in ("search_tables", "inspect_columns", "expand_candidates")
                and not self.is_tool_available(tool_name)
            )
            try:
                output = fn(*args, **kwargs)
            except Exception as exc:
                message = str(exc)
                if tool_name == "confirm_unified_selection":
                    repeated = message == self.state.last_confirm_error
                    self.state.last_confirm_error = message
                    self.state.confirm_blocks += 1
                    if repeated and self.discovery.state_board:
                        # A live architect resubmitted the same blocked
                        # selection three times, each costing a full turn.
                        message = REPEATED_BLOCK + message
                        self.context.record(tool_name, message, error=True)
                        raise ValueError(message) from exc
                self.context.record(tool_name, message, error=True)
                raise
            if withdrawn:
                output = (
                    f"{tool_name} is no longer available in this attempt; the "
                    "state block lists the tools still worth calling. " + output
                )
            self.context.record(tool_name, output)
            return output

        return call

    def get_tools(self) -> list[FunctionTool]:
        def tool(fn, **kwargs) -> FunctionTool:
            # An explicit description keeps LlamaIndex from prepending the
            # Python signature, which only repeated the parameter schema.
            kwargs.setdefault("description", inspect.cleandoc(fn.__doc__ or ""))
            if self.context.enabled:
                fn = self._tracked(fn, fn.__name__)
            return FunctionTool.from_defaults(fn=fn, **kwargs)

        search = self._search_tool()
        if self.context.enabled:
            search = FunctionTool.from_defaults(
                fn=self._tracked(search.fn, "search_tables"),
                name="search_tables",
                description=search.metadata.description,
            )
        return [
            search,
            tool(self.inspect_columns),
            tool(self.expand_candidates),
            tool(self.check_join_union),
            tool(self.confirm_unified_selection, fn_schema=ConfirmUnifiedSelectionSchema, return_direct=True),
            tool(self.reject_unified_selection, fn_schema=RejectUnifiedSelectionSchema, return_direct=True),
        ]
