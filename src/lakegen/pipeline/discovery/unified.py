"""The unified discovery agent: one MemGPT agent that searches the catalogue and selects from what it finds.

It is the divided architecture's MemGPT selector (``selectors.memgpt``) with a ``search`` tool, and
one rule the system keeps about it (``UnifiedAgent.search_offered``): search is offered while there is
nothing to work from -- before the first search finds candidates, and after one that found nothing
new -- and withdrawn as soon as a search has found at least one candidate. It comes back when a
requirement is still open, at least one table has been inspected since, nothing is left to inspect,
and no hidden candidate mentions it (or the expansions are spent): the point where the divided round
rejects its keywords and starts over. At most ``REACTIVATIONS`` such searches, ``MAX_SEARCHES`` in all.

The first search is production's retrieval of the question with the query's words as keywords
(``divided.retrieve_candidates``: strict AND, the question to the semantic branch, reranked), so the
agent's query plays the keyword writer's part; a search made later is the retrieval service's search
for one requirement (``search_for``: OR over the words, KNN over the query, reranked), whose new
tables are appended. Once the agent confirms, the contrast, drop and alternatives checks run as in the
divided round, and the selection is recorded on the round's state the same way.

Measured in experiments/unified_lab (dev1, 228 questions, OCI gpt-oss-20b): exact selections 0.750
against 0.566 for the agent it replaced (49 fixed / 7 broken, p < 0.001) and 0.768 for the divided
round (6 / 10, within rerun noise), at 72 s per question against 78 and 97.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from llama_index.core.llms import LLM

from lakegen.config.registry import catalogue_note
from lakegen.datastructures.errors import WorkflowCancelled
from lakegen.datastructures.types import SolrMetadata, StreamCallback, ToolCallback
from lakegen.llm.prompts import PromptManager
from lakegen.llm.token_usage import get_llm_token_usage, reset_llm_token_usage
from lakegen.pipeline.discovery.divided import (
    no_selection,
    record_selection,
    retrieve_candidates,
    streamer,
)
from lakegen.pipeline.discovery.selectors import SelectorUnavailable
from lakegen.pipeline.discovery.selectors.memgpt import agent as memgpt
from lakegen.pipeline.discovery.selectors.memgpt import alternatives, contrast, drop, selection_of
from lakegen.pipeline.discovery.selectors.memgpt.agent import (
    ACTIONS,
    INITIAL_VISIBLE,
    INSPECT_BATCH,
    MAX_EXPANSIONS,
    MAX_TURNS,
    TERMINAL,
    MemGPTAgent,
)
from lakegen.pipeline.discovery.selectors.memgpt.text import YEAR, content_words
from lakegen.pipeline.discovery.state import DiscoveryState
from lakegen.retrieval import RetrievalConfig
from lakegen.retrieval.factory import get_table_retrieval_service
from lakegen.retrieval.solr.catalogue import (
    lake_titles,
    match_local_csv,
    solr_metadata_from_doc,
    with_dataset_title,
)
from lakegen.retrieval.solr.client import SolrClient

MAX_SEARCHES = 4
REACTIVATIONS = 2
SEARCH_ADDED = 10
PROMPT = "unified_architect"
TITLE = "Unified MemGPT agent"
SEARCH_TOOL = (
    "Search the catalogue: query = the words a table's title, description or columns would use for "
    'the data the question needs, e.g. "preferenze lista elezioni 2009"; requirement = the id of the '
    "open requirement it is for, when there is one. Candidates found are numbered after the last; "
    "inspect the ones that could show the requirements."
)
"""What the model reads as the search tool's description."""

Searcher = Callable[[str, bool], list[tuple[str, dict[str, Any]]]]
"""``(query, first)`` -> the tables found with their catalogue entries, best first; ``first`` before any candidate."""


def search_words(query: str) -> list[str]:
    """The words of a query, each once, with its years."""
    return list(dict.fromkeys(content_words(query) + YEAR.findall(query)))


class CatalogueSearcher:
    """The agent's searches over the lake's catalogue, honouring what earlier rounds carried and excluded.

    Args:
        question: The user's question.
        solr_client: Client of the lake's Solr core.
        all_files: Local table files of the lake.
        csv_dir: Folder of the lake's tables.
        retrieval_config: Retrieval settings; the defaults when None.
        state: The round's state: the carried and excluded tables, where the searches are recorded.
        lake: Lake name, for the dataset titles; empty for none.
    """

    def __init__(
        self,
        question: str,
        solr_client: SolrClient,
        all_files: list[str],
        csv_dir: Path,
        retrieval_config: RetrievalConfig | None,
        state: DiscoveryState,
        lake: str,
    ) -> None:
        self.question, self.solr, self.all_files, self.csv_dir = (
            question,
            solr_client,
            all_files,
            csv_dir,
        )
        self.config = retrieval_config or RetrievalConfig()
        self.state, self.lake = state, lake

    def __call__(self, query: str, first: bool) -> list[tuple[str, dict[str, Any]]]:
        """The tables a search finds: production's retrieval first, then the search for one requirement.

        Args:
            query: The agent's query.
            first: Whether the agent has no candidates yet.

        Returns:
            The tables found with their catalogue entries, best first.
        """
        words = search_words(query)
        found = self._retrieve(words) if first else self._search(query, words)
        self.state.search_attempts.append(
            {
                "query": query,
                "keywords": words,
                "first": first,
                "found": [name for name, _ in found],
            }
        )
        if first and found:
            self.state.used_keywords = words
        return found

    def _retrieve(self, words: list[str]) -> list[tuple[str, dict[str, Any]]]:
        candidates, metadata = retrieve_candidates(
            self.question,
            words,
            self.solr,
            self.all_files,
            self.config,
            self.csv_dir,
            None,
            carried_tables=self.state.carried_tables,
            carried_metadata=self.state.carried_metadata,
            excluded_tables=self.state.excluded_tables,
            lake=self.lake,
        )
        return [(name, metadata[name]) for name in candidates]

    def _search(self, query: str, words: list[str]) -> list[tuple[str, dict[str, Any]]]:
        service = get_table_retrieval_service(
            self.solr,
            self.config,
            **({"table_dir": self.csv_dir} if self.config.mode.requires_table_dir else {}),
        )
        titles = lake_titles(self.lake) if self.lake else {}
        found: list[tuple[str, dict[str, Any]]] = []
        for hit in service.search_for(query, words, top_k=SEARCH_ADDED * 2):
            name = match_local_csv(hit.document, self.all_files)
            if name is None or name.casefold() in self.state.excluded_tables:
                continue
            if any(name == seen for seen, _ in found):
                continue
            meta = with_dataset_title(solr_metadata_from_doc(hit.document), hit.document, titles)
            meta["retrieval"] = hit.to_log_dict()
            found.append((name, meta))
        return found


class UnifiedAgent(MemGPTAgent):
    """The MemGPT agent that searches for its own candidates, under the search rule.

    Attributes:
        searcher: Answers the searches, set by the caller.
        searches: Searches made.
        reactivations: Searches made after search was withdrawn.
        closed: Search withdrawn: a search has found candidates (the rule reopens it).
        inspected_at_close: Tables inspected when search was last withdrawn; the rule
            wants at least one more before it reopens.
        queries: One record per search: the query, the requirement it was for, what it found.

    Args:
        question: The user's question.
        table_dir: Folder of the lake's tables.
        llm: The run's LLM.
        prompts: The run's prompt manager.
        portal: Name of the open-data portal.
        lake: The lake's name, for the portal's catalogue note.
        hint: Feedback from the previous attempt.
        emit: Receives the live lines of the run, if any.
        cancel_check: Raises when the run is cancelled.
        tool_callback: Receives each tool call, for a surface that shows them as steps.
    """

    searcher: Searcher | None = None

    def __init__(
        self,
        *,
        question: str,
        table_dir: Path,
        llm: Any,
        prompts: Any,
        portal: str = "",
        lake: str = "",
        hint: str = "",
        emit: Callable[[str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        tool_callback: ToolCallback | None = None,
    ) -> None:
        super().__init__(
            question=question,
            candidates=[],
            metadata={},
            table_dir=table_dir,
            llm=llm,
            prompts=prompts,
            portal=portal,
            lake=lake,
            hint=hint,
            emit=emit,
            cancel_check=cancel_check,
            tool_callback=tool_callback,
        )
        self.searches = 0
        self.reactivations = 0
        self.closed = False
        self.inspected_at_close = 0
        self.queries: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- the rule

    def search_offered(self) -> bool:
        """Whether search is offered this turn."""
        if self.searches >= MAX_SEARCHES or self.actions_left <= 0:
            return False
        if not self.closed:
            return True
        m = self.memory
        return (
            self.reactivations < REACTIVATIONS
            and bool(m.open_requirements())
            and not m.queue()
            and len(m.cards) > self.inspected_at_close
            and (not m.hidden_matches() or self.expansions >= MAX_EXPANSIONS)
        )

    def available(self) -> list[str]:
        """The tools offered now: search first, the selector's once there are candidates."""
        if self.decision_mode:
            return list(TERMINAL)
        if not self.memory.requirements:
            return ["define_requirements"]
        if not self.candidates:
            return ["search", "reject"] if self.search_offered() else ["reject"]
        names = super().available()
        if self.search_offered():
            names.insert(0, "search")
        return names

    def next_step(self) -> str:
        """What STATE tells the agent to do next."""
        m = self.memory
        if m.requirements and not self.decision_mode and "search" in self.available():
            if not self.candidates:
                return (
                    "search(query) with the words a table's title, description or columns would use"
                    + (
                        " -- other words: the previous search found nothing"
                        if self.searches
                        else ""
                    )
                )
            opened = m.open_requirements()
            if opened:
                return (
                    f"search for {opened[0].id} ({opened[0].text[:50]}) with other words: "
                    "no candidate left mentions it"
                )
        return super().next_step()

    def status(self) -> list[str]:
        """STATE's closing lines, with the searches spent."""
        lines = super().status()
        lines[0] += f"; searches {self.searches} of {MAX_SEARCHES}"
        return lines

    # ---------------------------------------------------------------- the tool

    def search(self, query: str, requirement: str = "") -> str:
        """The search tool: ask the catalogue for candidates and append the new ones.

        The model reads ``SEARCH_TOOL`` as this tool's description.

        Args:
            query: The agent's query.
            requirement: The id of the open requirement the search is for, when there is one.

        Returns:
            The tool's result for the model.
        """
        query = " ".join(str(query or "").split())
        if not query:
            return "Error: give the words to search for."
        first = not self.candidates
        self.searches += 1
        self.actions += 1
        if self.closed:
            self.reactivations += 1
        found = self.searcher(query, first) if self.searcher is not None else []
        new = [(name, meta) for name, meta in found if name not in self.candidates]
        if not first:
            new = new[:SEARCH_ADDED]
        self.queries.append(
            {
                "query": query,
                "requirement": str(requirement),
                "first": first,
                "found": len(found),
                "new": [name for name, _ in new],
            }
        )
        if not new:
            return (
                f"No new candidate for '{query}'"
                + (
                    ": every result is already a candidate"
                    if found
                    else ": the catalogue matched nothing"
                )
                + ". Search with other words (the subject and the item together, a year, a place, "
                "the portal's own language)"
                + ("" if first else ", or decide with the inspected tables")
                + "."
            )
        start = len(self.candidates)
        for name, meta in new:
            self.candidates.append(name)
            self.metadata[name] = meta
        self.memory.visible = (
            min(INITIAL_VISIBLE, len(self.candidates)) if first else len(self.candidates)
        )
        self.closed = True
        self.inspected_at_close = len(self.memory.cards)
        numbers = list(range(start + 1, self.memory.visible + 1))
        hidden = len(self.candidates) - self.memory.visible
        lines = [f"#{n} {self.memory.title(n)[:100]}" for n in numbers]
        opened = self.memory.open_requirements()
        matching = [n for n in numbers if any(self.memory.matches(n, r) for r in opened)]
        head = f"Found {len(new)} candidates for '{query}'" + (
            f"; #{numbers[0]}-#{numbers[-1]} shown, {hidden} hidden (expand reveals them)"
            if hidden
            else f" (#{numbers[0]}-#{numbers[-1]})"
        )
        tail = (
            f"Their catalogue entries mention an open requirement: {matching}. Inspect them."
            if matching
            else "None of their catalogue entries mentions an open requirement: inspect the likeliest."
        )
        return head + ":\n" + "\n".join(lines) + "\n" + tail

    def functions(self) -> dict[str, Callable[..., str]]:
        """The tools by name, as bound methods."""
        return {**super().functions(), "search": self.search}

    # ---------------------------------------------------------------- the read-only block

    def retrieval_notes(self) -> str:
        """The read-only block's lines on retrieval: the searches so far, the previous attempt's feedback."""
        lines = [
            f"Search {i}: '{q['query']}' -> {len(q['new'])} new candidates"
            for i, q in enumerate(self.queries, 1)
        ]
        if self.hint:
            lines.append(f"Feedback from the previous attempt: {self.hint}")
        return "\n".join(lines)

    def system_message(self) -> str:
        """The read-only block: instructions, tools, question, the searches and the requirements."""
        return self.prompts.render(
            PROMPT,
            "system_prompt",
            batch=INSPECT_BATCH,
            max_expansions=MAX_EXPANSIONS,
            actions=ACTIONS,
            turns=MAX_TURNS,
            searches=MAX_SEARCHES,
            portal=self.portal,
            catalogue_notes=catalogue_note(self.lake),
            question=self.question,
            retrieval_notes=self.retrieval_notes(),
            requirements=self.memory.render_requirements(),
        )


UnifiedAgent.search.__doc__ = SEARCH_TOOL

# A search call the model writes as text is recovered like the selector's own calls.
if ("search", {"query"}) not in memgpt.RECOVERY:
    memgpt.RECOVERY = (("search", {"query"}), *memgpt.RECOVERY)


def discover_tables(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    solr_client: SolrClient,
    csv_dir: Path,
    hint: str = "",
    portal_name: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check: Callable[[], None] | None = None,
    tool_callback: ToolCallback | None = None,
    retrieval_config: RetrievalConfig | None = None,
    state: DiscoveryState | None = None,
    lake: str = "",
) -> tuple[list[str], list[str], SolrMetadata, str, str, int]:
    """The unified agent on one question: the selected tables, keywords, metadata, reasoning, trace and tokens.

    A run the agent could not finish (a failure, a run that ends with nothing
    selected) rejects the round like a divided selector that could not select,
    so the pipeline starts another round with the feedback.

    Args:
        query: The user's question.
        llm: The run's LLM.
        pm: Prompt manager of the run.
        all_files: Local table files of the lake.
        solr_client: Client of the lake's Solr core.
        csv_dir: Folder of the lake's tables.
        hint: Feedback from the previous attempt.
        portal_name: Name of the open-data portal.
        stream_callback: UI channel for the live output, if any.
        cancel_check: Raises when the run is cancelled.
        tool_callback: Receives the agent's tool calls, if any.
        retrieval_config: Retrieval settings; the defaults when None.
        state: The round's state, which records the searches and the selection; a fresh one when None.
        lake: Lake name, for the dataset titles; empty for none.

    Returns:
        (selected, keywords, solr_meta, reasoning, full_trace, tokens)

    Raises:
        WorkflowCancelled: The run was cancelled.
    """
    state = state if state is not None else DiscoveryState()
    agent = UnifiedAgent(
        question=query,
        table_dir=csv_dir,
        llm=llm,
        prompts=pm,
        portal=portal_name,
        lake=lake,
        hint=hint,
        emit=streamer(stream_callback),
        cancel_check=cancel_check,
        tool_callback=tool_callback,
    )
    agent.searcher = CatalogueSearcher(
        query, solr_client, all_files, csv_dir, retrieval_config, state, lake
    )
    if stream_callback:
        stream_callback(f"\n**Phase 1 & 2 – {TITLE} started**\n")
    reset_llm_token_usage(llm)
    note = ""
    checked = dropped = offered = None
    try:
        agent.run()
        checked = contrast.check(agent, llm, pm)
        dropped = drop.check(agent, llm, pm)
        offered = alternatives.check(agent, llm, pm)
    except WorkflowCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 -- an agent that breaks asks for another round
        note = f"[unified agent failed] {type(exc).__name__}: {exc}\n"
    state.all_candidates = list(agent.candidates)
    state.solr_meta = dict(agent.metadata)
    state.visible_candidate_count = min(INITIAL_VISIBLE, len(agent.candidates))
    state.expansion_count = agent.expansions
    tokens = get_llm_token_usage(llm)
    selection = None
    if not note:
        try:
            selection = selection_of(agent, tokens, checked, dropped, offered)
        except SelectorUnavailable as exc:
            note = f"[unified agent unavailable] {exc}\n"
    if selection is None:
        return no_selection(note, state.all_candidates, state.solr_meta, stream_callback)
    trace = selection.trace
    print(f"[phase12] {trace}", flush=True)
    if stream_callback:
        stream_callback(f"\n**Phase 1 & 2 – {TITLE}**\n{selection.reasoning}\n")
    record_selection(selection, state)
    return (
        selection.selected,
        list(state.used_keywords),
        state.solr_meta,
        selection.reasoning,
        trace,
        selection.tokens,
    )
