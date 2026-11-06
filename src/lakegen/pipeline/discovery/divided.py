"""Phase 2 of the divided architecture: retrieve with the writer's keywords, then select.

The configured retriever runs programmatically over the keywords Phase 1 wrote;
the table selector (the MemGPT agent) confirms tables among the retrieved
candidates or sends the keywords back to the writer.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from llama_index.core.llms import LLM

from lakegen.config.experiment import DiscoveryConfig
from lakegen.datastructures.types import (
    Phase2SelectionResult,
    SolrMetadata,
    StreamCallback,
    ToolCallback,
)
from lakegen.llm.prompts import PromptManager
from lakegen.pipeline.discovery.selectors import Selection, SelectionRequest, SelectorUnavailable
from lakegen.pipeline.discovery.selectors import choose as choose_selector
from lakegen.pipeline.evidence.candidates import match_local_csv, solr_metadata_from_doc
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.factory import get_table_retrieval_service
from lakegen.retrieval.solr.catalogue import lake_titles, with_dataset_title
from lakegen.retrieval.solr.client import SolrClient

logger = logging.getLogger(__name__)


def retrieve_candidates(
    query: str,
    keywords: list[str],
    solr_client: SolrClient,
    all_files: list[str],
    retrieval_config: RetrievalConfig | None = None,
    table_dir: Path | None = None,
    entities: Sequence[str] | None = None,
    carried_tables: Sequence[str] = (),
    carried_metadata: SolrMetadata | None = None,
    excluded_tables: Iterable[str] = (),
    lake: str = "",
) -> tuple[list[str], SolrMetadata]:
    """Execute the configured retriever and return candidates + metadata.

    ``carried_tables`` are candidates a previous, rejected attempt already
    verified as partially satisfying the question; they are re-surfaced here
    regardless of whether this round's retrieval finds them again.
    ``excluded_tables`` were inspected and ruled out by a previous attempt and
    must not be re-proposed. A keyword-mode retrieval error leaves the
    retrieved candidates empty; any other mode's error escapes.

    Args:
        query: The user's question.
        keywords: The writer's keywords.
        solr_client: Client of the lake's Solr core.
        all_files: Local table files of the lake.
        retrieval_config: Retrieval settings; the defaults when None.
        table_dir: Folder of the lake's tables, for modes that read them.
        entities: Named entities of the question, for retrievers that use them.
        carried_tables: Tables a rejected attempt kept, put first.
        carried_metadata: Catalogue metadata of the carried tables.
        excluded_tables: Tables a rejected attempt ruled out.
        lake: Lake name, for the dataset titles; empty for none.

    Returns:
        The candidate table files, at most ``top_k``, and their metadata.
    """
    candidates: list[str] = []
    metadata: SolrMetadata = {}
    config = retrieval_config or RetrievalConfig()
    excluded = {str(table).casefold() for table in excluded_tables}
    # Retrieval itself truncates to top_k, so asking for exactly top_k would
    # let a skip-listed hit shrink the final pool instead of being replaced by
    # the next-ranked table just outside the previous cutoff.
    fetch_k = config.top_k + len(excluded)
    titles = lake_titles(lake) if lake else {}

    try:
        retriever = get_table_retrieval_service(
            solr_client,
            config,
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
            f"[phase2] {config.mode} retrieval keywords={keywords} docs_returned={len(hits)}",
            flush=True,
        )

        for hit in hits:
            doc = hit.document
            matched = match_local_csv(doc, all_files)
            if matched is None or matched in candidates or matched.casefold() in excluded:
                continue
            candidates.append(matched)
            metadata[matched] = with_dataset_title(solr_metadata_from_doc(doc), doc, titles)
            metadata[matched]["retrieval"] = hit.to_log_dict()
            if len(candidates) >= config.top_k:
                break

    except Exception as solr_err:
        logger.warning("[phase2] Solr error %s: %s", type(solr_err).__name__, solr_err)
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


def _no_result(
    config: RetrievalConfig, keywords: list[str], stream_callback: StreamCallback | None
) -> Phase2SelectionResult:
    """Retrieval found nothing: the keywords go back to the writer, before any selector runs."""
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


def no_selection(
    note: str,
    candidates: list[str],
    solr_meta: SolrMetadata,
    stream_callback: StreamCallback | None,
) -> Phase2SelectionResult:
    """No selector could select: the keywords go back to the writer, as after an empty retrieval."""
    reason = note.strip().splitlines()[-1]
    message = (
        f"REJECT_KEYWORDS: No selection could be made from the {len(candidates)} retrieved candidates "
        f"({reason}). Try broader or different terms."
    )
    print(f"[phase2] {note}", flush=True)
    if stream_callback:
        stream_callback(
            f"\n⚠️ **No table selection**: {reason}\nSending feedback to Phase 1 for other keywords.\n"
        )
    return [], candidates, solr_meta, message, note, 0


def streamer(stream_callback: StreamCallback | None) -> Callable[[str], None]:
    """The selector's live output: printed to the terminal once and forwarded to the UI channel once."""

    def emit(delta: str) -> None:
        if not delta:
            return
        print(delta, end="", flush=True)
        if stream_callback is not None:
            stream_callback(delta)

    return emit


def _select(
    discovery: DiscoveryConfig | None, request: SelectionRequest
) -> tuple[Selection | None, str, str]:
    """The configured selector's selection; None when it could not select.

    Args:
        discovery: Discovery settings naming the selector; the defaults when None.
        request: What the selector selects from.

    Returns:
        The selection (None when the selector could not select), the note on
        what could not run (the trace starts with it) and the title of the selector.
    """
    selector = choose_selector(discovery)
    try:
        return selector.select(request), "", selector.title
    except SelectorUnavailable as exc:
        return None, f"[{selector.kind.value} selector unavailable] {exc}\n", selector.title


def record_selection(selection: Selection, selection_state: object | None) -> None:
    """The selection on the round's state: the plan for the coder, or the tables a rejection keeps."""
    if selection_state is None:
        return
    if selection.selected:
        selection_state.selection_plan = dict(selection.plan or {})
        selection_state.selection_advisories = list(selection.advisories)
        selection_state.selection_plan_source = selection.source
        selection_state.confirmed_tables = list(selection.selected)
        selection_state.selection_reasoning = selection.reasoning
    else:
        selection_state.rejection_keep_tables = list(selection.keep_tables)
        selection_state.rejection_skip_tables = list(selection.skip_tables)


def select_tables(
    query: str,
    llm: LLM,
    pm: PromptManager,
    all_files: list[str],
    keywords: list[str],
    solr_client: SolrClient,
    csv_dir: Path,
    hint: str = "",
    portal_name: str = "",
    stream_callback: StreamCallback | None = None,
    cancel_check: Callable[[], None] | None = None,
    tool_callback: ToolCallback | None = None,
    retrieval_config: RetrievalConfig | None = None,
    selection_state: object | None = None,
    entities: Sequence[str] | None = None,
    carried_tables: Sequence[str] = (),
    carried_metadata: SolrMetadata | None = None,
    excluded_tables: Iterable[str] = (),
    discovery: DiscoveryConfig | None = None,
    lake: str = "",
) -> Phase2SelectionResult:
    """Run the configured retriever, then select among the retrieved tables.

    Flow:
    1. Execute keyword, semantic, or hybrid retrieval programmatically
    2. If 0 results → immediately return REJECT_KEYWORDS
    3. If results found → the table selector, the MemGPT agent
    4. The selector confirms tables or rejects them (REJECT_KEYWORDS); a round the
       selector could not decide rejects too

    Args:
        query: The user's question.
        llm: The run's LLM, for the MemGPT selector.
        pm: Prompt manager of the run.
        all_files: Local table files of the lake.
        keywords: The writer's keywords.
        solr_client: Client of the lake's Solr core.
        csv_dir: Folder of the lake's tables.
        hint: The user's hint for this round.
        portal_name: Name of the open-data portal.
        stream_callback: UI channel for the live output, if any.
        cancel_check: Raises when the run is cancelled.
        tool_callback: Receives the selector's tool calls, if any.
        retrieval_config: Retrieval settings; the defaults when None.
        selection_state: The round's state, which records the selection.
        entities: Named entities of the question.
        carried_tables: Tables a rejected attempt kept.
        carried_metadata: Catalogue metadata of the carried tables.
        excluded_tables: Tables a rejected attempt ruled out.
        discovery: Discovery settings naming the selector.
        lake: Lake name, for the dataset titles.

    Returns:
        (selected, all_candidates, solr_meta, reasoning, full_trace, tokens)
    """
    if selection_state is not None:
        selection_state.rejection_keep_tables = []
        selection_state.rejection_skip_tables = []
    config = retrieval_config or RetrievalConfig()
    candidates, solr_meta = retrieve_candidates(
        query,
        keywords,
        solr_client,
        all_files,
        config,
        csv_dir,
        entities,
        carried_tables=carried_tables,
        carried_metadata=carried_metadata,
        excluded_tables=excluded_tables,
        lake=lake,
    )
    if not candidates:
        return _no_result(config, keywords, stream_callback)
    request = SelectionRequest(
        query,
        candidates,
        solr_meta,
        csv_dir,
        discovery or DiscoveryConfig(),
        llm=llm,
        keywords=tuple(keywords),
        portal_name=portal_name,
        hint=hint,
        retrieval_mode=config.mode,
        prompts=pm,
        lake=lake,
        stream_callback=streamer(stream_callback),
        cancel_check=cancel_check,
        tool_callback=tool_callback,
    )
    selection, note, title = _select(discovery, request)
    if selection is None:
        return no_selection(note, candidates, solr_meta, stream_callback)
    trace = note + selection.trace
    print(f"[phase2] {trace}", flush=True)
    if stream_callback:
        stream_callback(f"\n**Phase 2 – {title}**\n{selection.reasoning}\n")
    record_selection(selection, selection_state)
    return selection.selected, candidates, solr_meta, selection.reasoning, trace, selection.tokens
