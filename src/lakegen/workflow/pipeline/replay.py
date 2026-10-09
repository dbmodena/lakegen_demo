"""Offline replays: a run that starts from what an earlier run recorded.

A ``Replay`` names what is taken from the record instead of being run again:

* ``keywords``: the divided strategies take the stored keyword round as the
  approved one, so the keyword writer does not run (a run that starts at
  the selection);
* ``selection``: the coder starts from the stored tables, so no discovery
  runs at all (a run that starts at the code). The coder may still reject
  them; a replay has nothing new to discover, so the engine's repeat guard
  stops the run when the same selection comes back, as it would in a live
  run.

The catalogue entries of the stored tables are read from Solr when the
record does not hold them. The recorded metrics say which rounds were
replayed: the selector of a replayed round is ``stored``.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lakegen.domain.outcomes import DiscoveryOutcome, DiscoveryStatus, SelectorKind
from lakegen.domain.tables import resource_id
from lakegen.integrations.solr.catalogue import match_local_csv, solr_metadata_from_doc
from lakegen.tools.unified import P12State
from lakegen.workflow.pipeline.model.budgets import Attempt
from lakegen.workflow.pipeline.model.events import Answer, Request

if TYPE_CHECKING:
    from lakegen.workflow.pipeline.stages import Run

TRACE_HEADER = "--- Replayed Selection ---"


@dataclass(frozen=True)
class StoredKeywords:
    """A keyword round an earlier run wrote, taken as approved."""

    keywords: tuple[str, ...]
    raw: str = ""
    reasoning: str = "replayed keywords"

    def round(self) -> tuple[list[str], str, int, str]:
        """In the shape of the keyword writer's answer; its tokens were spent then, not now."""
        return (list(self.keywords), self.raw, 0, self.reasoning)


@dataclass(frozen=True)
class StoredSelection:
    """The tables an earlier run selected, to code from without rediscovering."""

    tables: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """The catalogue entries by table; read from Solr when empty."""
    selection_plan: Mapping[str, Any] | None = None
    reasoning: str = "Replayed from an earlier run's selection."

    @classmethod
    def from_result(cls, result: Mapping[str, Any]) -> StoredSelection:
        """From a stored ``QueryResult`` (the ``result`` of a job's result entry).

        The tables and keywords are the result's; the selection plan is the
        last discovery attempt's, when the record kept the attempts.
        """
        discovery = result.get("discovery") or {}
        tables = result.get("tables") or discovery.get("selected_datasets") or []
        keywords = result.get("keywords") or discovery.get("keywords") or []
        plan = next(
            (
                attempt.get("selection_plan")
                for attempt in reversed(discovery.get("selection_attempts") or [])
                if attempt.get("selection_plan")
            ),
            None,
        )
        return cls(tables=tuple(tables), keywords=tuple(keywords), selection_plan=plan)


@dataclass(frozen=True)
class Replay:
    keywords: StoredKeywords | None = None
    selection: StoredSelection | None = None


def catalogue_metadata(solr: Any, tables: list[str], all_files: list[str]) -> dict[str, dict[str, Any]]:
    """The catalogue entry of each table, from the Solr document of its resource.

    A table whose document is missing, or whose document names another local
    file, gets an empty entry: the coder then reads the schema from the file.
    """
    metadata: dict[str, dict[str, Any]] = {}
    for table in tables:
        response = solr.select([f'resource_id:"{resource_id(table)}"'], rows=5)
        docs = (response.get("response") or {}).get("docs") or []
        match = next((doc for doc in docs if match_local_csv(doc, all_files) == table), None)
        metadata[table] = solr_metadata_from_doc(match) if match is not None else {}
    return metadata


class ReplayedDiscovery:
    """Stands in for ``DiscoveryStage``: the round's outcome is the stored selection."""

    def __init__(self, run: Run, selection: StoredSelection) -> None:
        self.run = run
        self.selection = selection

    def run_round(
        self, attempt: Attempt, selection_record: dict[str, Any],
    ) -> Generator[Request, Answer, tuple[DiscoveryOutcome, bool]]:
        yield from ()  # a replay asks nothing; the signature is the stage's
        run, state, selection = self.run, self.run.state, self.selection
        started = time.monotonic()
        tables = list(selection.tables)
        metadata = {table: dict(entry) for table, entry in selection.metadata.items()}
        if not metadata:
            metadata = catalogue_metadata(run.solr, tables, run.all_files)
        outcome = DiscoveryOutcome(
            status=DiscoveryStatus.SELECTED, selected=tables, candidates=list(tables),
            keywords=list(selection.keywords), metadata=metadata, reasoning=selection.reasoning,
            trace=f"{TRACE_HEADER}\n{selection.reasoning}", tokens=0, selector=SelectorKind.STORED,
            selection_trace=TRACE_HEADER,
        )
        agent = P12State()
        if selection.selection_plan:
            agent.selection_plan = dict(selection.selection_plan)
            agent.selection_plan_source = "stored"
        agent.confirmed_tables = list(tables)
        agent.selection_reasoning = selection.reasoning
        state.selection_state = agent
        state.selection_states.append(agent)
        state.selected, state.keywords, state.solr_meta = tables, list(selection.keywords), metadata
        state.reasoning, state.trace = outcome.reasoning, outcome.trace
        run.recorder.discovery_round(
            outcome, state=state, selection_record=selection_record, keywords_rejected=False,
            attempt=attempt, elapsed=round(time.monotonic() - started, 6), retrieval_runs=run.retrieval_runs,
        )
        return outcome, False
