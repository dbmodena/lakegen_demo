"""What one discovery round knows: the candidates it searched, the selection it made, and what carries on."""

from __future__ import annotations

from typing import Any

from lakegen.datastructures.types import SolrMetadata


class DiscoveryState:
    """The round's record, written by the discovery phase and read by the pipeline and the recorder.

    Attributes:
        all_candidates: The candidate pool, best first; the first ``visible_candidate_count``
            were shown to the agent before any expansion.
        solr_meta: Each candidate's catalogue entry.
        used_keywords: The words of the first search that found candidates.
        search_attempts: One record per retrieval the round ran.
        expansion_count: How many hidden candidates' reveals the agent spent.
        visible_candidate_count: How many candidates the agent saw first.
        selection_plan: The confirmed selection's plan for the coder.
        selection_advisories: What the selector found weak about the confirmed selection.
        selection_plan_source: What made the plan (``memgpt_selector``, ``stored``).
        confirmed_tables: The tables of the confirmed selection.
        selection_reasoning: The confirmed selection's reasoning.
        rejection_keep_tables: On a rejection, the inspected tables worth carrying to the next round.
        rejection_skip_tables: On a rejection, the tables proven unusable.
        excluded_tables: Tables earlier rounds proved unusable, casefolded; retrieval leaves them out.
        carried_tables: Tables earlier rounds found useful; retrieval puts them first.
        carried_metadata: The catalogue entries of the carried tables.
    """

    def __init__(self) -> None:
        self.all_candidates: list[str] = []
        self.solr_meta: SolrMetadata = {}
        self.used_keywords: list[str] = []
        self.search_attempts: list[dict[str, Any]] = []
        self.expansion_count = 0
        self.visible_candidate_count = 0
        self.selection_plan: dict[str, Any] = {}
        self.selection_advisories: list[str] = []
        self.selection_plan_source = "none"
        self.confirmed_tables: list[str] = []
        self.selection_reasoning = ""
        self.rejection_keep_tables: list[str] = []
        self.rejection_skip_tables: list[str] = []
        self.excluded_tables: set[str] = set()
        self.carried_tables: list[str] = []
        self.carried_metadata: SolrMetadata = {}
