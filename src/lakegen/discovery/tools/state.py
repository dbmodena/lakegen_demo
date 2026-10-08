"""What one unified discovery round knows: candidates, inspections, bans, and what carries to the next."""

from typing import Mapping

from lakegen.core.types import SolrMetadata


class P12State:
    """State tracker for the Phase 1 & 2 unified agent."""
    def __init__(self):
        self.all_candidates: list[str] = []
        self.solr_meta: SolrMetadata = {}
        self.used_keywords: list[str] = []
        self.keyword_history: list[list[str]] = []
        # Minimal antichain of AND keyword combinations that returned no local
        # candidates.  If {a} fails, keeping the older {a, b} adds no
        # information: every future AND query containing a is already doomed.
        self.failed_keyword_combinations: list[frozenset[str]] = []
        # Same antichain for AND combinations whose every local match is a
        # table banned for this question (excluded_tables). Question-scoped:
        # never persisted to keyword memory, since those tables may be exactly
        # what another question needs.
        self.banned_table_keyword_combinations: list[frozenset[str]] = []
        self.best_ranks: dict[str, int] = {}
        self.candidate_scores: dict[str, float] = {}
        self.search_cache: dict[tuple[str, ...], str] = {}
        self.search_attempts: list[dict[str, object]] = []
        # search_tables calls that reached retrieval. One call can run several
        # retrievals (the split keyword searches), each tagged with its call.
        self.search_calls = 0
        # Retrieval representation (e.g. embedding) generation failure: blocks
        # re-running search_tables, since retrieval itself is what's broken.
        self.semantic_failure: str | None = None
        # semantic_plan / draft validation failure from confirm_unified_selection
        # or submit_semantic_plan_draft. Unrelated to retrieval, so it must never
        # gate search_tables -- only a bad plan, not the search, needs retrying.
        self.plan_failure: str | None = None
        self.initial_stall_reason: str | None = None
        self.recovery_started = False
        self.recovery_stop_reason: str | None = None
        self.selection_plan_source = "none"
        self.confirmed_tables: list[str] = []
        self.selection_reasoning = ""
        self.selection_requirements: dict[str, object] = {}
        self.semantic_draft: dict[str, object] = {}
        self.semantic_planner_attempts = 0
        self.inspection_cache: dict[str, str] = {}
        self.inspection_counts: dict[str, int] = {}
        self.visible_candidate_count = 0
        self.expansion_count = 0
        self.expansion_requirements: list[str] = []
        self.rejected_selections: set[tuple[str, ...]] = set()
        self.selection_plan: dict[str, object] = {}
        self.selection_advisories: list[str] = []
        # Latest confirm_unified_selection blocked by the requirement ledger:
        # its tables, reasoning, block message and would-be selection plan.
        self.blocked_selection: dict[str, object] = {}
        # Tables named in any confirm_unified_selection call that passed the
        # visibility and inspection checks. The architect vouched for these as
        # sources, so a later rejection cannot ban them as proven irrelevant.
        self.proposed_tables: list[str] = []
        # Latest confirm_unified_selection error and how many were blocked,
        # for the state board and the repeated-block guard.
        self.last_confirm_error: str | None = None
        self.confirm_blocks = 0
        self.rejection_keep_tables: list[str] = []
        self.rejection_skip_tables: list[str] = []
        # Question-scoped retrieval evidence persisted by the service after a
        # completed agent round. It is intentionally separate from the hard
        # zero-result AND banlist above.
        self.retrieval_memory_events: list[dict[str, object]] = []
        # Cross-round memory, seeded by the caller before this round starts
        # (mirrors DIVIDED's excluded_tables/carried_tables in service.py).
        # excluded_tables: proven-insufficient candidates from a prior round,
        # in this question, filtered out of fresh retrieval entirely.
        # carried_tables: candidates a prior round already verified satisfy
        # part of the question; re-surfaced here regardless of whether this
        # round's retrieval finds them again.
        self.excluded_tables: set[str] = set()
        self.carried_tables: list[str] = []
        self.carried_metadata: SolrMetadata = {}

    def search_call_outcomes(self) -> list[bool]:
        """Per search_tables call, whether any of its retrievals found a local
        table. The search budget counts calls, not retrievals, so a keyword
        cascade that falls through several empty sets spends one attempt."""
        found: dict[object, bool] = {}
        for index, attempt in enumerate(self.search_attempts):
            call = attempt.get("call", ("retrieval", index))
            found[call] = found.get(call, False) or bool(attempt.get("current_candidates"))
        return list(found.values())

    def inspected_candidates(self) -> list[str]:
        """Return successfully inspected candidates in retrieval order."""
        inspected: list[str] = []
        for candidate in self.all_candidates:
            filename: str | None = None
            if isinstance(candidate, str):
                filename = candidate.strip()
            elif isinstance(candidate, Mapping):
                filename = next((
                    str(candidate[key]).strip()
                    for key in ("file", "filename", "dataset")
                    if isinstance(candidate.get(key), str)
                    and str(candidate[key]).strip()
                ), None)
            if not filename:
                continue
            cached = self.inspection_cache.get(filename.casefold())
            if isinstance(cached, str) and not cached.startswith("Error:"):
                inspected.append(filename)
        return list(dict.fromkeys(inspected))


# ============================================================================
# keyword_search="decompose_preview" (DiscoveryConfig): split one question into
# its per-table distinguishing-detail search phrases, preview a few real
# AND-word candidates per phrase (with match counts and top matches) and let
# the model pick before any of them is actually run. Ported from
# experiments/retrieval_lab's decompose_tuned_preview_noacronym, the strongest
# keyword-search arm found there -- a real, if modest, improvement over a
# single free-form search at every scale tested (84/50/100/249 questions).
# ============================================================================

# Generic function words, stripped before a phrase's content words are counted
# against the index.
