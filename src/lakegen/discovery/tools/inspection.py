"""The inspection tools of the unified agent: a candidate's columns, the hidden candidates, a join or union check."""



from lakegen.agent_tools.tools_p2 import (
    _check_join_union,
    _inspect_columns,
    _rank_for_missing_requirements,
    _requirement_terms,
)
from lakegen.phases.utils import format_candidate_context


class InspectionMixin:
    """The inspection methods of ``Phase12ToolsManager``."""

    def inspect_columns(
        self,
        file_name: str | None = None,
        filename: str | None = None,
        candidate_number: int | None = None,
    ) -> str:
        """
        Return a compact profile of one visible candidate: row count, min/max
        coverage of its date columns, column types, and sample values of
        low-cardinality columns. Compare a requested date or period with the
        reported coverage before selecting the table. Pass candidate_number,
        the "Candidate N" label printed above each candidate, rather than a
        filename: generated filenames are long and easy to mistype. Asking
        for the same file again points back to the profile already shown.
        """
        visible_candidates = self.state.all_candidates[
            : self.state.visible_candidate_count
        ]
        if candidate_number is not None:
            if not (1 <= candidate_number <= len(visible_candidates)):
                return (
                    f"Error: candidate_number must be between 1 and "
                    f"{len(visible_candidates)} (the currently visible candidates)."
                )
            name = visible_candidates[candidate_number - 1]
        else:
            name = file_name or filename
            if not name:
                return (
                    "Error: candidate_number (preferred) or file_name/filename "
                    "is required."
                )
            name = name.strip()
        key = name.casefold()
        if name not in visible_candidates:
            return (
                f"Error: {name} is not currently visible. Inspect only candidates "
                "already shown by search_tables or expand_candidates."
            )
        attempted_candidates = len(self.state.inspection_counts)
        if key not in self.state.inspection_cache:
            current_limit = (
                self.discovery.initial_shortlist_size
                if self.state.expansion_count == 0
                else self.discovery.max_inspected_candidates
            )
            if attempted_candidates >= current_limit:
                if (
                    self.state.expansion_count == 0
                    and self.state.visible_candidate_count < len(self.state.all_candidates)
                ):
                    return (
                        "Inspection blocked: the initial shortlist is limited to "
                        f"{self.discovery.initial_shortlist_size} "
                        "candidates. If coverage is incomplete, call expand_candidates "
                        "before inspecting another candidate."
                    )
                return (
                    "Inspection blocked: at most "
                    f"{self.discovery.max_inspected_candidates} distinct candidates "
                    "may be inspected for this request."
                )
        count = self.state.inspection_counts.get(key, 0) + 1
        self.state.inspection_counts[key] = count
        if count > self.discovery.max_inspections_per_file:
            return (
                f"Inspection skipped: {name} has already been inspected "
                f"{self.discovery.max_inspections_per_file} times. Use the cached schema and "
                "continue with confirm_unified_selection."
            )
        if key not in self.state.inspection_cache:
            self.state.inspection_cache[key] = _inspect_columns(self.csv_dir, name)
            return self.state.inspection_cache[key]
        if count == 1:
            # First time this schema is actually shown this round -- whether
            # freshly cached above or carried over from a prior round -- so
            # the full profile earns its place in context.
            return (
                f"Cached inspection (attempt {count}/"
                f"{self.discovery.max_inspections_per_file}):\n"
                + self.state.inspection_cache[key]
            )
        # Already shown once this round: point back instead of re-sending a
        # profile the agent has already seen, to keep context lean.
        candidate_label = f"Candidate {visible_candidates.index(name) + 1}"
        return (
            f"Cached inspection (attempt {count}/"
            f"{self.discovery.max_inspections_per_file}): already shown above "
            f"for {candidate_label}. Reuse that schema; do not request it again."
        )

    def expand_candidates(self, missing_requirements: str) -> str:
        """Reveal hidden candidates whose metadata best covers one concrete
        missing measure, dimension, filter, period, or join key that the
        inspected candidates lack. It does not re-run retrieval. Afterwards,
        inspect only candidates whose metadata could fill that gap.
        """
        if not self.state.inspected_candidates():
            return (
                "Expansion blocked: inspect at least one plausible visible candidate "
                "before requesting more results."
            )
        requirements = _requirement_terms(missing_requirements)
        if not requirements:
            return (
                "Expansion blocked: provide concrete missing requirements such as "
                "a measure, dimension, period, filter, or join key."
            )
        if self.state.expansion_count >= self.discovery.max_expansions:
            return (
                "Expansion limit reached. Do not call expand_candidates again "
                "or search_tables again; select among the inspected candidates."
            )
        if self.state.visible_candidate_count >= len(self.state.all_candidates):
            return "No additional candidates are available."
        start = self.state.visible_candidate_count
        hidden = self.state.all_candidates[start:]
        newly_visible = _rank_for_missing_requirements(
            hidden, self.state.solr_meta, requirements, self.discovery.expansion_size
        )
        if not newly_visible:
            self.state.expansion_count += 1
            self.state.expansion_requirements = requirements
            return (
                "No hidden candidate has metadata matching the missing requirements. "
                "Do not call expand_candidates or search_tables again; select or "
                "reject using the inspected evidence."
            )
        selected = set(newly_visible)
        self.state.all_candidates[start:] = [
            *newly_visible,
            *(candidate for candidate in hidden if candidate not in selected),
        ]
        self.state.visible_candidate_count = start + len(newly_visible)
        self.state.expansion_count += 1
        self.state.expansion_requirements = requirements
        remaining = len(self.state.all_candidates) - self.state.visible_candidate_count
        next_step = (
            f"{remaining} ranked candidates remain hidden, but the single guided "
            "expansion has been used. Do not call expand_candidates again."
            if remaining
            else (
                "All ranked candidates are now visible. Do not call "
                "expand_candidates again."
            )
        )
        return (
            "Guided expansion for missing requirements: "
            + ", ".join(requirements)
            + f"\nRevealed {len(newly_visible)} best-matching hidden candidates "
            "(original retrieval ranks are preserved in metadata):\n"
            + format_candidate_context(
                newly_visible, self.state.solr_meta, start_rank=start + 1
            )
            + f"\n\n{next_step}"
        )

    def check_join_union(self, file_name_1: str, file_name_2: str) -> str:
        """
        Check whether two tables join or union with each other. Reports the join
        key columns when they join, the aligned columns when they union, or that
        they neither join nor union.
        """
        return _check_join_union(self.csv_dir, file_name_1, file_name_2)
