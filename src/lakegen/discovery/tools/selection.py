"""The selection tools of the unified agent: confirm the tables with a plan, reject them with a reason, draft the plan."""


from lakegen.discovery.ledger import (
    build_requirement_ledger,
)
from lakegen.discovery.tools import confirmation
from lakegen.discovery.tools.judge import (
    MIN_BAN_JUSTIFICATION_CHARS,
)
from lakegen.discovery.tools.plan_models import (
    SemanticAnalysisPlan,
    compile_semantic_plan_draft,
)
from lakegen.lakes.table_io import read_table


class SelectionMixin:
    """The selection methods of ``Phase12ToolsManager``."""

    def confirm_unified_selection(
        self,
        reasoning: str,
        tables: list[str],
        requirement_coverage: dict[str, dict[str, object]] | None = None,
        table_roles: dict[str, str] | None = None,
        combination_strategy: str = "single_table",
        uncovered_requirements: list[str] | None = None,
        alternatives_rejected: dict[str, dict[str, object] | str] | None = None,
        requirements: dict[str, object] | None = None,
        semantic_plan: dict[str, object] | SemanticAnalysisPlan | None = None,
    ) -> str:
        """
        Confirm the selected tables once inspection has verified that they
        cover every essential requirement. Ends discovery.
        """
        selected = confirmation.admissible_tables(self.state, self.question, tables)
        proposal = confirmation.proposal(
            selected, reasoning, requirement_coverage=requirement_coverage, table_roles=table_roles,
            combination_strategy=combination_strategy, uncovered_requirements=uncovered_requirements,
            alternatives_rejected=alternatives_rejected, requirements=requirements,
        )
        if semantic_plan is not None:
            proposal.semantic_plan = confirmation.bound_plan(self.state, self.csv_dir, proposal, semantic_plan)
        advisories, blockers = confirmation.review(self.state, proposal)
        proposal.advisories = advisories
        if proposal.contract_first and blockers:
            raise ValueError(
                "Selection blocked by contract-first validation: "
                + "; ".join(blockers)
                + ". Inspect/expand the existing ranked candidates and confirm again."
            )
        return confirmation.commit(self, proposal)

    def reject_unified_selection(
        self, reasoning: str, suggestion: str, ban_tables: dict[str, str] | None = None
    ) -> str:
        """
        End this attempt when the candidates cannot cover the question's
        essential requirements. Inspected candidates carry over to the next
        attempt unless banned in ban_tables.
        """
        normalized_reason = " ".join(str(reasoning).casefold().split())
        text_dtype_claim = any(
            marker in normalized_reason
            for marker in ("string", "text", "object dtype", "object-typed")
        )
        numeric_operation_claim = any(
            marker in normalized_reason
            for marker in (
                "numeric", "sum", "summed", "average", "averaged",
                "aggregate", "aggregation", "convert", "conversion",
            )
        )
        type_only_rejection = any(
            marker in normalized_reason
            for marker in (
                "cannot", "can't", "not supported", "unsupported",
                "not possible", "unable",
            )
        )
        runtime_unparseable_claim = (
            any(marker in normalized_reason for marker in (
                "sample values", "sampled values", "observed values",
                "runtime values",
            ))
            and any(marker in normalized_reason for marker in (
                "unparseable", "cannot be parsed", "no numeric values",
                "no parseable values",
            ))
        )
        if (
            text_dtype_claim
            and numeric_operation_claim
            and type_only_rejection
            and not runtime_unparseable_claim
        ):
            raise ValueError(
                "Selection rejection blocked: a numeric or monetary measure stored "
                "as text is convertible by the Coder and is not evidence that the "
                "dataset is insufficient. Confirm the table when the required column "
                "exists, unless inspected runtime values specifically prove that the "
                "measure cannot be parsed."
            )
        inspected = self.state.inspected_candidates()
        by_fold = {table.casefold(): table for table in inspected}
        unjustified = [
            table for table, justification in (ban_tables or {}).items()
            if table.casefold() in by_fold
            and len(str(justification).strip()) < MIN_BAN_JUSTIFICATION_CHARS
        ]
        if unjustified:
            raise ValueError(
                "Selection blocked: ban_tables must map each highly-irrelevant "
                "candidate to the concrete evidence proving it satisfies "
                "nothing, not a bare name. Missing or too-short justification "
                f"for: {', '.join(unjustified)}. Give it a real justification "
                "or drop it from ban_tables."
            )
        # A table this attempt already proposed as a source is not "proven to
        # satisfy nothing": a live architect, blocked three times confirming
        # its primary source, ended by writing that same table into its bans.
        proposed = {table.casefold() for table in self.state.proposed_tables}
        requested = [
            by_fold[table.casefold()]
            for table in (ban_tables or {})
            if table.casefold() in by_fold
        ]
        banned = [table for table in requested if table.casefold() not in proposed]
        protected = [table for table in requested if table.casefold() in proposed]
        self.state.rejection_skip_tables = banned
        self.state.rejection_keep_tables = [
            table for table in inspected if table not in banned
        ]
        self._record_retrieval_memory_event(
            outcome="insufficient_coverage",
            terms=list(self.state.used_keywords),
            tables=list(inspected),
            reason=reasoning,
            suggestion=suggestion,
            evidence={
                table: justification
                for table, justification in (ban_tables or {}).items()
                if table.casefold() not in proposed
            },
        )
        response = f"REJECT_KEYWORDS: {reasoning}\nSuggestion: {suggestion}"
        if protected:
            response += (
                "\nNot banned, kept for the next attempt (proposed as a selected "
                "source earlier in this attempt): " + ", ".join(protected)
            )
        return response

    def _build_requirement_ledger(
        self,
        coverage: dict[str, dict[str, object]],
        requirements: dict[str, object],
        uncovered: list[str],
        semantic_plan: dict[str, object] | None,
    ) -> list[dict[str, object]]:
        """Create one compact mode-independent, non-prescriptive checklist."""
        return build_requirement_ledger(
            self.question, coverage, requirements, uncovered, semantic_plan,
            inspections=self.state.inspection_cache,
        )

    def submit_semantic_plan_draft(self, draft: dict[str, object]) -> str:
        """Compile and validate a compact benchmark-blind semantic draft."""
        self.state.semantic_planner_attempts += 1
        if self.state.semantic_planner_attempts > 2:
            raise ValueError("semantic planner correction limit reached (2)")
        tables = list(self.state.confirmed_tables)
        if not tables:
            raise ValueError("table selection must be confirmed before semantic planning")
        roles = dict(self.state.selection_plan.get("table_roles") or {})
        schema_by_table = {
            table: {str(column) for column in read_table(self.csv_dir / table, nrows=0).columns}
            for table in tables
        }
        try:
            compiled = compile_semantic_plan_draft(
                draft, tables, roles, schema_by_table
            )
            self.state.semantic_draft = dict(draft)
            return self.confirm_unified_selection(
                self.state.selection_reasoning,
                tables,
                requirement_coverage=dict(
                    self.state.selection_plan.get("requirement_coverage") or {}
                ),
                table_roles=roles,
                combination_strategy=str(
                    self.state.selection_plan.get("combination_strategy") or "single_table"
                ),
                uncovered_requirements=list(
                    self.state.selection_plan.get("uncovered_requirements") or []
                ),
                alternatives_rejected=dict(
                    self.state.selection_plan.get("alternatives_rejected") or {}
                ),
                requirements=dict(self.state.selection_requirements),
                semantic_plan=compiled,
            )
        except Exception as exc:
            self.state.plan_failure = str(exc)
            raise ValueError(
                "Semantic draft validation failed. Correct only the reported fields "
                f"and call submit_semantic_plan_draft again. Error: {exc}"
            ) from exc
