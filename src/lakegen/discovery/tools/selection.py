"""The selection tools of the unified agent: confirm the tables with a plan, reject them with a reason, draft the plan."""

import json

from lakegen.agent_tools.requirement_ledger import (
    build_requirement_ledger,
    requirement_ledger_block_message,
)
from lakegen.agent_tools.tools_p2 import (
    MIN_BAN_JUSTIFICATION_CHARS,
    _temporal_coverage_issue,
)
from lakegen.core.table_io import read_table
from lakegen.discovery.tools.plan_models import (
    SemanticAnalysisPlan,
    _derive_output_columns,
    _normalize_requirement_coverage,
    _normalize_semantic_plan,
    _resolve_partition_table_aliases,
    compile_semantic_plan_draft,
)


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
        normalized_tables = list(dict.fromkeys(str(table).strip() for table in tables))
        if not normalized_tables:
            raise ValueError("Selection blocked: at least one inspected table is required.")
        visible_candidates = self.state.all_candidates[
            : self.state.visible_candidate_count
        ]
        unknown = [table for table in normalized_tables if table not in visible_candidates]
        if unknown:
            raise ValueError(
                "Selection blocked: only currently visible candidates may be "
                f"selected. Unknown table(s): {unknown}."
            )

        inspected = {table.casefold() for table in self.state.inspected_candidates()}
        uninspected = [
            table for table in normalized_tables if table.casefold() not in inspected
        ]
        if uninspected:
            raise ValueError(
                "Selection blocked: inspect_columns is mandatory for every selected "
                f"table. Inspect {uninspected}, verify requirement and temporal "
                "coverage, then confirm again."
            )

        selection_key = tuple(sorted(table.casefold() for table in normalized_tables))
        if selection_key in self.state.rejected_selections:
            raise ValueError(
                "Selection blocked: this exact table combination was already "
                "rejected by the full-context coder. Select at least one different "
                "table or reject the keywords if no alternative exists."
            )

        coverage_issue = _temporal_coverage_issue(
            self.question,
            normalized_tables,
            self.state.inspection_cache,
        )
        if coverage_issue:
            raise ValueError(
                "Selection blocked by temporal validation: " + coverage_issue + "."
            )
        for table in normalized_tables:
            if table not in self.state.proposed_tables:
                self.state.proposed_tables.append(table)

        requirement_coverage = _normalize_requirement_coverage(
            requirement_coverage or {}, normalized_tables
        )
        table_roles = table_roles or {}
        uncovered_requirements = list(dict.fromkeys(
            str(item).strip() for item in (uncovered_requirements or [])
            if str(item).strip()
        ))
        normalized_alternatives: dict[str, dict[str, object]] = {}
        for raw_table, raw_evidence in (alternatives_rejected or {}).items():
            table = str(raw_table).strip()
            if not table:
                continue
            if isinstance(raw_evidence, dict):
                matched = list(dict.fromkeys(
                    str(item).strip()
                    for item in raw_evidence.get("matched_requirements", [])
                    if str(item).strip()
                ))
                missing = str(raw_evidence.get("missing_requirement", "")).strip()
            else:
                matched = []
                missing = str(raw_evidence).strip()
            normalized_alternatives[table] = {
                "matched_requirements": matched,
                "missing_requirement": missing,
            }
        alternatives_rejected = normalized_alternatives
        selected_set = set(normalized_tables)
        advisories: list[str] = []

        contract_first = semantic_plan is not None
        if isinstance(semantic_plan, SemanticAnalysisPlan):
            semantic_plan = semantic_plan.model_dump()
        if contract_first:
            if isinstance(semantic_plan, dict) and not semantic_plan.get("table_roles"):
                semantic_plan = {**semantic_plan, "table_roles": dict(table_roles)}
            if isinstance(semantic_plan, dict):
                semantic_plan = _normalize_semantic_plan(
                    semantic_plan, normalized_tables
                )
                if not semantic_plan.get("output_columns"):
                    semantic_plan["output_columns"] = _derive_output_columns(semantic_plan)
            try:
                semantic_plan = SemanticAnalysisPlan.model_validate(semantic_plan).model_dump()
            except Exception as exc:
                self.state.plan_failure = str(exc)
                raise ValueError(
                    "Selection blocked: invalid semantic_plan: "
                    f"{exc}\nCorrect the listed fields and call "
                    "confirm_unified_selection again. Do not return the corrected "
                    "JSON only as assistant text."
                ) from exc

            schema_by_table: dict[str, set[str]] = {}
            for table in normalized_tables:
                try:
                    schema_by_table[table] = {
                        str(column) for column in read_table(
                            self.csv_dir / table, nrows=0
                        ).columns
                    }
                except Exception as exc:
                    raise ValueError(
                        f"Selection blocked: cannot validate schema for {table}: {exc}"
                    ) from exc
            semantic_plan = _resolve_partition_table_aliases(
                semantic_plan, normalized_tables, table_roles,
                schema_by_table, combination_strategy,
            )
            invalid_bindings: list[str] = []
            bindings = [
                *semantic_plan["filters"],
                *semantic_plan["temporal_filters"],
                *semantic_plan["dimensions"],
                *semantic_plan["measures"],
            ]
            for binding in bindings:
                table = str(binding.get("table", ""))
                columns = binding.get("columns", [binding.get("column", "")])
                if table not in selected_set:
                    invalid_bindings.append(f"{table}: table is not selected")
                    continue
                for column in columns:
                    if str(column) not in schema_by_table.get(table, set()):
                        invalid_bindings.append(f"{table}.{column}: column not found")
                if not str(binding.get("evidence", "")).strip():
                    invalid_bindings.append(f"{table}: missing binding evidence")
            if invalid_bindings:
                raise ValueError(
                    "Selection blocked: semantic bindings are not supported by the "
                    "selected schemas: " + "; ".join(invalid_bindings)
                )

        missing_roles = [table for table in normalized_tables if not table_roles.get(table)]
        if missing_roles:
            advisories.append(
                "Selected table(s) without an explicit role: " + ", ".join(missing_roles)
            )

        role_extras = [table for table in table_roles if table not in selected_set]
        if role_extras:
            advisories.append(
                "Table role(s) refer to unselected tables: " + ", ".join(role_extras)
            )

        covered_tables: set[str] = set()
        malformed_requirements: list[str] = []
        for requirement, evidence in requirement_coverage.items():
            if not isinstance(evidence, dict):
                malformed_requirements.append(requirement)
                continue
            table = str(evidence.get("table", "")).strip()
            columns = evidence.get("columns", [])
            if table in selected_set:
                covered_tables.add(table)
            if table not in selected_set or not isinstance(columns, list) or not columns:
                malformed_requirements.append(requirement)
        if not requirement_coverage:
            advisories.append("No explicit requirement coverage was supplied.")
        elif malformed_requirements:
            advisories.append(
                "Requirement coverage lacks selected-table/column evidence for: "
                + ", ".join(malformed_requirements)
            )

        uncovered_tables = [table for table in normalized_tables if table not in covered_tables]
        if uncovered_tables:
            advisories.append(
                "Selected table(s) cover no explicit requirement: "
                + ", ".join(uncovered_tables)
            )

        if len(normalized_tables) > 1 and combination_strategy == "single_table":
            advisories.append(
                "Multiple tables were selected but the combination strategy is single_table."
            )
        elif len(normalized_tables) == 1 and combination_strategy != "single_table":
            advisories.append(
                "One table was selected but the combination strategy is "
                f"{combination_strategy}."
            )
        if uncovered_requirements:
            advisories.append(
                "Selection was confirmed with requirements still marked uncovered: "
                + ", ".join(uncovered_requirements)
            )
        if len(alternatives_rejected) > 2:
            advisories.append(
                "More than two rejected alternatives were supplied; keep only the "
                "strongest inspected alternatives in future confirmations."
            )
        inspected_names = set(self.state.inspected_candidates())
        unsupported_alternatives = [
            table for table in alternatives_rejected
            if table not in inspected_names
        ]
        if unsupported_alternatives:
            advisories.append(
                "Rejected alternative(s) were not inspected: "
                + ", ".join(unsupported_alternatives)
            )
        vague_missing = {
            "", "less relevant", "not relevant", "not needed", "weaker match",
            "lower ranked", "redundant", "inferior",
        }
        for table, evidence in alternatives_rejected.items():
            matched = evidence.get("matched_requirements", [])
            missing = str(evidence.get("missing_requirement", "")).casefold()
            if len(matched) >= 2 and missing in vague_missing:
                advisories.append(
                    f"{table} matches multiple essential requirements but no "
                    "concrete missing requirement justifies excluding it. Reconsider "
                    "including or preferring this inspected alternative."
                )

        selection_blockers: list[str] = []
        if missing_roles:
            selection_blockers.append("every selected table needs an explicit role")
        if not requirement_coverage:
            selection_blockers.append("requirement_coverage is required")
        if malformed_requirements:
            selection_blockers.append(
                "requirement coverage must bind selected tables and columns"
            )
        if uncovered_tables:
            selection_blockers.append(
                "every selected table must support at least one requirement"
            )
        if uncovered_requirements:
            selection_blockers.append(
                "uncovered_requirements must be resolved before confirmation"
            )
        if (
            len(normalized_tables) > 1
            and combination_strategy == "single_table"
        ) or (
            len(normalized_tables) == 1
            and combination_strategy != "single_table"
        ):
            selection_blockers.append(
                "combination_strategy must match the selected table count"
            )
        if contract_first and selection_blockers:
            raise ValueError(
                "Selection blocked by contract-first validation: "
                + "; ".join(selection_blockers)
                + ". Inspect/expand the existing ranked candidates and confirm again."
            )

        normalized_requirements = dict(requirements or {})
        coder_brief = self._build_coder_brief(
            normalized_tables,
            requirement_coverage,
            normalized_requirements,
            combination_strategy,
            table_roles,
            semantic_plan if isinstance(semantic_plan, dict) else None,
        )
        requirement_ledger = self._build_requirement_ledger(
            requirement_coverage,
            normalized_requirements,
            uncovered_requirements,
            semantic_plan if isinstance(semantic_plan, dict) else None,
        )
        selection_plan = {
            "requirement_coverage": requirement_coverage,
            "table_roles": table_roles,
            "combination_strategy": combination_strategy,
            "uncovered_requirements": uncovered_requirements,
            "alternatives_rejected": alternatives_rejected,
            "requirements": normalized_requirements,
            "requirement_ledger": requirement_ledger,
            "coder_brief": coder_brief,
            **({"semantic_plan": semantic_plan} if contract_first else {}),
        }
        block_message = requirement_ledger_block_message(
            requirement_ledger, normalized_tables
        )
        if block_message:
            # Kept so an attempt that ends without a terminal tool call falls
            # back to the architect's own bindings, not to bare top candidates.
            self.state.blocked_selection = {
                "tables": list(normalized_tables),
                "reasoning": reasoning,
                "block_message": block_message,
                "selection_plan": selection_plan,
            }
            raise ValueError(block_message)
        self.state.selection_plan = selection_plan
        self.state.plan_failure = None
        self.state.selection_plan_source = (
            "confirm_unified_selection" if contract_first else "selection_only"
        )
        self.state.confirmed_tables = list(normalized_tables)
        self.state.selection_reasoning = reasoning
        self.state.selection_requirements = normalized_requirements
        self.state.selection_advisories = advisories

        final_tables = ", ".join(normalized_tables)

        dati_uscita = {
            "tables": final_tables,
            "reasoning": reasoning,
            "selection_plan": self.state.selection_plan,
            "advisories": advisories,
        }
        return f"FINAL_PAYLOAD: {json.dumps(dati_uscita)}"

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
