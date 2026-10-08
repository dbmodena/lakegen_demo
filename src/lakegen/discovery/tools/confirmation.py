"""Confirming the unified agent's selection, step by step.

A confirmation is admitted (the tables exist, were inspected, were not
rejected before and cover the question's period), normalised into a
``Proposal``, bound to the tables' real columns when it carries a semantic
plan, reviewed for what is weak about it (advisories, which block a
contract-first confirmation), and committed as the selection plan the coder
starts from. Each step raises ``ValueError`` with the message the agent
reads when the selection is blocked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lakegen.agent_tools.requirement_ledger import requirement_ledger_block_message
from lakegen.agent_tools.tools_p2 import _temporal_coverage_issue
from lakegen.core.table_io import read_table
from lakegen.discovery.tools.plan_models import (
    SemanticAnalysisPlan,
    _derive_output_columns,
    _normalize_requirement_coverage,
    _normalize_semantic_plan,
    _resolve_partition_table_aliases,
)
from lakegen.discovery.tools.state import P12State

if TYPE_CHECKING:
    from lakegen.discovery.tools.manager import Phase12ToolsManager

VAGUE_MISSING = frozenset({
    "", "less relevant", "not relevant", "not needed", "weaker match",
    "lower ranked", "redundant", "inferior",
})
"""Reasons that do not justify excluding an alternative that matched several requirements."""


@dataclass
class Proposal:
    """One confirmation's arguments, normalised."""

    tables: list[str]
    reasoning: str
    requirement_coverage: dict[str, Any]
    table_roles: dict[str, str]
    combination_strategy: str
    uncovered_requirements: list[str]
    alternatives_rejected: dict[str, dict[str, Any]]
    requirements: dict[str, Any]
    semantic_plan: dict[str, Any] | None = None
    """The validated, bound plan of a contract-first confirmation."""
    advisories: list[str] = field(default_factory=list)

    @property
    def contract_first(self) -> bool:
        return self.semantic_plan is not None

    @property
    def selected(self) -> set[str]:
        return set(self.tables)


def admissible_tables(state: P12State, question: str, tables: list[str]) -> list[str]:
    """The tables a confirmation may name, or why it may not; they join the proposed ones."""
    normalized = list(dict.fromkeys(str(table).strip() for table in tables))
    if not normalized:
        raise ValueError("Selection blocked: at least one inspected table is required.")
    visible = state.all_candidates[: state.visible_candidate_count]
    unknown = [table for table in normalized if table not in visible]
    if unknown:
        raise ValueError(
            "Selection blocked: only currently visible candidates may be "
            f"selected. Unknown table(s): {unknown}."
        )
    inspected = {table.casefold() for table in state.inspected_candidates()}
    uninspected = [table for table in normalized if table.casefold() not in inspected]
    if uninspected:
        raise ValueError(
            "Selection blocked: inspect_columns is mandatory for every selected "
            f"table. Inspect {uninspected}, verify requirement and temporal "
            "coverage, then confirm again."
        )
    if tuple(sorted(table.casefold() for table in normalized)) in state.rejected_selections:
        raise ValueError(
            "Selection blocked: this exact table combination was already "
            "rejected by the full-context coder. Select at least one different "
            "table or reject the keywords if no alternative exists."
        )
    coverage_issue = _temporal_coverage_issue(question, normalized, state.inspection_cache)
    if coverage_issue:
        raise ValueError("Selection blocked by temporal validation: " + coverage_issue + ".")
    for table in normalized:
        if table not in state.proposed_tables:
            state.proposed_tables.append(table)
    return normalized


def _normalised_alternatives(raw: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Each rejected alternative with what it matched and what it lacked."""
    alternatives: dict[str, dict[str, Any]] = {}
    for raw_table, raw_evidence in (raw or {}).items():
        table = str(raw_table).strip()
        if not table:
            continue
        if isinstance(raw_evidence, dict):
            matched = list(dict.fromkeys(
                str(item).strip() for item in raw_evidence.get("matched_requirements", []) if str(item).strip()
            ))
            missing = str(raw_evidence.get("missing_requirement", "")).strip()
        else:
            matched, missing = [], str(raw_evidence).strip()
        alternatives[table] = {"matched_requirements": matched, "missing_requirement": missing}
    return alternatives


def proposal(
    tables: list[str],
    reasoning: str,
    *,
    requirement_coverage: dict[str, Any] | None,
    table_roles: dict[str, str] | None,
    combination_strategy: str,
    uncovered_requirements: list[str] | None,
    alternatives_rejected: dict[str, Any] | None,
    requirements: dict[str, Any] | None,
) -> Proposal:
    """The confirmation's arguments, normalised; the plan is bound separately."""
    return Proposal(
        tables=tables,
        reasoning=reasoning,
        requirement_coverage=_normalize_requirement_coverage(requirement_coverage or {}, tables),
        table_roles=table_roles or {},
        combination_strategy=combination_strategy,
        uncovered_requirements=list(dict.fromkeys(
            str(item).strip() for item in (uncovered_requirements or []) if str(item).strip()
        )),
        alternatives_rejected=_normalised_alternatives(alternatives_rejected),
        requirements=dict(requirements or {}),
    )


def _schemas(csv_dir: Path, tables: list[str]) -> dict[str, set[str]]:
    schemas: dict[str, set[str]] = {}
    for table in tables:
        try:
            schemas[table] = {str(column) for column in read_table(csv_dir / table, nrows=0).columns}
        except Exception as exc:
            raise ValueError(f"Selection blocked: cannot validate schema for {table}: {exc}") from exc
    return schemas


def _invalid_bindings(plan: dict[str, Any], selected: set[str], schemas: dict[str, set[str]]) -> list[str]:
    invalid: list[str] = []
    bindings = [*plan["filters"], *plan["temporal_filters"], *plan["dimensions"], *plan["measures"]]
    for binding in bindings:
        table = str(binding.get("table", ""))
        columns = binding.get("columns", [binding.get("column", "")])
        if table not in selected:
            invalid.append(f"{table}: table is not selected")
            continue
        for column in columns:
            if str(column) not in schemas.get(table, set()):
                invalid.append(f"{table}.{column}: column not found")
        if not str(binding.get("evidence", "")).strip():
            invalid.append(f"{table}: missing binding evidence")
    return invalid


def bound_plan(
    state: P12State, csv_dir: Path, proposal: Proposal, raw_plan: dict[str, Any] | SemanticAnalysisPlan,
) -> dict[str, Any]:
    """The semantic plan validated and bound to the selected tables' real columns."""
    plan: Any = raw_plan.model_dump() if isinstance(raw_plan, SemanticAnalysisPlan) else raw_plan
    if isinstance(plan, dict) and not plan.get("table_roles"):
        plan = {**plan, "table_roles": dict(proposal.table_roles)}
    if isinstance(plan, dict):
        plan = _normalize_semantic_plan(plan, proposal.tables)
        if not plan.get("output_columns"):
            plan["output_columns"] = _derive_output_columns(plan)
    try:
        plan = SemanticAnalysisPlan.model_validate(plan).model_dump()
    except Exception as exc:
        state.plan_failure = str(exc)
        raise ValueError(
            "Selection blocked: invalid semantic_plan: "
            f"{exc}\nCorrect the listed fields and call "
            "confirm_unified_selection again. Do not return the corrected "
            "JSON only as assistant text."
        ) from exc
    schemas = _schemas(csv_dir, proposal.tables)
    plan = _resolve_partition_table_aliases(
        plan, proposal.tables, proposal.table_roles, schemas, proposal.combination_strategy,
    )
    invalid = _invalid_bindings(plan, proposal.selected, schemas)
    if invalid:
        raise ValueError(
            "Selection blocked: semantic bindings are not supported by the "
            "selected schemas: " + "; ".join(invalid)
        )
    return plan


def _coverage_facts(proposal: Proposal) -> tuple[list[str], list[str], list[str], list[str]]:
    """What the roles and the requirement coverage leave out: missing roles, extra roles, malformed, uncovered tables."""
    missing_roles = [table for table in proposal.tables if not proposal.table_roles.get(table)]
    role_extras = [table for table in proposal.table_roles if table not in proposal.selected]
    covered: set[str] = set()
    malformed: list[str] = []
    for requirement, evidence in proposal.requirement_coverage.items():
        if not isinstance(evidence, dict):
            malformed.append(requirement)
            continue
        table = str(evidence.get("table", "")).strip()
        columns = evidence.get("columns", [])
        if table in proposal.selected:
            covered.add(table)
        if table not in proposal.selected or not isinstance(columns, list) or not columns:
            malformed.append(requirement)
    uncovered_tables = [table for table in proposal.tables if table not in covered]
    return missing_roles, role_extras, malformed, uncovered_tables


def _strategy_mismatch(proposal: Proposal) -> str | None:
    """Why the combination strategy does not fit the number of tables, if it does not."""
    count, strategy = len(proposal.tables), proposal.combination_strategy
    if count > 1 and strategy == "single_table":
        return "Multiple tables were selected but the combination strategy is single_table."
    if count == 1 and strategy != "single_table":
        return f"One table was selected but the combination strategy is {strategy}."
    return None


def _alternatives_advisories(state: P12State, proposal: Proposal) -> list[str]:
    advisories: list[str] = []
    alternatives = proposal.alternatives_rejected
    if len(alternatives) > 2:
        advisories.append(
            "More than two rejected alternatives were supplied; keep only the "
            "strongest inspected alternatives in future confirmations."
        )
    inspected = set(state.inspected_candidates())
    unsupported = [table for table in alternatives if table not in inspected]
    if unsupported:
        advisories.append("Rejected alternative(s) were not inspected: " + ", ".join(unsupported))
    for table, evidence in alternatives.items():
        matched = evidence.get("matched_requirements", [])
        missing = str(evidence.get("missing_requirement", "")).casefold()
        if len(matched) >= 2 and missing in VAGUE_MISSING:
            advisories.append(
                f"{table} matches multiple essential requirements but no "
                "concrete missing requirement justifies excluding it. Reconsider "
                "including or preferring this inspected alternative."
            )
    return advisories


@dataclass(frozen=True)
class _Facts:
    """What the review found about the roles, the coverage and the strategy."""

    missing_roles: list[str]
    role_extras: list[str]
    malformed: list[str]
    uncovered_tables: list[str]
    mismatch: str | None


def _facts(proposal: Proposal) -> _Facts:
    missing_roles, role_extras, malformed, uncovered_tables = _coverage_facts(proposal)
    return _Facts(missing_roles, role_extras, malformed, uncovered_tables, _strategy_mismatch(proposal))


def _advisories(state: P12State, proposal: Proposal, facts: _Facts) -> list[str]:
    advisories: list[str] = []
    if facts.missing_roles:
        advisories.append("Selected table(s) without an explicit role: " + ", ".join(facts.missing_roles))
    if facts.role_extras:
        advisories.append("Table role(s) refer to unselected tables: " + ", ".join(facts.role_extras))
    if not proposal.requirement_coverage:
        advisories.append("No explicit requirement coverage was supplied.")
    elif facts.malformed:
        advisories.append(
            "Requirement coverage lacks selected-table/column evidence for: " + ", ".join(facts.malformed)
        )
    if facts.uncovered_tables:
        advisories.append(
            "Selected table(s) cover no explicit requirement: " + ", ".join(facts.uncovered_tables)
        )
    if facts.mismatch:
        advisories.append(facts.mismatch)
    if proposal.uncovered_requirements:
        advisories.append(
            "Selection was confirmed with requirements still marked uncovered: "
            + ", ".join(proposal.uncovered_requirements)
        )
    advisories.extend(_alternatives_advisories(state, proposal))
    return advisories


def _blockers(proposal: Proposal, facts: _Facts) -> list[str]:
    blockers: list[str] = []
    if facts.missing_roles:
        blockers.append("every selected table needs an explicit role")
    if not proposal.requirement_coverage:
        blockers.append("requirement_coverage is required")
    if facts.malformed:
        blockers.append("requirement coverage must bind selected tables and columns")
    if facts.uncovered_tables:
        blockers.append("every selected table must support at least one requirement")
    if proposal.uncovered_requirements:
        blockers.append("uncovered_requirements must be resolved before confirmation")
    if facts.mismatch:
        blockers.append("combination_strategy must match the selected table count")
    return blockers


def review(state: P12State, proposal: Proposal) -> tuple[list[str], list[str]]:
    """What is weak about the proposal (advisories) and what blocks a contract-first one."""
    facts = _facts(proposal)
    return _advisories(state, proposal, facts), _blockers(proposal, facts)


def commit(manager: Phase12ToolsManager, proposal: Proposal) -> str:
    """The selection plan on the state, or the ledger's block; the tool's final payload."""
    state = manager.state
    coder_brief = manager._build_coder_brief(
        proposal.tables, proposal.requirement_coverage, proposal.requirements,
        proposal.combination_strategy, proposal.table_roles, proposal.semantic_plan,
    )
    requirement_ledger = manager._build_requirement_ledger(
        proposal.requirement_coverage, proposal.requirements, proposal.uncovered_requirements, proposal.semantic_plan,
    )
    selection_plan = {
        "requirement_coverage": proposal.requirement_coverage,
        "table_roles": proposal.table_roles,
        "combination_strategy": proposal.combination_strategy,
        "uncovered_requirements": proposal.uncovered_requirements,
        "alternatives_rejected": proposal.alternatives_rejected,
        "requirements": proposal.requirements,
        "requirement_ledger": requirement_ledger,
        "coder_brief": coder_brief,
        **({"semantic_plan": proposal.semantic_plan} if proposal.contract_first else {}),
    }
    block_message = requirement_ledger_block_message(requirement_ledger, proposal.tables)
    if block_message:
        # Kept so an attempt that ends without a terminal tool call falls
        # back to the architect's own bindings, not to bare top candidates.
        state.blocked_selection = {
            "tables": list(proposal.tables),
            "reasoning": proposal.reasoning,
            "block_message": block_message,
            "selection_plan": selection_plan,
        }
        raise ValueError(block_message)
    state.selection_plan = selection_plan
    state.plan_failure = None
    state.selection_plan_source = "confirm_unified_selection" if proposal.contract_first else "selection_only"
    state.confirmed_tables = list(proposal.tables)
    state.selection_reasoning = proposal.reasoning
    state.selection_requirements = proposal.requirements
    state.selection_advisories = proposal.advisories
    payload = {
        "tables": ", ".join(proposal.tables),
        "reasoning": proposal.reasoning,
        "selection_plan": state.selection_plan,
        "advisories": proposal.advisories,
    }
    return f"FINAL_PAYLOAD: {json.dumps(payload)}"
