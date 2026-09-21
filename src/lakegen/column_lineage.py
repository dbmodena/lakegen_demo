"""Plan representation and lineage check for hallucinated columns.

Ported from scratchpad validation (`coder_plan.py`/`column_lineage.py`),
modeled on OrQa's `column_provenance.py`, which frames the core idea
precisely: "the only thing that can now fail is a name no step declared and
no table contains -- which is the definition of a hallucinated column."

Adapted from the scratchpad design in one deliberate way: the scratchpad
assumed a rich, LLM-authored structured plan (filters/dimensions/measures
each with table/column/output) to derive multi-step lineage from. In
production, that artifact (`coder_brief`/`AnalysisContractSchema`) is
frequently an almost-empty "runtime_fallback" with prose-only fields --
confirmed by direct smoke test, not assumed. The one reliable, deterministic,
code-derived structured artifact available post-execution is
`Phase3Result.operation_trace` (built by
`Phase3ToolsManager._build_operation_trace` from the ACTUALLY EXECUTED code):
`used_tables`/`used_columns` are dependable; `applied_filters`/
`aggregations`/`grouping_columns`/`output_columns` are frequently empty even
when the code clearly does those things. `derive_plan_from_operation_trace`
therefore produces a single flat step (no `produces` claims, since we don't
reliably know what the code derived) rather than inventing multi-step
structure LakeGen's coder doesn't reliably supply -- `resolve_plan_columns`
below still supports true multi-step/`produces` plans generically, so this
degrades gracefully rather than needing a rewrite if a richer plan source
(a genuine pre-code planner) is ever built.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lakegen.column_resolution import resolve_column_name


@dataclass
class DerivedColumn:
    name: str
    sources: list[str]


@dataclass
class PlanStep:
    order: int
    op: str  # filter | join | group | aggregate | sort | select | derive | execute
    tables: list[str]
    reads: list[str]
    produces: list[DerivedColumn] = field(default_factory=list)
    description: str = ""


def derive_plan_from_operation_trace(operation_trace: dict) -> list[PlanStep]:
    """One flat step from the deterministic, code-derived fields that are
    actually reliable (`used_tables`/`used_columns`) -- see module docstring
    for why this isn't a multi-step plan today."""
    tables = [str(t) for t in (operation_trace.get("used_tables") or [])]
    columns = [str(c) for c in (operation_trace.get("used_columns") or [])]
    if not tables and not columns:
        return []
    return [PlanStep(
        order=1, op="execute", tables=tables, reads=columns,
        description="columns referenced by the executed code",
    )]


def all_produced_names(steps: list[PlanStep]) -> set[str]:
    return {dc.name for step in steps for dc in step.produces}


@dataclass
class Violation:
    category: str  # unknown_column | forward_reference
    step_order: int
    column: str
    suggestion: str | None = None


def resolve_plan_columns(steps: list[PlanStep], schemas: dict[str, list[str]]) -> list[Violation]:
    """`schemas` maps table filename -> list of real column names. Walks
    steps in order, maintaining a registry of every column any earlier step
    declared as `produces`; a read resolving against neither the real raw
    schema of its own step's tables nor an earlier `produces` declaration is
    the hallucinated-column case."""
    violations: list[Violation] = []
    produced_so_far: set[str] = set()

    for step in steps:
        raw_schema_for_step: list[str] = []
        for t in step.tables:
            raw_schema_for_step.extend(schemas.get(t, []))
        for col in step.reads:
            if not col:
                continue
            if resolve_column_name(col, raw_schema_for_step) is not None:
                continue
            if col in produced_so_far:
                continue
            later_produces = {dc.name for later in steps if later.order > step.order for dc in later.produces}
            if col in later_produces:
                violations.append(Violation(
                    category="forward_reference", step_order=step.order, column=col,
                    suggestion=f"'{col}' is produced by a later step -- reorder or move the read after it.",
                ))
                continue
            all_real = [c for cols in schemas.values() for c in cols]
            suggestion = resolve_column_name(col, all_real)
            violations.append(Violation(
                category="unknown_column", step_order=step.order, column=col,
                suggestion=(f"did you mean {suggestion!r}?" if suggestion else None),
            ))
        for dc in step.produces:
            produced_so_far.add(dc.name)
            for src in dc.sources:
                if resolve_column_name(src, raw_schema_for_step) is None and src not in produced_so_far:
                    all_real = [c for cols in schemas.values() for c in cols]
                    suggestion = resolve_column_name(src, all_real)
                    violations.append(Violation(
                        category="unknown_column", step_order=step.order, column=src,
                        suggestion=(f"did you mean {suggestion!r}?" if suggestion else None),
                    ))

    return violations


def compose_feedback(violations: list[Violation]) -> str:
    if not violations:
        return ""
    lines = ["Plan lineage check failed -- the following columns are hallucinated "
             "(not a real column of any given table, and not declared by any earlier step):"]
    for v in violations:
        line = f"  step {v.step_order}: {v.column!r} ({v.category})"
        if v.suggestion:
            line += f" -- {v.suggestion}"
        lines.append(line)
    lines.append("Fix: use exact column names copied from the real schema, or reorder steps "
                 "so a derived column is declared before it is read.")
    return "\n".join(lines)
