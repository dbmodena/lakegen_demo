"""The analysis plan: what the coder is asked to do, written down and checked
BEFORE any code exists.

The reviewed coder separates two concerns, each with its own judge:

* the PLAN JUDGE decides whether this plan is faithful to the QUESTION;
* the CODE JUDGE decides whether the code is faithful to this PLAN.

Neither re-judges the other's ground. This module holds what they share: the
plan schema, the deterministic validation that runs before the plan judge
(columns exist, filter values are observed in the data, text columns used as
numbers are converted first), and the mechanical hints both judges receive.

`question_coverage` is the field that makes a plan checkable against a
question. The planner must list every constraint the question states (an
entity, a period, an amount threshold, a category, a measure, an output shape)
and name the plan element that implements it. A constraint may be marked as
not needed only with a fact about the data; the plan judge reads that map
instead of hunting through the plan for what may have been dropped.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lakegen.column_lineage import Violation, compose_feedback

FilterOperator = Literal[
    "equals", "not_equals", "in", "contains", "gt", "gte", "lt", "lte", "between",
    "not_null",
]
MeasureOperation = Literal[
    "count_rows", "count_distinct", "sum", "mean", "median", "min", "max", "ratio",
    "difference", "custom",
]
Combination = Literal[
    "single_table", "aggregate_separately", "concat_partitions", "join", "compare",
]
PlanValue = str | int | float | list[str | int | float] | None

_NUMERIC_FILTERS = {"gt", "gte", "lt", "lte", "between"}
_NUMERIC_MEASURES = {"sum", "mean", "median", "min", "max", "ratio", "difference"}
_MAX_OBSERVED_SHOWN = 8


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PlanFilter(_PlanModel):
    requirement: str = Field(description="The wording of the question this filter implements.")
    tables: list[str] = Field(
        default_factory=list, description="Tables it applies to; empty means every selected table."
    )
    column: str
    operator: FilterOperator
    value: PlanValue = None
    evidence: str = Field(description="What the data shows that grounds this column and value.")


class PlanPreparation(_PlanModel):
    """What must happen to a column before it is used, such as turning text into numbers."""

    tables: list[str] = Field(default_factory=list)
    column: str
    action: str
    evidence: str


class PlanDimension(_PlanModel):
    output: str
    source: Literal["column", "table_role"] = "column"
    column: str | None = None
    tables: list[str] = Field(default_factory=list)
    requirement: str


class PlanMeasure(_PlanModel):
    output: str
    operation: MeasureOperation
    columns: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    requirement: str
    evidence: str = ""


class PlanJoin(_PlanModel):
    tables: list[str]
    keys: dict[str, str] = Field(default_factory=dict)
    how: Literal["inner", "left", "right", "outer"] = "inner"


class PlanOrdering(_PlanModel):
    output: str
    direction: Literal["ascending", "descending"]


class AnalysisPlan(_PlanModel):
    summary: str
    table_roles: dict[str, str]
    combination: Combination = "single_table"
    joins: list[PlanJoin] = Field(default_factory=list)
    filters: list[PlanFilter] = Field(default_factory=list)
    preparation: list[PlanPreparation] = Field(default_factory=list)
    dimensions: list[PlanDimension] = Field(default_factory=list)
    measures: list[PlanMeasure]
    ordering: list[PlanOrdering] = Field(default_factory=list)
    limit: int | None = None
    output_columns: list[str]
    question_coverage: dict[str, str]

    @model_validator(mode="after")
    def _complete(self) -> "AnalysisPlan":
        if not self.table_roles:
            raise ValueError("table_roles must give every selected table a role")
        if not self.measures:
            raise ValueError("measures must contain at least one measure")
        if not self.output_columns:
            raise ValueError("output_columns must not be empty")
        if not self.question_coverage:
            raise ValueError(
                "question_coverage must list every constraint the question states"
            )
        empty = [k for k, v in self.question_coverage.items() if not str(v).strip()]
        if empty:
            raise ValueError(
                f"question_coverage entries need the plan element that implements "
                f"them or the data fact that makes them unnecessary: {empty}"
            )
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be a positive integer or null")
        return self


def _values(value: PlanValue) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if str(item).strip()]


def plan_to_text(plan: AnalysisPlan | dict[str, Any]) -> str:
    """The plan as the JSON both judges and the coder read."""
    data = plan.model_dump(exclude_none=True) if isinstance(plan, AnalysisPlan) else plan
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


@dataclass
class PlanValidation:
    violations: list[Violation] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations and not self.diagnostics

    def feedback(self) -> str:
        parts = []
        if self.violations:
            parts.append(compose_feedback(self.violations))
        parts.extend(self.diagnostics)
        return " | ".join(part for part in parts if part)


def _is_text(series: pd.Series) -> bool:
    return not (
        pd.api.types.is_numeric_dtype(series.dtype)
        or pd.api.types.is_datetime64_any_dtype(series.dtype)
        or pd.api.types.is_bool_dtype(series.dtype)
    )


def _observed(series: pd.Series) -> list[str]:
    counts = series.dropna().astype(str).str.strip().value_counts()
    return [str(v) for v in counts.head(_MAX_OBSERVED_SHOWN).index]


def validate_analysis_plan(
    plan: AnalysisPlan, frames: dict[str, pd.DataFrame]
) -> PlanValidation:
    """Deterministic checks that need no model: the plan must refer to real
    tables and columns, filter on values the data holds, and convert a text
    column before using it as a number. Whether the plan answers the QUESTION
    is not decided here; that is the plan judge's ground."""
    out = PlanValidation()
    selected = list(frames)

    def check_tables(names: list[str], where: str) -> list[str]:
        unknown = [n for n in names if n not in frames]
        for name in unknown:
            close = difflib.get_close_matches(name, selected, n=1)
            out.violations.append(Violation(
                "unknown_table", 0, name,
                f"{where} names a table that is not selected"
                + (f"; closest: {close[0]}" if close else ""),
            ))
        return [n for n in names if n in frames]

    def scoped(names: list[str], where: str) -> list[str]:
        """The tables a binding applies to: those it names, else every selected one."""
        known = check_tables(names, where)
        return known if names else list(frames)

    def check_column(column: str, tables: list[str], where: str) -> list[str]:
        present = []
        for table in tables:
            columns = [str(c) for c in frames[table].columns]
            if column in columns:
                present.append(table)
                continue
            close = difflib.get_close_matches(column, columns, n=1)
            out.violations.append(Violation(
                "unknown_column", 0, column,
                f"{where}: not a column of {table}"
                + (f"; closest: {close[0]}" if close else ""),
            ))
        return present

    check_tables(list(plan.table_roles), "table_roles")
    unused = [t for t in selected if t not in plan.table_roles]
    if unused:
        out.diagnostics.append(
            f"selected table(s) with no role in the plan: {unused}; every selected "
            "table was chosen for a reason, so give each a role or say why it is not used"
        )

    def prepared(column: str, table: str) -> bool:
        return any(
            p.column == column and (not p.tables or table in p.tables)
            for p in plan.preparation
        )

    for index, flt in enumerate(plan.filters):
        where = f"filters[{index}] ({flt.requirement})"
        for table in check_column(flt.column, scoped(flt.tables, where), where):
            series = frames[table][flt.column]
            wanted = _values(flt.value)
            if flt.operator in {"equals", "in", "contains"} and wanted:
                if pd.api.types.is_numeric_dtype(series.dtype):
                    for v in wanted:
                        try:
                            hit = bool((series == float(v)).any())
                        except ValueError:
                            hit = False
                        if not hit:
                            out.diagnostics.append(
                                f"{where}: value {v!r} never appears in {flt.column!r} of {table}"
                            )
                    continue
                observed = series.dropna().astype(str).str.strip().str.casefold()
                seen = set(observed.unique())
                for v in wanted:
                    needle = v.casefold()
                    hit = (
                        observed.str.contains(needle, regex=False).any()
                        if flt.operator == "contains" else needle in seen
                    )
                    if not hit:
                        out.diagnostics.append(
                            f"{where}: value {v!r} never appears in {flt.column!r} of {table} "
                            f"(observed: {_observed(series)})"
                        )
            if flt.operator in _NUMERIC_FILTERS and _is_text(series) and not prepared(flt.column, table):
                out.diagnostics.append(
                    f"{where}: {flt.column!r} is stored as text in {table} but is compared "
                    "numerically; add a preparation step that converts it first"
                )

    for index, prep in enumerate(plan.preparation):
        where = f"preparation[{index}]"
        check_column(prep.column, scoped(prep.tables, where), where)

    for index, dim in enumerate(plan.dimensions):
        if dim.source == "column" and dim.column:
            where = f"dimensions[{index}]"
            check_column(dim.column, scoped(dim.tables, where), where)

    for index, measure in enumerate(plan.measures):
        where = f"measures[{index}] ({measure.requirement})"
        for column in measure.columns:
            for table in check_column(column, scoped(measure.tables, where), where):
                if (
                    measure.operation in _NUMERIC_MEASURES
                    and _is_text(frames[table][column])
                    and not prepared(column, table)
                ):
                    out.diagnostics.append(
                        f"{where}: {column!r} is stored as text in {table} but is aggregated "
                        "numerically; add a preparation step that converts it first"
                    )

    outputs = {m.output for m in plan.measures} | {d.output for d in plan.dimensions}
    for column in plan.output_columns:
        if column not in outputs:
            out.diagnostics.append(
                f"output column {column!r} is produced by no measure or dimension"
            )
    for name in sorted(outputs - set(plan.output_columns)):
        out.diagnostics.append(f"{name!r} is computed but missing from output_columns")
    return out


_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])")
_YEARISH = re.compile(r"(?:19|20)\d{2}")


def question_numbers_missing_from_plan(question: str, plan: AnalysisPlan | dict[str, Any]) -> list[str]:
    """Numbers the question states (an amount, a top-N) that appear nowhere in
    the plan. A prompt for the plan judge, never a verdict: the question may
    mean the number some other way. Years and fiscal-year forms are skipped;
    periods are a different check."""
    text = plan_to_text(plan)
    stripped = re.sub(r"(?<!\d)(?:19|20)\d{2}\s*[/\-]\s*(?:\d{4}|\d{2})(?!\d)", " ", question)
    missing: list[str] = []
    for raw in _NUMBER.findall(stripped):
        number = raw.replace(",", "")
        if _YEARISH.fullmatch(number) or number in missing:
            continue
        if number not in text and raw not in text:
            missing.append(number)
    return missing


def alignment_notes(plan: AnalysisPlan | dict[str, Any], code: str) -> str:
    """Mechanical plan-to-code hints for the code judge: plan columns, filter
    values and output names that never appear as a literal in the code. Like
    OrQa's plan/code alignment check this is a prompt to look closer, not
    proof: code can reach a column through a variable or a rename."""
    data = plan.model_dump(exclude_none=True) if isinstance(plan, AnalysisPlan) else plan
    missing_columns: list[str] = []
    missing_values: list[str] = []
    missing_outputs: list[str] = []

    def quoted(literal: str) -> bool:
        return re.search(r"['\"]" + re.escape(literal) + r"['\"]", code) is not None

    def note(target: list[str], item: str) -> None:
        if item and item not in target:
            target.append(item)

    for key in ("filters", "preparation", "dimensions", "measures"):
        for entry in data.get(key, []):
            for column in [entry.get("column"), *(entry.get("columns") or [])]:
                if column and not quoted(str(column)):
                    note(missing_columns, str(column))
    for flt in data.get("filters", []):
        raw = flt.get("value")
        for value in raw if isinstance(raw, list) else [raw]:
            if value is None or str(value).strip() == "":
                continue
            literal = str(value).strip().replace(",", "")
            if not (quoted(str(value).strip()) or re.search(r"(?<![\w.])" + re.escape(literal) + r"(?![\w])", code)):
                note(missing_values, str(value).strip())
    for column in data.get("output_columns", []):
        if not quoted(str(column)):
            note(missing_outputs, str(column))

    lines = []
    if missing_columns:
        lines.append(f"plan columns never quoted in the code: {missing_columns}")
    if missing_values:
        lines.append(f"plan filter values never appearing in the code: {missing_values}")
    if missing_outputs:
        lines.append(f"plan output columns never quoted in the code: {missing_outputs}")
    return "\n".join(lines)
