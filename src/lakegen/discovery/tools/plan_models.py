"""The schemas of the unified agent's selection and of the semantic plan it binds to the tables."""

import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SemanticFilterBinding(BaseModel):
    requirement: str
    table: str
    column: str
    operator: Literal["equals", "contains", "in", "range", "not_null", "other"]
    value: str = ""
    evidence: str


class SemanticTemporalFilterBinding(SemanticFilterBinding):
    """A row-level time constraint (dataset-edition years do not belong here)."""


class SemanticJoinBinding(BaseModel):
    tables: list[str]
    keys: dict[str, str]
    how: Literal["inner", "left", "right", "outer"] = "inner"
    evidence: str


class SemanticDimensionBinding(BaseModel):
    output: str
    table: str
    column: str
    evidence: str


class SemanticMeasureBinding(BaseModel):
    output: str
    operation: Literal[
        "count_rows", "count_distinct", "sum", "mean", "min", "max",
        "ratio", "difference", "custom",
    ]
    table: str
    # count_rows counts rows and reads no column; a live architect counting
    # licences was blocked for omitting one. Every other operation names the
    # inspected column(s) it reads.
    columns: list[str] = Field(default_factory=list)
    distinct: bool = False
    evidence: str

    @model_validator(mode="after")
    def validate_columns_for_operation(self) -> "SemanticMeasureBinding":
        if not self.columns and self.operation != "count_rows":
            raise ValueError(
                f"columns is required for operation {self.operation!r}: name "
                "the inspected column(s) it reads; only count_rows may omit it"
            )
        return self


class SemanticOrdering(BaseModel):
    output: str
    direction: Literal["ascending", "descending"]


class SemanticAnalysisPlan(BaseModel):
    filters: list[SemanticFilterBinding] = Field(default_factory=list)
    temporal_filters: list[SemanticTemporalFilterBinding] = Field(default_factory=list)
    dimensions: list[SemanticDimensionBinding] = Field(default_factory=list)
    measures: list[SemanticMeasureBinding]
    joins: list[SemanticJoinBinding | str] = Field(default_factory=list)
    ordering: list[SemanticOrdering] = Field(default_factory=list)
    limit: int | None = None
    output_columns: list[str]
    null_policy: str = "preserve nulls unless the requested operation requires exclusion"
    table_roles: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_complete_non_empty_plan(self) -> "SemanticAnalysisPlan":
        if not self.measures:
            raise ValueError("measures must contain at least one structured binding")
        if not self.output_columns:
            raise ValueError("output_columns must not be empty")
        if not self.table_roles:
            raise ValueError("table_roles must cover the selected runtime tables")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be a positive integer or null")
        return self


class ConfirmUnifiedSelectionSchema(BaseModel):
    reasoning: str = Field(description="Required. Brief English explanation of why these tables answer the question.")
    tables: list[str] = Field(description="Every exact filename needed, e.g. ['sales.parquet', 'dates.parquet'].")
    requirement_coverage: dict[str, dict[str, object]] = Field(
        description=(
            "Every essential requirement, each its own entry, mapped to "
            "{'table': one of `tables`, 'columns': [exact inspected column "
            "names]} -- `columns` is always a list, even for one column. Name "
            "the question's own period and places in the key, e.g. "
            "{'year 2021': {'table': 'permits.parquet', 'columns': "
            "['issue_date']}}. Every selected table must back at least one entry."
        ),
    )
    table_roles: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Each selected filename mapped to its distinct role, e.g. fact "
            "records, lookup, or yearly partition."
        ),
    )
    combination_strategy: Literal[
        "single_table",
        "join",
        "concat_partitions",
        "aggregate_separately",
        "lookup",
        "compare",
    ] = Field(
        default="single_table",
        description="How the coder should combine the selected tables.",
    )
    uncovered_requirements: list[str] = Field(
        default_factory=list,
        description=(
            "Essential requirements still not proven by the selected tables. "
            "Use an empty list when coverage is complete."
        ),
    )
    alternatives_rejected: dict[str, dict[str, object] | str] = Field(
        default_factory=dict,
        description=(
            "Up to two inspected alternative filenames, each mapped to "
            "{matched_requirements, missing_requirement} with one concrete "
            "missing requirement."
        ),
    )
    requirements: dict[str, object] = Field(
        description=(
            "Compact semantics: grouping, measures, filters, result_type, "
            "ordering, limit. Each filter names its table, column, operator and "
            "value; keep different periods separate. Joins: left_table, "
            "left_columns, right_table, right_columns, how."
        ),
    )
    semantic_plan: dict[str, object] | None = Field(
        default=None,
        description=(
            "Executable semantics from the question and inspected evidence "
            "only, never benchmark expectations; a JSON object, never SQL or "
            "prose: `measures` (at least one) [{output, operation: count_rows|"
            "count_distinct|sum|mean|min|max|ratio|difference|custom, table, "
            "columns (omit only for count_rows), evidence}]; `dimensions` "
            "[{output, table, column, evidence}]; `filters`/`temporal_filters` "
            "[{requirement, table, column, operator: equals|contains|in|range|"
            "not_null|other, value, evidence}]; `joins` [{tables, keys: {table: "
            "column}, how, evidence}]; `ordering` [{output, direction}]; "
            "`limit` (int or null); `null_policy`. Tables must be selected ones, "
            "columns exact inspected names, `evidence` the inspected schema "
            "behind the binding. `output_columns` and `table_roles` are filled "
            "in automatically."
        )
    )


class RejectUnifiedSelectionSchema(BaseModel):
    reasoning: str = Field(description="Explain step-by-step why the current candidates are not good.")
    suggestion: str = Field(
        description="Suggest dataset concepts, not analytical operations or row-filter values."
    )
    ban_tables: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Inspected candidates proven to satisfy none of the essential "
            "requirements, each mapped to the concrete evidence (e.g. 'Parent "
            "Department is only Crown Prosecution Service, never the requested "
            "department'); they are excluded from later retrieval. Every other "
            "inspected candidate carries over to the next attempt. A table "
            "without concrete evidence is not banned, nor one you proposed in "
            "confirm_unified_selection during this attempt."
        ),
    )


class SubmitSemanticPlanDraftSchema(BaseModel):
    draft: dict[str, object] = Field(
        description=(
            "Compact draft with filters [column, operator, value], dimensions "
            "[output, column], measures [output, operation, columns (omit only "
            "for count_rows)], optional "
            "joins, ordering [output, direction], and limit."
        )
    )


def _normalize_semantic_plan(
    plan: dict[str, object], selected_tables: list[str]
) -> dict[str, object]:
    """Normalize harmless tool-input variants without inventing semantic evidence."""
    normalized = dict(plan)
    operation_aliases = {
        "count": "count_rows", "avg": "mean", "average": "mean",
        # A scalar lookup ("the area of X") is not an aggregate; the model
        # reaches for a passthrough name here often enough that it is worth
        # mapping onto the schema's own escape hatch rather than failing.
        "value": "custom", "identity": "custom", "lookup": "custom",
        "none": "custom", "select": "custom", "raw": "custom",
    }
    operator_aliases = {
        "is_not_null": "not_null", "year_eq": "equals", "year_equals": "equals",
    }
    direction_aliases = {
        "asc": "ascending", "desc": "descending",
    }
    default_table = selected_tables[0] if len(selected_tables) == 1 else ""
    for key in ("filters", "temporal_filters", "dimensions", "measures"):
        items: list[object] = []
        for raw in normalized.get(key, []) if isinstance(normalized.get(key), list) else []:
            if not isinstance(raw, dict):
                items.append(raw)
                continue
            item = dict(raw)
            if default_table and not item.get("table"):
                item["table"] = default_table
            if key in {"filters", "temporal_filters"}:
                operator = str(item.get("operator") or item.get("type") or "").casefold()
                item["operator"] = operator_aliases.get(operator, operator)
                value = item.get("value", "")
                if not isinstance(value, str):
                    item["value"] = json.dumps(value, ensure_ascii=False)
            elif key == "measures":
                operation = str(
                    item.get("operation") or item.get("aggregation") or item.get("type") or ""
                ).casefold()
                item["operation"] = operation_aliases.get(operation, operation)
                if "columns" not in item and item.get("column"):
                    item["columns"] = [item.pop("column")]
            items.append(item)
        normalized[key] = items
    if isinstance(normalized.get("ordering"), list):
        # A coder_brief's own `ordering` may be an arbitrary pass-through value
        # (e.g. a free-form string), not this structured shape -- touch it only
        # when it is already a list, never coerce a non-list value away.
        ordering: list[object] = []
        for raw in normalized["ordering"]:
            if not isinstance(raw, dict):
                ordering.append(raw)
                continue
            item = dict(raw)
            direction = str(item.get("direction") or "").casefold()
            item["direction"] = direction_aliases.get(direction, direction)
            ordering.append(item)
        normalized["ordering"] = ordering
    joins: list[object] = []
    for raw in normalized.get("joins", []) if isinstance(normalized.get("joins"), list) else []:
        if not isinstance(raw, dict):
            joins.append(raw)
            continue
        join = dict(raw)
        if not join.get("tables") and join.get("left_table") and join.get("right_table"):
            join["tables"] = [join.pop("left_table"), join.pop("right_table")]
        if not join.get("keys") and join.get("left_columns") and join.get("right_columns"):
            tables = list(join.get("tables") or [])
            left_columns = list(join.pop("left_columns") or [])
            right_columns = list(join.pop("right_columns") or [])
            if len(tables) == 2 and len(left_columns) == len(right_columns) == 1:
                join["keys"] = {
                    str(tables[0]): str(left_columns[0]),
                    str(tables[1]): str(right_columns[0]),
                }
        if not join.get("keys") and join.get("left_key") and join.get("right_key"):
            tables = list(join.get("tables") or [])
            if len(tables) == 2:
                join["keys"] = {
                    str(tables[0]): join.pop("left_key"),
                    str(tables[1]): join.pop("right_key"),
                }
        joins.append(join)
    normalized["joins"] = joins
    return normalized


def _resolve_partition_table_aliases(
    plan: dict[str, object], selected_tables: list[str],
    table_roles: dict[str, str], schema_by_table: dict[str, set[str]],
    combination_strategy: str,
) -> dict[str, object]:
    """Resolve only aliases proven unambiguous by this confirmed selection."""
    normalized = dict(plan)
    role_targets: dict[str, list[str]] = {}
    for table, role in table_roles.items():
        if table in selected_tables and str(role).strip():
            role_targets.setdefault(str(role).strip().casefold(), []).append(table)

    for field in ("filters", "temporal_filters", "dimensions", "measures"):
        resolved: list[object] = []
        for raw in normalized.get(field, []):
            if not isinstance(raw, dict):
                resolved.append(raw)
                continue
            item = dict(raw)
            alias = str(item.get("table") or "").strip()
            role_matches = role_targets.get(alias.casefold(), [])
            if len(role_matches) == 1:
                item["table"] = role_matches[0]
                resolved.append(item)
                continue
            if alias.casefold() != "any" or combination_strategy != "concat_partitions":
                resolved.append(item)
                continue
            columns = item.get("columns")
            if columns is None:
                columns = [item.get("column")] if item.get("column") else []
            names = {str(column) for column in columns}
            common = bool(names) and all(
                names <= schema_by_table.get(table, set())
                for table in selected_tables
            )
            if not common:
                resolved.append(item)
                continue
            if field in {"filters", "temporal_filters"}:
                resolved.extend({**item, "table": table} for table in selected_tables)
            else:
                # This binding is evaluated over the concatenated frame. One
                # concrete source proves the shared column without duplicating
                # the requested output for every partition.
                item["table"] = selected_tables[0]
                resolved.append(item)
        normalized[field] = resolved
    return normalized


def _normalize_requirement_coverage(
    requirement_coverage: dict[str, object], selected_tables: list[str]
) -> dict[str, object]:
    """Normalize harmless requirement_coverage input variants.

    Never guesses a table when more than one is selected -- that would bind a
    requirement to evidence the model never actually gave -- but the two
    mistakes actually seen in practice (a single column string instead of a
    list, and an omitted table when only one is selected) are unambiguous.
    """
    default_table = selected_tables[0] if len(selected_tables) == 1 else ""
    normalized: dict[str, object] = {}
    for requirement, evidence in requirement_coverage.items():
        if not isinstance(evidence, dict):
            normalized[requirement] = evidence
            continue
        item = dict(evidence)
        table = str(item.get("table") or "").strip()
        item["table"] = table or default_table
        columns = item.get("columns")
        if columns is None and item.get("column"):
            columns = [item.pop("column")]
        elif isinstance(columns, str):
            columns = [columns]
        if columns is not None:
            item["columns"] = columns
        normalized[requirement] = item
    return normalized


def _derive_output_columns(plan: dict[str, object]) -> list[str]:
    """Name every result column once, from each dimension's/measure's `output`.

    Repeating that same list back as `output_columns` is a common, harmless
    omission in a model-supplied semantic_plan; only used where that field is
    genuinely optional input, never to invent a coder_brief's own explicit
    (and separately meaningful, possibly intentionally empty) output_columns.
    """
    return [
        str(item.get("output")) for item in [
            *plan.get("dimensions", []), *plan.get("measures", [])
        ]
        if isinstance(item, dict) and str(item.get("output") or "").strip()
    ]


def _draft_item(raw: object, names: tuple[str, ...]) -> dict[str, object]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (list, tuple)):
        return {name: raw[index] for index, name in enumerate(names) if index < len(raw)}
    return {}


def compile_semantic_plan_draft(
    draft: dict[str, object], selected_tables: list[str],
    table_roles: dict[str, str], schema_by_table: dict[str, set[str]],
) -> dict[str, object]:
    """Compile an explicit model draft; never infer missing semantic choices."""
    default_table = selected_tables[0] if len(selected_tables) == 1 else ""

    def bind(raw: object, names: tuple[str, ...], kind: str) -> dict[str, object]:
        item = _draft_item(raw, names)
        table = str(item.get("table") or default_table)
        if not table:
            raise ValueError(f"{kind} binding must name a table for multi-table plans")
        column_names = item.get("columns")
        if column_names is None and item.get("column"):
            column_names = [item["column"]]
        columns = [str(value) for value in (column_names or [])]
        evidence_columns = columns or [str(item.get("column") or "")]
        evidence_columns = [column for column in evidence_columns if column]
        missing = [column for column in evidence_columns if column not in schema_by_table.get(table, set())]
        if table not in selected_tables or missing:
            raise ValueError(
                f"{kind} binding is not supported by inspected schema: "
                f"table={table!r}, missing_columns={missing}"
            )
        output = str(item.get("output") or "").strip()
        column = str(item.get("column") or "").strip()
        requirement = str(item.get("requirement") or output or column).strip()
        if not requirement:
            raise ValueError(f"{kind} binding needs an explicit output or requirement")
        item["table"] = table
        item["requirement"] = requirement
        item["evidence"] = (
            f"Inspected runtime schema for {table} contains: "
            + ", ".join(evidence_columns)
        )
        return item

    filters = [bind(raw, ("column", "operator", "value"), "filter")
               for raw in draft.get("filters", []) if raw is not None]
    temporal = [bind(raw, ("column", "operator", "value"), "temporal_filter")
                for raw in draft.get("temporal_filters", []) if raw is not None]
    dimensions = [bind(raw, ("output", "column"), "dimension")
                  for raw in draft.get("dimensions", []) if raw is not None]
    measures = [bind(raw, ("output", "operation", "columns"), "measure")
                for raw in draft.get("measures", []) if raw is not None]
    if not measures:
        raise ValueError("draft must contain at least one explicit measure")
    output_columns = [
        str(item.get("output")) for item in [*dimensions, *measures]
        if str(item.get("output") or "").strip()
    ]
    if not output_columns:
        raise ValueError("draft bindings must declare output names")
    normalized_join_plan = _normalize_semantic_plan(
        {"joins": list(draft.get("joins", []))}, selected_tables
    )
    joins: list[dict[str, object] | str] = []
    for raw in normalized_join_plan.get("joins", []):
        if isinstance(raw, str):
            joins.append(raw)
            continue
        if not isinstance(raw, dict):
            raise ValueError("join must be a structured object or explicit legacy string")
        join = dict(raw)
        tables = [str(table) for table in join.get("tables", [])]
        keys = {str(table): str(column) for table, column in dict(join.get("keys") or {}).items()}
        if len(tables) < 2 or any(table not in selected_tables for table in tables):
            raise ValueError("join tables must be explicitly selected")
        missing_keys = [
            f"{table}.{keys.get(table, '')}" for table in tables
            if not keys.get(table) or keys[table] not in schema_by_table.get(table, set())
        ]
        if missing_keys:
            raise ValueError("join keys missing from inspected schemas: " + ", ".join(missing_keys))
        join["tables"] = tables
        join["keys"] = keys
        join["evidence"] = "Inspected runtime schemas contain join keys: " + ", ".join(
            f"{table}.{keys[table]}" for table in tables
        )
        joins.append(join)
    plan = {
        "filters": filters,
        "temporal_filters": temporal,
        "dimensions": dimensions,
        "measures": measures,
        "joins": joins,
        "ordering": [
            _draft_item(raw, ("output", "direction"))
            for raw in draft.get("ordering", []) if raw is not None
        ],
        "limit": draft.get("limit"),
        "output_columns": output_columns,
        "null_policy": str(draft.get("null_policy") or (
            "preserve nulls unless the requested operation requires exclusion"
        )),
        "table_roles": dict(table_roles),
    }
    return _normalize_semantic_plan(plan, selected_tables)
