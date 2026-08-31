"""Allowlisted, benchmark-blind context supplied to the coding agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping


_METADATA_FIELDS = frozenset({"title", "description", "topics", "columns"})
_COLUMN_FIELDS = frozenset({"name", "description", "data_type", "dtype"})
_PLAN_FIELDS = frozenset({
    "requirement_coverage", "table_roles", "combination_strategy",
    "uncovered_requirements", "alternatives_rejected", "semantic_plan",
    "recovered_from_existing_discovery_context",
})
_SEMANTIC_PLAN_FIELDS = frozenset({
    "filters", "temporal_filters", "dimensions", "measures", "joins",
    "ordering", "limit", "output_columns", "null_policy", "table_roles",
    "evidence_map",
})
_BINDING_FIELDS = frozenset({
    "requirement", "table", "column", "columns", "operator", "value",
    "output", "operation", "distinct", "evidence", "tables", "keys",
    "how", "direction", "observed_values", "observed_range", "origin",
})


def infer_output_shape(question: str) -> dict[str, Any]:
    """Derive only generic result shape from the question, never from gold."""
    lowered = question.casefold()
    limit = None
    match = re.search(r"\btop\s+(\d+)\b", lowered)
    if match:
        limit = int(match.group(1))
    grouped = bool(re.search(r"\b(?:by|per|for each|each)\b", lowered))
    ranked = bool(re.search(r"\b(?:top|bottom|highest|lowest|most|least)\b", lowered))
    scalar = bool(re.search(r"^(?:how many|what is the (?:total|average|mean|sum))\b", lowered)) and not grouped
    return {
        "result_type": "number" if scalar else "table",
        "ordered": ranked,
        "row_limit": limit,
        "source": "derived_from_question",
    }


def _allowlisted_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key in _METADATA_FIELDS:
        if key not in value:
            continue
        item = value[key]
        if key == "columns" and isinstance(item, list):
            clean[key] = [
                {k: column[k] for k in _COLUMN_FIELDS if k in column}
                for column in item if isinstance(column, Mapping)
            ]
        elif isinstance(item, (str, int, float, bool, list, tuple)) or item is None:
            clean[key] = item
    return clean


def _allowlisted_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key in _PLAN_FIELDS:
        if key not in plan:
            continue
        value = plan[key]
        if key == "semantic_plan" and isinstance(value, Mapping):
            semantic: dict[str, Any] = {}
            for semantic_key in _SEMANTIC_PLAN_FIELDS:
                if semantic_key not in value:
                    continue
                item = value.get(semantic_key)
                if isinstance(item, list):
                    semantic[semantic_key] = [
                        {k: entry[k] for k in _BINDING_FIELDS if k in entry}
                        if isinstance(entry, Mapping) else entry
                        for entry in item
                    ]
                elif isinstance(item, Mapping):
                    semantic[semantic_key] = dict(item)
                elif item is None or isinstance(item, (str, int, float, bool)):
                    semantic[semantic_key] = item
            clean[key] = semantic
        elif isinstance(value, Mapping):
            clean[key] = dict(value)
        elif isinstance(value, (list, str, int, float, bool)) or value is None:
            clean[key] = value
    return clean


@dataclass(frozen=True)
class CoderContext:
    question: str
    selected_tables: tuple[str, ...]
    table_metadata: dict[str, dict[str, Any]]
    selection_plan: dict[str, Any]
    output_shape: dict[str, Any]
    execution_error: dict[str, Any] = field(default_factory=dict)
    excluded_fields: tuple[str, ...] = ()

    @classmethod
    def build(
        cls, *, question: str, selected_tables: list[str],
        table_metadata: Mapping[str, Any] | None,
        selection_plan: Mapping[str, Any] | None,
        source_payload: Mapping[str, Any] | None = None,
        execution_error: Mapping[str, Any] | None = None,
    ) -> "CoderContext":
        metadata = {
            table: _allowlisted_metadata(
                (table_metadata or {}).get(table, (table_metadata or {}).get(table.rsplit(".", 1)[0], {}))
            )
            for table in selected_tables
        }
        forbidden = []
        allowed_source = {"question"}
        for key in (source_payload or {}):
            if str(key).casefold() not in allowed_source:
                forbidden.append(str(key))
        return cls(
            question=question,
            selected_tables=tuple(selected_tables),
            table_metadata=metadata,
            selection_plan=_allowlisted_plan(selection_plan or {}),
            output_shape=infer_output_shape(question),
            execution_error=dict(execution_error or {}),
            excluded_fields=tuple(sorted(set(forbidden))),
        )

    def audit(self) -> dict[str, Any]:
        return {
            "included_fields": [
                "question", "selected_tables", "table_metadata",
                "selection_plan", "output_shape", "execution_error",
            ],
            "evidence_origins": ["question", "runtime_retrieval", "runtime_table_inspection", "runtime_execution"],
            "excluded_field_names": list(self.excluded_fields),
            "context_filtered": bool(self.excluded_fields),
            "reference_accessed_by_coder": False,
        }
