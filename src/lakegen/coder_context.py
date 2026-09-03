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
    "recovered_from_existing_discovery_context", "coder_brief",
})
_CODER_BRIEF_FIELDS = frozenset({
    "tables", "selected_columns", "task", "filters", "operations", "result_type",
    "temporal_filters", "dimensions", "measures", "output_columns", "null_policy",
    "table_roles", "ordering", "limit", "joins", "combination_strategy",
    "normalization_errors",
})
_SEMANTIC_PLAN_FIELDS = frozenset({
    "filters", "temporal_filters", "dimensions", "measures", "joins",
    "ordering", "limit", "output_columns", "null_policy", "table_roles",
})
_BINDING_FIELDS = frozenset({
    "table", "column", "columns", "operator", "value",
    "operation", "distinct", "tables", "keys", "how", "direction",
    "requirement", "output", "evidence",
})


def _strings(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {item for child in value.values() for item in _strings(child)}
    if isinstance(value, (list, tuple, set)):
        return {item for child in value for item in _strings(child)}
    return {value} if isinstance(value, str) and len(value) >= 8 else set()


def _scrub_sensitive(value: Any, sensitive: set[str]) -> Any:
    if isinstance(value, Mapping):
        return {key: _scrub_sensitive(item, sensitive) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_sensitive(item, sensitive) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_sensitive(item, sensitive) for item in value)
    if isinstance(value, str):
        marker = re.search(
            r"(?:benchmark|reference|gold).{0,40}(?:secret|do.?not.?leak)|do.?not.?leak",
            value, re.IGNORECASE,
        )
        if marker or any(secret and secret in value for secret in sensitive):
            return "[excluded_untrusted_value]"
    return value


def _disallowed_field_names(value: Any) -> set[str]:
    blocked = re.compile(
        r"(?:reference|gold|benchmark|expected|evaluator|semantic.?judge|history|reasoning)",
        re.IGNORECASE,
    )
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if blocked.search(str(key)):
                found.add(str(key))
            found.update(_disallowed_field_names(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            found.update(_disallowed_field_names(child))
    return found


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
        if key == "coder_brief" and isinstance(value, Mapping):
            clean[key] = {
                field: value[field] for field in _CODER_BRIEF_FIELDS
                if field in value
            }
        elif key == "semantic_plan" and isinstance(value, Mapping):
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
        forbidden = list(_disallowed_field_names(table_metadata or {}))
        forbidden.extend(_disallowed_field_names(selection_plan or {}))
        sensitive: set[str] = set()
        allowed_source = {"question"}
        for key in (source_payload or {}):
            if str(key).casefold() not in allowed_source:
                forbidden.append(str(key))
                sensitive.update(_strings((source_payload or {})[key]))
        metadata = _scrub_sensitive(metadata, sensitive)
        clean_plan = _scrub_sensitive(
            _allowlisted_plan(selection_plan or {}), sensitive
        )
        return cls(
            question=question,
            selected_tables=tuple(selected_tables),
            table_metadata=metadata,
            selection_plan=clean_plan,
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
