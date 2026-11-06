"""Allowlisted, benchmark-blind context supplied to the coding agent."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_METADATA_FIELDS = frozenset(
    {
        "title",
        "description",
        "publisher",
        "resource_name",
        "topics",
        "columns",
    }
)
_COLUMN_FIELDS = frozenset({"name", "description", "data_type", "dtype"})
_PLAN_FIELDS = frozenset(
    {
        "requirement_coverage",
        "table_roles",
        "combination_strategy",
        "uncovered_requirements",
        "alternatives_rejected",
        "semantic_plan",
        "recovered_from_existing_discovery_context",
        "coder_brief",
        "requirements",
        "requirement_ledger",
    }
)
_LEDGER_ITEM_FIELDS = frozenset(
    {
        "kind",
        "request",
        "status",
        "evidence",
        "role",
        "computation",
        "shape",
    }
)
_REQUIREMENT_FIELDS = frozenset(
    {
        "grouping",
        "measures",
        "filters",
        "temporal_filters",
        "ordering",
        "limit",
        "joins",
        "output_columns",
        "null_policy",
    }
)
_CODER_BRIEF_FIELDS = frozenset(
    {
        "tables",
        "selected_columns",
        "task",
        "filters",
        "operations",
        "temporal_filters",
        "dimensions",
        "measures",
        "output_columns",
        "null_policy",
        "table_roles",
        "ordering",
        "limit",
        "joins",
        "combination_strategy",
        "normalization_errors",
    }
)
_SEMANTIC_PLAN_FIELDS = frozenset(
    {
        "filters",
        "temporal_filters",
        "dimensions",
        "measures",
        "joins",
        "ordering",
        "limit",
        "output_columns",
        "null_policy",
        "table_roles",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "table",
        "column",
        "columns",
        "operator",
        "value",
        "operation",
        "distinct",
        "tables",
        "keys",
        "how",
        "direction",
        "requirement",
        "output",
        "evidence",
    }
)


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
            value,
            re.IGNORECASE,
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
    """Derive ordering and limit cues without predicting a result type."""
    lowered = question.casefold()
    limit = None
    match = re.search(r"\btop\s+(\d+)\b", lowered)
    if match:
        limit = int(match.group(1))
    ranked = bool(re.search(r"\b(?:top|bottom|highest|lowest|most|least)\b", lowered))
    return {
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
                for column in item
                if isinstance(column, Mapping)
            ]
        elif isinstance(item, (str, int, float, bool, list, tuple)) or item is None:
            clean[key] = item
    return clean


def _allowlisted_ledger_evidence(evidence: Any) -> Any:
    if isinstance(evidence, Mapping):
        return {key: evidence[key] for key in _BINDING_FIELDS if key in evidence}
    if isinstance(evidence, list):
        return [
            {key: entry[key] for key in _BINDING_FIELDS if key in entry}
            if isinstance(entry, Mapping)
            else entry
            for entry in evidence
        ]
    return evidence


def _allowlisted_requirement_ledger(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        entry = {key: item[key] for key in _LEDGER_ITEM_FIELDS if key in item}
        if "evidence" in entry:
            entry["evidence"] = _allowlisted_ledger_evidence(entry["evidence"])
        cleaned.append(entry)
    return cleaned


def _allowlisted_semantic_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """The semantic plan's allowlisted fields, list entries cut to their binding fields."""
    semantic: dict[str, Any] = {}
    for semantic_key in _SEMANTIC_PLAN_FIELDS:
        if semantic_key not in value:
            continue
        item = value.get(semantic_key)
        if isinstance(item, list):
            semantic[semantic_key] = [
                {k: entry[k] for k in _BINDING_FIELDS if k in entry}
                if isinstance(entry, Mapping)
                else entry
                for entry in item
            ]
        elif isinstance(item, Mapping):
            semantic[semantic_key] = dict(item)
        elif item is None or isinstance(item, (str, int, float, bool)):
            semantic[semantic_key] = item
    return semantic


def _allowlisted_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key in _PLAN_FIELDS:
        if key not in plan:
            continue
        value = plan[key]
        if key == "requirements" and isinstance(value, Mapping):
            clean[key] = {field: value[field] for field in _REQUIREMENT_FIELDS if field in value}
        elif key == "coder_brief" and isinstance(value, Mapping):
            clean[key] = {field: value[field] for field in _CODER_BRIEF_FIELDS if field in value}
        elif key == "requirement_ledger":
            clean[key] = _allowlisted_requirement_ledger(value)
        elif key == "semantic_plan" and isinstance(value, Mapping):
            clean[key] = _allowlisted_semantic_plan(value)
        elif isinstance(value, Mapping):
            clean[key] = dict(value)
        elif isinstance(value, (list, str, int, float, bool)) or value is None:
            clean[key] = value
    return clean


@dataclass(frozen=True)
class CoderContext:
    """What the coder may see: the question, the selected tables and allowlisted metadata and plan.

    Only allowlisted fields pass. Field names in the metadata or plan that
    mention the reference, gold, benchmark, expected values, the evaluator,
    the semantic judge, history or reasoning are recorded as excluded, and
    the values of source fields other than the question are scrubbed.

    Attributes:
        question: The user's question.
        selected_tables: The selected table file names.
        table_metadata: The allowlisted catalogue metadata of each selected table.
        selection_plan: The allowlisted selection plan.
        output_shape: The ordering and limit cues derived from the question.
        execution_error: The previous execution error, if any.
        excluded_fields: The field names kept from the coder.
    """

    question: str
    selected_tables: tuple[str, ...]
    table_metadata: dict[str, dict[str, Any]]
    selection_plan: dict[str, Any]
    output_shape: dict[str, Any]
    execution_error: dict[str, Any] = field(default_factory=dict)
    excluded_fields: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        question: str,
        selected_tables: list[str],
        table_metadata: Mapping[str, Any] | None,
        selection_plan: Mapping[str, Any] | None,
        source_payload: Mapping[str, Any] | None = None,
        execution_error: Mapping[str, Any] | None = None,
    ) -> CoderContext:
        """Build the coder's context from the selection, keeping only what is allowlisted.

        Args:
            question: The user's question.
            selected_tables: The selected table file names.
            table_metadata: The catalogue metadata by file name or stem.
            selection_plan: The selection's plan and coder brief.
            source_payload: The source record's fields; any field but the
                question is excluded and its string values are scrubbed from
                the metadata and plan.
            execution_error: The previous execution error, if any.

        Returns:
            The context.
        """
        metadata = {
            table: _allowlisted_metadata(
                (table_metadata or {}).get(
                    table, (table_metadata or {}).get(table.rsplit(".", 1)[0], {})
                )
            )
            for table in selected_tables
        }
        forbidden = list(_disallowed_field_names(table_metadata or {}))
        forbidden.extend(_disallowed_field_names(selection_plan or {}))
        sensitive: set[str] = set()
        allowed_source = {"question"}
        for key in source_payload or {}:
            if str(key).casefold() not in allowed_source:
                forbidden.append(str(key))
                sensitive.update(_strings((source_payload or {})[key]))
        metadata = _scrub_sensitive(metadata, sensitive)
        clean_plan = _scrub_sensitive(_allowlisted_plan(selection_plan or {}), sensitive)
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
        """What the context included and excluded, for the run's telemetry."""
        return {
            "included_fields": [
                "question",
                "selected_tables",
                "table_metadata",
                "selection_plan",
                "output_shape",
                "execution_error",
            ],
            "evidence_origins": [
                "question",
                "runtime_retrieval",
                "runtime_table_inspection",
                "runtime_execution",
            ],
            "excluded_field_names": list(self.excluded_fields),
            "context_filtered": bool(self.excluded_fields),
            "reference_accessed_by_coder": False,
        }
