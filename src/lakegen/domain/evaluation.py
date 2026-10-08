"""What a benchmark question brings for evaluating its run.

A batch question arrives with ``log_fields``: ``SOURCE_*`` entries from the
benchmark file (its id, gold tables, reference result, expected result type)
plus whatever the submitter adds (``JOB_ID``, ``EXECUTION_ATTEMPT``). The
engine used to read the ``SOURCE_*`` keys here and there; ``EvaluationContext``
parses them once and keeps every field for the logs, which still receive
the same keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationContext:
    fields: dict[str, Any] = field(default_factory=dict)
    """Every log-context field as given, ``SOURCE_*`` and the submitter's alike."""
    source_id: str | int | None = None
    expected_result_type: str = ""
    expected_description: str = ""
    reference_result: Any = None
    evaluation_contract: dict[str, Any] = field(default_factory=dict)
    relevant_tables: list[str] = field(default_factory=list)
    """The gold table ids; empty when the question has none (then no ranking metrics)."""

    @classmethod
    def from_log_context(cls, log_context: Mapping[str, Any] | None) -> EvaluationContext:
        fields = dict(log_context or {})
        contract = fields.get("SOURCE_EVALUATION_CONTRACT", {})
        gold = fields.get("SOURCE_RELEVANT_TABLE_IDS")
        return cls(
            fields=fields,
            source_id=fields.get("SOURCE_ID"),
            expected_result_type=(
                str(fields["SOURCE_EXPECTED_RESULT_TYPE"]) if fields.get("SOURCE_EXPECTED_RESULT_TYPE") else ""
            ),
            expected_description=str(fields.get("SOURCE_EXPECTED_RESULT_DESCRIPTION") or ""),
            reference_result=fields.get("SOURCE_REFERENCE_RESULT"),
            evaluation_contract=dict(contract) if isinstance(contract, Mapping) else {},
            relevant_tables=list(gold) if isinstance(gold, list) else [],
        )

    @property
    def field_names(self) -> list[str]:
        return list(self.fields)

    def to_log_fields(self) -> dict[str, Any]:
        """The fields for a log row: the same keys the run was given."""
        return dict(self.fields)
