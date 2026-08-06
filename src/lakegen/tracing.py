"""Provider-neutral, typed helpers for structured run telemetry."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


PhaseName = Literal["discovery", "code", "result"]


class TelemetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMPhaseRecord(TelemetryModel):
    phase: PhaseName
    phase_invocation_count: int = Field(ge=0)
    provider_call_count: int | None = Field(default=None, ge=0)
    total_tokens: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    usage_breakdown_available: bool = False
    # Backward-compatible alias. It has always counted phase-function
    # invocations, not necessarily calls made internally by the provider.
    call_count: int = Field(ge=0)


class HumanGate(StrEnum):
    KEYWORD_APPROVAL = "keyword_approval"
    DATASET_APPROVAL = "dataset_approval"
    KEYWORD_HINT = "keyword_hint"
    DATASET_HINT = "dataset_hint"
    FORCE_EXECUTION_CONFIRMATION = "force_execution_confirmation"


class HumanIntervention(TelemetryModel):
    phase: PhaseName
    gate: HumanGate
    type: Literal["approval", "hint"]
    elapsed_seconds: float = Field(ge=0.0)
    approved: bool | None = None
    provided: bool | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "HumanIntervention":
        if self.type == "approval" and (
            self.approved is None or self.provided is not None
        ):
            raise ValueError("approval interventions require only 'approved'")
        if self.type == "hint" and (
            self.provided is None or self.approved is not None
        ):
            raise ValueError("hint interventions require only 'provided'")
        return self


class HumanInterventionRecorder:
    """Run-local recorder that never stores free-form answers or hints."""

    def __init__(self) -> None:
        self._events: list[HumanIntervention] = []

    def record_approval(
        self,
        *,
        phase: PhaseName,
        gate: HumanGate,
        approved: bool,
        elapsed_seconds: float,
    ) -> None:
        self._events.append(HumanIntervention(
            phase=phase,
            gate=gate,
            type="approval",
            approved=approved,
            elapsed_seconds=max(0.0, elapsed_seconds),
        ))

    def record_hint(
        self,
        *,
        phase: PhaseName,
        gate: HumanGate,
        provided: bool,
        elapsed_seconds: float,
    ) -> None:
        self._events.append(HumanIntervention(
            phase=phase,
            gate=gate,
            type="hint",
            provided=provided,
            elapsed_seconds=max(0.0, elapsed_seconds),
        ))

    def to_list(self) -> list[dict[str, Any]]:
        return [event.model_dump(mode="json") for event in self._events]


_TOOL_PATTERNS = (
    re.compile(r"Tool:\s*`([^`]+)`"),
    re.compile(r"ToolCall:\s*([\w.-]+)"),
    re.compile(r"tool_name=['\"]([\w.-]+)['\"]"),
)


def summarize_tool_calls(trace: str) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for pattern in _TOOL_PATTERNS:
        for name in pattern.findall(trace or ""):
            counts[name] = counts.get(name, 0) + 1
    return [
        {"phase": "discovery", "type": name, "count": count}
        for name, count in sorted(counts.items())
    ]


def build_llm_phase_records(
    *,
    total_tokens: Mapping[str, int],
    phase_invocations: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Return one non-overlapping record for each logical pipeline phase."""

    records = []
    for phase in ("discovery", "code", "result"):
        invocations = max(0, int(phase_invocations.get(phase, 0)))
        record = LLMPhaseRecord(
            phase=phase,
            phase_invocation_count=invocations,
            provider_call_count=None,
            total_tokens=max(0, int(total_tokens.get(phase, 0))),
            prompt_tokens=None,
            completion_tokens=None,
            usage_breakdown_available=False,
            call_count=invocations,
        )
        records.append(record.model_dump(mode="json"))
    return records


def _identity_values(value: Any) -> set[str]:
    if value is None:
        return set()
    text = str(value).strip().casefold()
    if not text:
        return set()
    values = {text, Path(text).stem}
    if ":" in text:
        suffix = text.rsplit(":", 1)[-1]
        values.update({suffix, Path(suffix).stem})
    return values


def summarize_final_ranking(
    retrieval_runs: Sequence[Mapping[str, Any]],
    selected_tables: Sequence[str],
) -> list[dict[str, Any]]:
    """Summarize only the final retrieval attempt; full history remains elsewhere."""

    if not retrieval_runs:
        return []
    indexed_runs = list(enumerate(retrieval_runs, start=1))
    final_index, final_run = max(
        indexed_runs,
        key=lambda item: int(item[1].get("retrieval_attempt") or item[0]),
    )
    attempt = int(final_run.get("retrieval_attempt") or final_index)
    mode = str(final_run.get("mode") or "unknown")
    selected_identities = {
        identity
        for table in selected_tables
        for identity in _identity_values(table)
    }
    ranking: list[dict[str, Any]] = []
    for hit in final_run.get("hits", []):
        if not isinstance(hit, Mapping):
            continue
        hit_identities = {
            identity
            for key in ("resource_id", "dataset_id", "title", "document_key")
            for identity in _identity_values(hit.get(key))
        }
        ranking.append({
            "attempt": attempt,
            "mode": mode,
            "resource_id": hit.get("resource_id"),
            "rank": hit.get("rank"),
            "score": hit.get("score"),
            "selected": bool(selected_identities & hit_identities),
        })
    return ranking
