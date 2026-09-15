"""Canonical, backend-independent retrieval-intent contract."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IntentFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    operator: str
    value: Any


class IntentOrder(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    direction: Literal["asc", "desc"]


class JoinRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    left: str
    right: str
    keys: list[str] = Field(default_factory=list)


class RetrievalIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["resolved", "unresolved"]
    concepts: list[str] = Field(max_length=2)
    entities: list[str]
    # Asked for only when retrieval matches cell contents (grep_values): the
    # values to search for, each kept whole. Optional, so every other mode's
    # intent is unchanged.
    search_values: list[str] = Field(default_factory=list)
    measures: list[str]
    filters: list[IntentFilter]
    time_constraints: list[IntentFilter]
    group_by: list[str]
    order_by: list[IntentOrder]
    limit: int | None = Field(default=None, gt=0)
    join_requirements: list[JoinRequirement]
    missing_evidence: list[str]

    @field_validator(
        "concepts", "entities", "search_values", "measures", "group_by",
        "missing_evidence", mode="before",
    )
    @classmethod
    def normalize_strings(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError("field must be a list")
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("items must be strings")
            normalized = " ".join(item.split())
            if normalized and normalized.casefold() not in {x.casefold() for x in result}:
                result.append(normalized)
        return result

    @field_validator("missing_evidence")
    @classmethod
    def status_has_missing_evidence(cls, value: list[str], info):
        if info.data.get("status") == "unresolved" and not value:
            raise ValueError("unresolved intents require missing_evidence")
        return value

    @model_validator(mode="after")
    def validate_status_contract(self):
        if self.status == "resolved" and not self.concepts:
            raise ValueError("resolved intents require at least one concept")
        return self

    @property
    def keywords(self) -> list[str]:
        return list(self.concepts)

    def search_terms(self, value_search: bool) -> list[str]:
        """What retrieval searches with: the listed cell values, or the concepts."""
        return list(self.search_values) if value_search else self.keywords


def parse_retrieval_intent(response: str) -> RetrievalIntent:
    match = re.fullmatch(r"\s*RETRIEVAL_INTENT:\s*(\{.*\})\s*", response, re.DOTALL)
    if match is None:
        raise ValueError("invalid RETRIEVAL_INTENT envelope")
    try:
        payload = json.loads(match.group(1))
        return RetrievalIntent.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid retrieval_intent: {exc}") from exc


def intent_entities(response: str) -> list[str]:
    """Entities from a raw ``RETRIEVAL_INTENT`` envelope, or none.

    The Pneuma-Seeker content search wants strings that appear verbatim in
    tables, which is exactly what the discovery agent already extracted. This
    reads them back out of the envelope Phase 1 returns, so no call site has to
    change shape and no second LLM call is made. An unresolved or unparsable
    intent yields no entities, and the retriever falls back to its tokenizer.
    """
    try:
        intent = parse_retrieval_intent(response)
    except ValueError:
        return []
    return list(intent.entities) if intent.status == "resolved" else []
