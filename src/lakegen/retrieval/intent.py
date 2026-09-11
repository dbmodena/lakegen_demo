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
    measures: list[str]
    filters: list[IntentFilter]
    time_constraints: list[IntentFilter]
    group_by: list[str]
    order_by: list[IntentOrder]
    limit: int | None = Field(default=None, gt=0)
    join_requirements: list[JoinRequirement]
    missing_evidence: list[str]

    @field_validator("concepts", "entities", "measures", "group_by", "missing_evidence", mode="before")
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


def parse_retrieval_intent(response: str) -> RetrievalIntent:
    match = re.fullmatch(r"\s*RETRIEVAL_INTENT:\s*(\{.*\})\s*", response, re.DOTALL)
    if match is None:
        raise ValueError("invalid RETRIEVAL_INTENT envelope")
    try:
        payload = json.loads(match.group(1))
        return RetrievalIntent.model_validate(payload)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid retrieval_intent: {exc}") from exc
