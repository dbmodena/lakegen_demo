"""Canonical, backend-independent retrieval-intent contract."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IntentFilter(BaseModel):
    """A condition on a field, as the intent states it.

    Attributes:
        field: The field the condition is on.
        operator: The comparison, as the LLM wrote it.
        value: The value compared with.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    operator: str
    value: Any


class IntentOrder(BaseModel):
    """A sort the answer needs.

    Attributes:
        field: The field sorted on.
        direction: ``asc`` or ``desc``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    field: str
    direction: Literal["asc", "desc"]


class JoinRequirement(BaseModel):
    """Two datasets the answer joins, and the keys it joins them on.

    Attributes:
        left: The first dataset.
        right: The second dataset.
        keys: The join keys; may be empty.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    left: str
    right: str
    keys: list[str] = Field(default_factory=list)


class RetrievalIntent(BaseModel):
    """What the discovery agent understood a question to need from the lake.

    The agent returns it as JSON after ``RETRIEVAL_INTENT:``; the validators
    canonicalise the shapes LLMs commonly write and enforce the status contract
    (a resolved intent has a concept, an unresolved one says what is missing).

    Attributes:
        status: ``resolved``, or ``unresolved`` when the question lacks evidence.
        concepts: The concepts to search for; the keywords of retrieval.
        entities: Named entities of the question, verbatim.
        search_values: Cell values to search for, each kept whole, for the modes
            that match cell contents directly.
        measures: What the answer measures.
        filters: The conditions on fields.
        time_constraints: The conditions on time.
        group_by: The fields the answer groups by.
        order_by: The sorts the answer needs.
        limit: How many rows the answer keeps, when it says.
        join_requirements: The joins the answer needs.
        missing_evidence: What an unresolved intent lacks.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    status: Literal["resolved", "unresolved"]
    concepts: list[str]
    entities: list[str]
    # Reserved for retrieval modes that match cell contents directly: the
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

    @model_validator(mode="before")
    @classmethod
    def normalize_join_shapes(cls, value: Any) -> Any:
        """Canonicalize common, unambiguous LLM join representations.

        Args:
            value: The raw input of the model.

        Returns:
            The input with its ``join_requirements`` canonicalised, or the input
            unchanged when it has no list of them.
        """
        if not isinstance(value, dict):
            return value
        raw_items = value.get("join_requirements")
        if not isinstance(raw_items, list):
            return value
        normalized: list[Any] = []
        singletons: list[tuple[str, list[str]]] = []
        for item in raw_items:
            if not isinstance(item, dict) or set(item) <= {"left", "right", "keys"}:
                normalized.append(item)
                continue
            canonical = _two_sided_join(item)
            if canonical is not None:
                normalized.append(canonical)
                continue
            singleton = _one_sided_join(item)
            if singleton is not None:
                singletons.append(singleton)
                continue
            normalized.append(item)
        if singletons:
            if len(singletons) == 2:
                shared = [key for key in singletons[0][1] if key in singletons[1][1]]
                normalized.append(
                    {
                        "left": singletons[0][0],
                        "right": singletons[1][0],
                        "keys": shared or list(dict.fromkeys(singletons[0][1] + singletons[1][1])),
                    }
                )
            else:
                # A singleton cannot describe a join without inventing a side.
                normalized.extend(
                    {"dataset": dataset, "keys": keys} for dataset, keys in singletons
                )
        return {**value, "join_requirements": normalized}

    @field_validator(
        "concepts",
        "entities",
        "search_values",
        "measures",
        "group_by",
        "missing_evidence",
        mode="before",
    )
    @classmethod
    def normalize_strings(cls, value: list[str]) -> list[str]:
        """Collapse the whitespace of each string and drop empty and case-insensitive repeats.

        Args:
            value: The raw list.

        Returns:
            The cleaned strings, in their first order.

        Raises:
            ValueError: When the value is not a list of strings.
        """
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
        """Require ``missing_evidence`` on an unresolved intent.

        Args:
            value: The missing evidence.
            info: The validation info, holding the fields validated so far.

        Returns:
            The value unchanged.

        Raises:
            ValueError: When the intent is unresolved and names no missing evidence.
        """
        if info.data.get("status") == "unresolved" and not value:
            raise ValueError("unresolved intents require missing_evidence")
        return value

    @model_validator(mode="after")
    def validate_status_contract(self):
        """Require a concept on a resolved intent.

        Returns:
            The intent unchanged.

        Raises:
            ValueError: When the intent is resolved and has no concept.
        """
        if self.status == "resolved" and not self.concepts:
            raise ValueError("resolved intents require at least one concept")
        return self

    @property
    def keywords(self) -> list[str]:
        """The concepts, as the keywords retrieval searches with."""
        return list(self.concepts)

    def search_terms(self, value_search: bool) -> list[str]:
        """What retrieval searches with: the listed cell values, or the concepts.

        Args:
            value_search: Whether the mode matches cell contents directly.

        Returns:
            The search values when ``value_search``, else the concepts.
        """
        return list(self.search_values) if value_search else self.keywords


def _two_sided_join(item: dict[str, Any]) -> dict[str, Any] | None:
    """A join an LLM wrote with both sides in one item, in canonical form.

    Two shapes are recognised: ``datasets`` as a pair of names with ``keys``
    (or ``key``), and ``left_dataset``/``right_dataset`` with
    ``left_key``/``right_key``.

    Args:
        item: One raw join requirement.

    Returns:
        ``{"left", "right", "keys"}``, or None when the item has neither shape.
    """
    datasets = item.get("datasets")
    keys = item.get("keys", item.get("key", []))
    if isinstance(keys, str):
        keys = [keys]
    if (
        isinstance(datasets, list)
        and len(datasets) == 2
        and all(isinstance(part, str) and part.strip() for part in datasets)
        and isinstance(keys, list)
        and all(isinstance(key, str) for key in keys)
    ):
        return {"left": datasets[0], "right": datasets[1], "keys": keys}
    if all(name in item for name in ("left_dataset", "right_dataset")):
        join_keys = [item.get("left_key"), item.get("right_key")]
        join_keys = list(
            dict.fromkeys(key for key in join_keys if isinstance(key, str) and key.strip())
        )
        return {
            "left": item["left_dataset"],
            "right": item["right_dataset"],
            "keys": join_keys,
        }
    return None


def _one_sided_join(item: dict[str, Any]) -> tuple[str, list[str]] | None:
    """One side of a join, as an LLM writes it: ``dataset`` with ``key`` (or ``keys``).

    Args:
        item: One raw join requirement.

    Returns:
        The dataset and its keys, or None when the item is not that shape.
    """
    dataset = item.get("dataset")
    key = item.get("key", item.get("keys", []))
    if isinstance(key, str):
        key = [key]
    if (
        isinstance(dataset, str)
        and isinstance(key, list)
        and all(isinstance(part, str) for part in key)
    ):
        return dataset, key
    return None


def parse_retrieval_intent(response: str) -> RetrievalIntent:
    """Parse the JSON object that follows ``RETRIEVAL_INTENT:`` in an LLM response.

    Args:
        response: The raw response; only a closing code fence may follow the object.

    Returns:
        The validated intent.

    Raises:
        ValueError: When the envelope is missing, the JSON is invalid, something
            else follows it, or the intent fails validation.
    """
    match = re.search(r"\bRETRIEVAL_INTENT:\s*", response)
    if match is None:
        raise ValueError("invalid RETRIEVAL_INTENT envelope")
    try:
        payload, end = json.JSONDecoder().raw_decode(response, match.end())
        trailing = response[end:].strip()
        if trailing and trailing not in {"```", "`"}:
            raise ValueError("unexpected content after RETRIEVAL_INTENT object")
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

    Args:
        response: The raw response of the discovery agent.

    Returns:
        The entities of a resolved intent; empty otherwise.
    """
    try:
        intent = parse_retrieval_intent(response)
    except ValueError:
        return []
    return list(intent.entities) if intent.status == "resolved" else []
