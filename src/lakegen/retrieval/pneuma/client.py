"""The Pneuma service: the client of its HTTP index and what its answer holds.

The service itself (the index builder and the HTTP server) is ``pneuma_engine/``
at the repository root, run in its own environment; LakeGen only talks to it
over HTTP.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import requests

from lakegen.config.retrieval import (
    RetrievalConfig,
)


class PneumaClient(Protocol):
    """Queries a Pneuma index; the answer is its JSON, decoded or not."""

    def query_index(
        self, index_name: str, queries: str, *, k: int, n: int, alpha: float
    ) -> str | dict[str, Any]:
        """Ask index ``index_name`` for the tables relevant to ``queries``.

        Args:
            index_name: The index of the lake.
            queries: The question.
            k: How many documents of tables to retrieve.
            n: The multiple of ``k`` pooled for Pneuma's hybrid retrieval.
            alpha: The weight of the full-text retriever against the vector one
                (lower favours the vectors).

        Returns:
            The service's JSON answer, decoded or as text.
        """
        ...


def create_pneuma_client(config: RetrievalConfig) -> PneumaClient:
    """Create a lightweight client for the independently hosted backend."""
    return PneumaHttpClient(config.pneuma_base_url, timeout=config.pneuma_timeout_seconds)


class PneumaHttpClient:
    """The client of the Pneuma HTTP service's ``POST /query``.

    Args:
        base_url: The service's base URL.
        timeout: The request timeout, in seconds.

    Attributes:
        query_url: The ``/query`` URL.
        timeout: The request timeout, in seconds.
    """

    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.query_url = base_url.rstrip("/") + "/query"
        self.timeout = timeout

    def query_index(
        self, index_name: str, queries: str, *, k: int, n: int, alpha: float
    ) -> dict[str, Any]:
        """Post the query and return the decoded JSON answer.

        Args:
            index_name: The index of the lake.
            queries: The question.
            k: How many documents of tables to retrieve.
            n: The multiple of ``k`` pooled for Pneuma's hybrid retrieval.
            alpha: The weight of the full-text retriever against the vector one
                (lower favours the vectors).

        Returns:
            The decoded answer.

        Raises:
            RuntimeError: When the service cannot be reached, answers with an
                error status, or answers with something that is not JSON.
        """
        try:
            response = requests.post(
                self.query_url,
                json={
                    "index_name": index_name,
                    "query": queries,
                    "k": k,
                    "n": n,
                    "alpha": alpha,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Pneuma service unavailable at {self.query_url}: {exc}") from exc


def _table_ids(payload: str | dict[str, Any]) -> list[str]:
    """The ordered table ids of a Pneuma answer.

    Args:
        payload: The answer, decoded or as text.

    Returns:
        The non-blank ``retrieved_tables``, in Pneuma's order.

    Raises:
        RuntimeError: When the answer is not JSON, reports a failure, or does not
            hold exactly one result with a ``retrieved_tables`` list.
    """
    try:
        decoded = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise RuntimeError("Pneuma returned invalid JSON") from exc
    if not isinstance(decoded, dict) or decoded.get("status") != "SUCCESS":
        message = decoded.get("message") if isinstance(decoded, dict) else None
        raise RuntimeError(f"Pneuma query failed: {message or 'invalid response'}")
    data = decoded.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise RuntimeError("Pneuma returned an unexpected query response")
    tables = data[0].get("retrieved_tables")
    if not isinstance(tables, list):
        raise RuntimeError("Pneuma response has no retrieved_tables list")
    return [str(value) for value in tables if str(value).strip()]
