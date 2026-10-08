"""The Pneuma service: the client of its HTTP index and what its answer holds."""

from __future__ import annotations

import json
from typing import Any, Protocol

import requests

from lakegen.retrieval.config import (
    RetrievalConfig,
)


class PneumaClient(Protocol):
    def query_index(
        self, index_name: str, queries: str, *, k: int, n: int, alpha: float
    ) -> str | dict[str, Any]: ...


def create_pneuma_client(config: RetrievalConfig) -> PneumaClient:
    """Create a lightweight client for the independently hosted backend."""
    return HttpPneumaClient(
        config.pneuma_base_url, timeout=config.pneuma_timeout_seconds
    )


class HttpPneumaClient:
    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.query_url = base_url.rstrip("/") + "/query"
        self.timeout = timeout

    def query_index(
        self, index_name: str, queries: str, *, k: int, n: int, alpha: float
    ) -> dict[str, Any]:
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
            raise RuntimeError(
                f"Pneuma service unavailable at {self.query_url}: {exc}"
            ) from exc


def _table_ids(payload: str | dict[str, Any]) -> list[str]:
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
