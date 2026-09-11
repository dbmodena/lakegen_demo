"""Optional adapter for the original MIT-licensed Pneuma implementation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import Any, Protocol

import requests

from lakegen.retrieval.config import RetrievalConfig
from lakegen.retrieval.models import RetrievalHit, document_key


class PneumaClient(Protocol):
    def query_index(
        self, index_name: str, queries: str, *, k: int, n: int, alpha: float
    ) -> str | dict[str, Any]: ...


DocumentResolver = Callable[[str], dict[str, Any] | None]


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


class SolrPneumaDocumentResolver:
    """Map Pneuma's path-like table IDs back to LakeGen catalog documents."""

    FIELDS = (
        "id", "resource_id", "dataset_id", "title", "description", "tags",
        "columns.name", "columns.description", "columns.type", "schema",
        "dataset_url", "download_url", "url", "permalink", "link", "source",
        "portal", "provenance",
    )

    def __init__(self, solr: Any) -> None:
        self._documents = list(
            solr.iter_documents(fields=self.FIELDS, sort_field="resource_id")
        )

    @staticmethod
    def _aliases(value: Any) -> set[str]:
        if value is None or not str(value).strip():
            return set()
        text = str(value).strip()
        path = Path(text)
        return {text.casefold(), path.name.casefold(), path.stem.casefold()}

    def __call__(self, table_id: str) -> dict[str, Any] | None:
        wanted = self._aliases(table_id)
        for document in self._documents:
            aliases: set[str] = set()
            for field in (
                "id", "resource_id", "dataset_id", "download_url", "url", "source"
            ):
                aliases.update(self._aliases(document.get(field)))
            if wanted & aliases:
                return document
        return None


class PneumaRetriever:
    """Use Pneuma's original ranking while returning LakeGen RetrievalHit values."""

    def __init__(
        self,
        config: RetrievalConfig,
        resolver: DocumentResolver,
        *,
        client: PneumaClient | None = None,
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.client = client or create_pneuma_client(config)

    def retrieve(self, question: str, *, top_k: int) -> list[RetrievalHit]:
        response = self.client.query_index(
            self.config.pneuma_index_name,
            question,
            k=top_k,
            n=self.config.candidate_multiplier,
            alpha=self.config.alpha,
        )
        hits: list[RetrievalHit] = []
        seen: set[str] = set()
        for table_id in _table_ids(response):
            document = self.resolver(table_id)
            if document is None:
                continue
            key = document_key(document)
            if key in seen:
                continue
            seen.add(key)
            rank = len(hits) + 1
            # Pneuma 0.0.4 exposes ordered table IDs but not final table scores.
            hits.append(RetrievalHit(document=document, score=1.0 / rank, rank=rank))
            if len(hits) >= top_k:
                break
        return hits
