from __future__ import annotations

from collections.abc import Sequence
import json
import math
from typing import Any

import requests


class LocalSolrClient:
    _COLUMN_DEFAULTS = {"description": None}

    def __init__(
        self,
        core: str,
        base_url: str = "http://localhost:8983/solr",
        timeout: float = 30.0,
    ) -> None:
        self.core = core.strip("/")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def core_url(self) -> str:
        return f"{self.base_url}/{self.core}"

    @property
    def select_url(self) -> str:
        return f"{self.core_url}/select"

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        return [value]

    @classmethod
    def _restore_columns(cls, doc: dict[str, Any]) -> dict[str, Any]:
        restored_doc: dict[str, Any] = {}
        column_values: dict[str, list[Any]] = {}

        for key, value in doc.items():
            if key.startswith("columns."):
                column_values[key.removeprefix("columns.")] = cls._as_list(value)
            else:
                restored_doc[key] = value

        if not column_values:
            return restored_doc

        column_count = max(len(values) for values in column_values.values())
        columns: list[dict[str, Any]] = []
        for index in range(column_count):
            column = {
                field_name: values[index]
                for field_name, values in column_values.items()
                if index < len(values)
            }
            for field_name, default in cls._COLUMN_DEFAULTS.items():
                column.setdefault(field_name, default)
            columns.append(column)

        restored_doc["columns"] = columns
        return restored_doc

    @classmethod
    def _restore_response_docs(cls, result: dict[str, Any]) -> dict[str, Any]:
        response = result.get("response")
        if not isinstance(response, dict):
            return result

        docs = response.get("docs")
        if not isinstance(docs, list):
            return result

        response["docs"] = [
            cls._restore_columns(doc) if isinstance(doc, dict) else doc
            for doc in docs
        ]
        return result

    def select(
        self,
        tokens: Sequence[str],
        *,
        def_type: str = "edismax",
        q_op: str = "AND",
        indent: bool = True,
        **params: Any,
    ) -> dict[str, Any]:
        if isinstance(tokens, str):
            raise TypeError("tokens must be a sequence of strings, not a string")

        response = requests.get(
            self.select_url,
            params={
                "defType": def_type,
                "indent": str(indent).lower(),
                "q.op": q_op,
                "q": " ".join(tokens),
                "wt": "json",
                **params,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._restore_response_docs(response.json())

    def knn_select(
        self,
        vector: Sequence[float],
        *,
        vector_field: str,
        top_k: int,
        rows: int | None = None,
        filters: Sequence[str] = (),
        **params: Any,
    ) -> dict[str, Any]:
        """Run a Solr KNN query against a ``DenseVectorField``.

        Solr/Lucene computes the vector similarity; this client only validates
        and serializes the query vector. The configured Solr field must use
        ``similarityFunction=cosine`` for LakeGen's semantic experiments.
        """
        if not vector_field or not vector_field.replace("_", "").isalnum():
            raise ValueError("vector_field must be a simple Solr field name")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        values = [float(value) for value in vector]
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("vector must contain finite numeric values")

        query = (
            f"{{!knn f={vector_field} topK={top_k}}}"
            + json.dumps(values, separators=(",", ":"))
        )
        request_params: dict[str, Any] = {
            "q": query,
            "rows": rows if rows is not None else top_k,
            "fl": "*,score",
            "wt": "json",
            **params,
        }
        if filters:
            request_params["fq"] = list(filters)

        response = requests.get(
            self.select_url,
            params=request_params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._restore_response_docs(response.json())

    def iter_documents(
        self,
        *,
        fields: Sequence[str],
        batch_size: int = 100,
        sort_field: str = "resource_id",
    ):
        """Yield all documents using Solr's cursor API."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        cursor = "*"
        while True:
            response = requests.get(
                self.select_url,
                params={
                    "q": "*:*",
                    "fl": ",".join(fields),
                    "rows": batch_size,
                    "sort": f"{sort_field} asc",
                    "cursorMark": cursor,
                    "wt": "json",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = self._restore_response_docs(response.json())
            docs = payload.get("response", {}).get("docs", [])
            for doc in docs:
                if isinstance(doc, dict):
                    yield doc

            next_cursor = payload.get("nextCursorMark")
            if not docs or not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

    def schema(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.core_url}/schema",
            params={"wt": "json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def update_schema(self, commands: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(
            f"{self.core_url}/schema",
            params={"wt": "json"},
            json=commands,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def update_documents(
        self,
        documents: Sequence[dict[str, Any]],
        *,
        commit: bool = False,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.core_url}/update",
            params={"commit": str(commit).lower(), "wt": "json"},
            json=list(documents),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def commit(self) -> dict[str, Any]:
        response = requests.post(
            f"{self.core_url}/update",
            params={"wt": "json"},
            json={"commit": {}},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
