from __future__ import annotations

from collections.abc import Sequence
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
