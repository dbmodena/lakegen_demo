"""The same-context selector comparison (experiments/selector_compare), as stored.

Its results directory holds three files the analyses read:

* ``contexts.jsonl``: one line per (lake, retrieval kind, question): the
  prepared discovery context with its ranked candidates, the question and its
  gold tables, keyed ``<lake>|<kind>|<id>``;
* ``selections.jsonl``: one line per context and selector (``llm``, ``jev``),
  keyed ``<context key>|<selector>``, with the tables selected;
* ``coded/<lake>.jsonl``: the coding harness run on a selected table set,
  keyed ``<lake>|<id>|<sorted tables>``, with the pipeline result.

The files are large (contexts.jsonl holds every candidate's scores), so the
readers stream them and keep one lake at a time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lakegen.core.config import BASE_DIR

RESULTS_DIR = BASE_DIR / "experiments" / "selector_compare" / "results"

# Signs, in a coding record's errors, that the provider failed rather than the
# pipeline (the list of experiments/selector_compare/llm_coding_report.py).
PROVIDER_FAILURES = (
    "RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
    "dynamic_request_limited", "dynamic_token_limited", "Error code: 429", "Error code: 503",
)


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def contexts(lake: str, results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    """The lake's prepared contexts by key."""
    return {row["key"]: row for row in _lines(results_dir / "contexts.jsonl") if row["lake"] == lake}


def selections(lake: str, results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    """The lake's selections by key (``<context key>|<selector>``)."""
    return {row["key"]: row for row in _lines(results_dir / "selections.jsonl") if row["key"].startswith(f"{lake}|")}


def coded(results_dir: Path = RESULTS_DIR) -> Iterator[dict[str, Any]]:
    """Every coding record of every lake, in file order."""
    for path in sorted((results_dir / "coded").glob("*.jsonl")):
        yield from _lines(path)


def provider_failure(record: dict[str, Any]) -> bool:
    """Whether a coding record failed on the provider (credit, rate limit, outage), not the task."""
    result = record.get("result") or {}
    text = json.dumps([
        record.get("error"), result.get("error"), result.get("errors"),
        (result.get("execution_outcome") or {}).get("error"),
    ], default=str)
    return any(sign in text for sign in PROVIDER_FAILURES)
