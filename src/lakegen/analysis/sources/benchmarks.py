"""The benchmark question files under benchmark/, by lake."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lakegen.core.config import BASE_DIR
from lakegen.experiments.questions import load_cases


def benchmark_path(lake: str, question_set: str = "100q") -> Path:
    return BASE_DIR / "benchmark" / f"{question_set}_{lake}.json"


def benchmark_cases(lake: str, question_set: str = "100q") -> dict[str, dict[str, Any]]:
    """The cases of ``benchmark/<set>_<lake>.json`` by their id (as a string)."""
    return {str(case["id"]): case for case in load_cases(benchmark_path(lake, question_set).resolve())}


def gold_is_any_of(case: dict[str, Any]) -> bool:
    """Whether any one of the case's gold tables answers it (Chicago's alternatives), not all of them."""
    return (case.get("pneuma") or {}).get("gold_semantics") == "any"
