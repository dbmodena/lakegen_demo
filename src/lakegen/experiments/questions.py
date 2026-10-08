"""The benchmark question files and their slices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_cases(path: Path) -> list[dict[str, Any]]:
    """The cases of a benchmark file (``{"cases": [...]}``) or a bare list of them."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError(f"{path}: expected a list of cases")
    return cases


def parse_slice(text: str | None) -> slice:
    """``"20:"``, ``":20"`` or ``"5:10"`` in Python's notation; None is everything."""
    if not text:
        return slice(None)
    start, _, stop = text.partition(":")
    return slice(int(start) if start.strip() else None, int(stop) if stop.strip() else None)


def select_cases(path: Path, text: str | None) -> list[dict[str, Any]]:
    return load_cases(path)[parse_slice(text)]
