"""The ``golden`` fixture: compare a payload with its JSON snapshot, or rewrite it.

Run ``pytest --update-golden`` to record new or changed snapshots. See
harness.py for what the snapshots hold and how they are normalised.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import pytest

SNAPSHOTS = Path(__file__).parent / "snapshots"


@pytest.fixture
def golden(request):
    """``golden(name, payload)`` compares ``payload`` with ``snapshots/<name>.json``."""
    update = request.config.getoption("--update-golden")

    def check(name: str, payload: Any) -> None:
        path = SNAPSHOTS / f"{name}.json"
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            return
        if not path.exists():
            pytest.fail(f"no golden snapshot {name}.json; run pytest --update-golden to record it")
        expected = path.read_text(encoding="utf-8")
        if text != expected:
            diff = list(difflib.unified_diff(
                expected.splitlines(True), text.splitlines(True),
                f"{name}.json (golden)", f"{name}.json (this run)",
            ))
            shown = "".join(diff[:160])
            if len(diff) > 160:
                shown += f"... ({len(diff) - 160} more diff lines)\n"
            pytest.fail(f"golden snapshot {name}.json differs:\n{shown}")

    return check
