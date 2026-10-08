"""The dependency rules of the refactor, checked on a fresh interpreter.

A fresh process is needed because the test session has already imported
chainlit for the UI tests; the rule is about what the batch, API and CLI
surfaces pull in on their own.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _modules_loaded_by(*imports: str) -> set[str]:
    code = (
        "import sys\n"
        + "".join(f"import {name}\n" for name in imports)
        + "print('\\n'.join(sorted(sys.modules)))\n"
    )
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT, env=env,
    )
    return set(result.stdout.split())


def test_the_service_api_and_cli_do_not_load_chainlit():
    loaded = _modules_loaded_by("lakegen.service", "api", "cli")
    assert "lakegen.service" in loaded and "api" in loaded and "cli" in loaded
    assert not {name for name in loaded if name == "chainlit" or name.startswith("chainlit.")}
    assert not {name for name in loaded if name.startswith("lakegen.ui")}
