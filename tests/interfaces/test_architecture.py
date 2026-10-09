"""What the packages may import: the frameworks stay in the interfaces, the layers point down."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_batch_service_the_api_and_the_cli_never_load_chainlit(tmp_path):
    script = (
        "import sys; import lakegen.workflow.pipeline.service, lakegen.interfaces.api.app, lakegen.interfaces.cli.ask; "
        "assert 'chainlit' not in sys.modules, 'chainlit was imported'; print('ok')"
    )
    run = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "", "LAKEGEN_STATE_DIR": str(tmp_path)},
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "ok"
