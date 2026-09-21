"""HTTP API that runs the isolated ``agent_tools_2`` / ``core_2`` variant.

Start with ``uv run uvicorn src.api_2:app --host 127.0.0.1 --port 8012``.
The regular ``src.api:app`` remains the baseline API.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _path in (_SRC_DIR, _ROOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lakegen.agent_tools_2 import tools_p12 as tools_p12_v2
from lakegen.agent_tools_2 import tools_p2 as tools_p2_v2


# Register the candidate modules before importing the API, service, and phases.
sys.modules["lakegen.agent_tools.tools_p2"] = tools_p2_v2
sys.modules["lakegen.agent_tools.tools_p12"] = tools_p12_v2

from src.api import app  # noqa: E402 - aliases must be registered first


app.title = "LakeGen API — agent tools v2"
app.description += " This instance runs the isolated agent_tools_2/core_2 candidate."
