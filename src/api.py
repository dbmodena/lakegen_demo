"""The HTTP API moved to ``lakegen.interfaces.api.app``; this entry keeps ``uvicorn src.api:app`` working.

Scripts that ``import api`` or ``import src.api`` get the one real module,
so what they read or replace on it is shared with the running app.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
for _path in (_SRC_DIR, _SRC_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import lakegen.interfaces.api.app as _moved  # noqa: E402

sys.modules[__name__] = _moved
