"""The step-by-step CLI moved to ``lakegen.interfaces.cli.ask`` (``lakegen ask``); this entry keeps ``python src/cli.py`` working.

Tests and scripts that ``import src.cli`` get the one real module.
"""

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
for _path in (_SRC_DIR, _SRC_DIR.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import lakegen.interfaces.cli.ask as _moved  # noqa: E402

sys.modules[__name__] = _moved

if __name__ == "__main__":
    _moved.main()
