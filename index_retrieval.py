"""The indexing command moved to ``lakegen.interfaces.cli.index`` (``lakegen index``); this entry keeps ``python index_retrieval.py`` working."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lakegen.interfaces.cli.index import (  # noqa: E402
    load_source_documents,
    main,
    validate_source_coverage,
)

__all__ = ["load_source_documents", "main", "validate_source_coverage"]

if __name__ == "__main__":
    main()
