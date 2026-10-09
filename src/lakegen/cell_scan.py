"""Moved to ``lakegen.retrieval.cell_scan``; this name stays for the scripts that import it here."""

import sys

import lakegen.retrieval.cell_scan as _moved

sys.modules[__name__] = _moved
