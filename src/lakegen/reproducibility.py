"""Moved to ``lakegen.observability.reproducibility``; this name stays for the scripts that import it here."""

import sys

import lakegen.observability.reproducibility as _moved

sys.modules[__name__] = _moved
