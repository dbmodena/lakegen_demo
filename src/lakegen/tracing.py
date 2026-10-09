"""Moved to ``lakegen.observability.tracing``; this name stays for the scripts that import it here."""

import sys

import lakegen.observability.tracing as _moved

sys.modules[__name__] = _moved
