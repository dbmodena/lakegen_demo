"""Moved to ``lakegen.observability.manifest``; this name stays for the scripts that import it here."""

import sys

import lakegen.observability.manifest as _moved

sys.modules[__name__] = _moved
