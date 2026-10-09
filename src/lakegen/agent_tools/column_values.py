"""Moved to ``lakegen.discovery.tools.column_values``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.tools.column_values as _moved

sys.modules[__name__] = _moved
