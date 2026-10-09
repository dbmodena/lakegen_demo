"""Moved to ``lakegen.coding.column_resolution``; this name stays for the scripts that import it here."""

import sys

import lakegen.coding.column_resolution as _moved

sys.modules[__name__] = _moved
