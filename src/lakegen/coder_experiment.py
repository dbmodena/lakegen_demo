"""Moved to ``lakegen.coding.sweep``; this name stays for the scripts that import it here."""

import sys

import lakegen.coding.sweep as _moved

sys.modules[__name__] = _moved
