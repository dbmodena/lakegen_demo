"""Moved to ``lakegen.coding.attempts``; this name stays for the scripts that import it here."""

import sys

import lakegen.coding.attempts as _moved

sys.modules[__name__] = _moved
