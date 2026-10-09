"""Moved to ``lakegen.coding.context``; this name stays for the scripts that import it here."""

import sys

import lakegen.coding.context as _moved

sys.modules[__name__] = _moved
