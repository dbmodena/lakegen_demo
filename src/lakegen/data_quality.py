"""Moved to ``lakegen.coding.data_quality``; this name stays for the scripts that import it here."""

import sys

import lakegen.coding.data_quality as _moved

sys.modules[__name__] = _moved
