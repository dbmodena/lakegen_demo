"""Moved to ``lakegen.coding.revision_policy``; this name stays for the scripts that import it here."""

import sys

import lakegen.coding.revision_policy as _moved

sys.modules[__name__] = _moved
