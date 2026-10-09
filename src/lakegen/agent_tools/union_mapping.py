"""Moved to ``lakegen.discovery.tools.union_mapping``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.tools.union_mapping as _moved

sys.modules[__name__] = _moved
