"""Moved to ``lakegen.discovery.tools.join_keys``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.tools.join_keys as _moved

sys.modules[__name__] = _moved
