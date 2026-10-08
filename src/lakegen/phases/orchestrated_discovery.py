"""Moved to ``lakegen.discovery.orchestrated`` (the orchestrated discovery paths); this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.orchestrated as _moved

sys.modules[__name__] = _moved
