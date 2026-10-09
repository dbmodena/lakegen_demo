"""Moved to ``lakegen.discovery.tools.schema_matching``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.tools.schema_matching as _moved

sys.modules[__name__] = _moved
