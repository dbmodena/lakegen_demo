"""Moved to ``lakegen.discovery.tools.context_editor``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.tools.context_editor as _moved

sys.modules[__name__] = _moved
