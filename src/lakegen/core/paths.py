"""Moved to ``lakegen.config.paths``; this name stays for the scripts that import it here."""

import sys

import lakegen.config.paths as _moved

sys.modules[__name__] = _moved
