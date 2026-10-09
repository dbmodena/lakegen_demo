"""Moved to ``lakegen.config.runtime``; this name stays for the scripts that import it here."""

import sys

import lakegen.config.runtime as _moved

sys.modules[__name__] = _moved
