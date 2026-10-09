"""Moved to ``lakegen.domain.results``; this name stays for the scripts that import it here."""

import sys

import lakegen.domain.results as _moved

sys.modules[__name__] = _moved
