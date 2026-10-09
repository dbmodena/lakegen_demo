"""Moved to ``lakegen.domain.errors``; this name stays for the scripts that import it here."""

import sys

import lakegen.domain.errors as _moved

sys.modules[__name__] = _moved
