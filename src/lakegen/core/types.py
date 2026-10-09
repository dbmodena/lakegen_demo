"""Moved to ``lakegen.domain.types``; this name stays for the scripts that import it here."""

import sys

import lakegen.domain.types as _moved

sys.modules[__name__] = _moved
