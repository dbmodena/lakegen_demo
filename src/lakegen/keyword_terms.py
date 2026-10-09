"""Moved to ``lakegen.discovery.terms``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.terms as _moved

sys.modules[__name__] = _moved
