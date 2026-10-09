"""Moved to ``lakegen.discovery.memory``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.memory as _moved

sys.modules[__name__] = _moved
