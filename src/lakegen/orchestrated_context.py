"""Moved to ``lakegen.discovery.context``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.context as _moved

sys.modules[__name__] = _moved
