"""Moved to ``lakegen.interfaces.chainlit.starters``; this name stays for the scripts that import it here."""

import sys

import lakegen.interfaces.chainlit.starters as _moved

sys.modules[__name__] = _moved
