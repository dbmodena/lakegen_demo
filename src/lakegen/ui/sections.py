"""Moved to ``lakegen.interfaces.chainlit.sections``; this name stays for the scripts that import it here."""

import sys

import lakegen.interfaces.chainlit.sections as _moved

sys.modules[__name__] = _moved
