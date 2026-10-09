"""Moved to ``lakegen.interfaces.chainlit.streaming``; this name stays for the scripts that import it here."""

import sys

import lakegen.interfaces.chainlit.streaming as _moved

sys.modules[__name__] = _moved
