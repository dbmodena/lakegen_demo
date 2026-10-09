"""Moved to ``lakegen.interfaces.chainlit.state``; this name stays for the scripts that import it here."""

import sys

import lakegen.interfaces.chainlit.state as _moved

sys.modules[__name__] = _moved
