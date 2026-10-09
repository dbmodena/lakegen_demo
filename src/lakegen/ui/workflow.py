"""Moved to ``lakegen.interfaces.chainlit.workflow``; this name stays for the scripts that import it here."""

import sys

import lakegen.interfaces.chainlit.workflow as _moved

sys.modules[__name__] = _moved
