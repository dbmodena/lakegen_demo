"""Moved to ``lakegen.config.retrieval``; this name stays for the scripts that import it here."""

import sys

import lakegen.config.retrieval as _moved

sys.modules[__name__] = _moved
