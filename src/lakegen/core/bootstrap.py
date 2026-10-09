"""Moved to ``lakegen.config.bootstrap``; this name stays for the scripts that import it here."""

import sys

import lakegen.config.bootstrap as _moved

sys.modules[__name__] = _moved
