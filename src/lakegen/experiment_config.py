"""Moved to ``lakegen.config.experiment``; this name stays for the scripts that import it here."""

import sys

import lakegen.config.experiment as _moved

sys.modules[__name__] = _moved
