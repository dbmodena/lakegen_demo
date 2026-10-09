"""Moved to ``lakegen.config.locations``; this name stays for the scripts that import it here."""

import sys

import lakegen.config.locations as _moved

sys.modules[__name__] = _moved
