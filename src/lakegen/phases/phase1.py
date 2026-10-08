"""Moved to ``lakegen.discovery.keywords`` (the divided architecture's keyword writer (phase 1)); this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.keywords as _moved

sys.modules[__name__] = _moved
