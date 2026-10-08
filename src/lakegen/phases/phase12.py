"""Moved to ``lakegen.discovery.unified`` (the unified agent that searches, inspects and selects (phase 1+2)); this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.unified as _moved

sys.modules[__name__] = _moved
