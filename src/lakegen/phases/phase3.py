"""Moved to ``lakegen.coding.agent`` (the coder agent (phase 3)); this name stays for the scripts that import it here."""

import sys

import lakegen.coding.agent as _moved

sys.modules[__name__] = _moved
