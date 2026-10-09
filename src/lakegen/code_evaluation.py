"""Moved to ``lakegen.evaluation.code``; this name stays for the scripts that import it here."""

import sys

import lakegen.evaluation.code as _moved

sys.modules[__name__] = _moved
