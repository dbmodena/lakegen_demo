"""Moved to ``lakegen.evaluation.reference_execution``; this name stays for the scripts that import it here."""

import sys

import lakegen.evaluation.reference_execution as _moved

sys.modules[__name__] = _moved
