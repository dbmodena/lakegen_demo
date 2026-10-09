"""Moved to ``lakegen.evaluation.output_validation``; this name stays for the scripts that import it here."""

import sys

import lakegen.evaluation.output_validation as _moved

sys.modules[__name__] = _moved
