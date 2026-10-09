"""Moved to ``lakegen.evaluation.semantic_judge``; this name stays for the scripts that import it here."""

import sys

import lakegen.evaluation.semantic_judge as _moved

sys.modules[__name__] = _moved
