"""Moved to ``lakegen.answer.synthesize`` (the answer written from the result (phase 4)); this name stays for the scripts that import it here."""

import sys

import lakegen.answer.synthesize as _moved

sys.modules[__name__] = _moved
