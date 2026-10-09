"""Moved to ``lakegen.pipeline.runner``; this name stays for the scripts that import it here."""

import sys

import lakegen.pipeline.runner as _moved

sys.modules[__name__] = _moved
