"""Moved to ``lakegen.pipeline.service``; this name stays for the scripts that import it here."""

import sys

import lakegen.pipeline.service as _moved

sys.modules[__name__] = _moved
