"""Moved to ``lakegen.llm.agents.instrumentation``; this name stays for the scripts that import it here."""

import sys

import lakegen.llm.agents.instrumentation as _moved

sys.modules[__name__] = _moved
