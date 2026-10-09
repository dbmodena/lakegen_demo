"""Moved to ``lakegen.llm.agents.runner``; this name stays for the scripts that import it here."""

import sys

import lakegen.llm.agents.runner as _moved

sys.modules[__name__] = _moved
