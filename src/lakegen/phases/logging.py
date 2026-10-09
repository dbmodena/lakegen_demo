"""Moved to ``lakegen.llm.agents.activity`` (the agents' activity formatting and stall detection); this name stays for the scripts that import it here."""

import sys

import lakegen.llm.agents.activity as _moved

sys.modules[__name__] = _moved
