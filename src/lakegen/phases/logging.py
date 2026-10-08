"""Moved to ``lakegen.agents.activity`` (the agents' activity formatting and stall detection); this name stays for the scripts that import it here."""

import sys

import lakegen.agents.activity as _moved

sys.modules[__name__] = _moved
