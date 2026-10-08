"""Moved to ``lakegen.llm.groq_offload``; this name stays for the scripts that import it here."""

import sys

from lakegen.llm import groq_offload as _moved

sys.modules[__name__] = _moved
