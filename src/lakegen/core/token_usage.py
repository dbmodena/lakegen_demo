"""Moved to ``lakegen.llm.token_usage``; this name stays for the scripts that import it here."""

import sys

from lakegen.llm import token_usage as _moved

sys.modules[__name__] = _moved
