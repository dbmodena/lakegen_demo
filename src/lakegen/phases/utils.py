"""Moved to ``lakegen.discovery.candidates`` (the candidate helpers the discovery agents share); this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.candidates as _moved

sys.modules[__name__] = _moved
