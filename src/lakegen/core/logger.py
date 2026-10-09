"""Moved to ``lakegen.observability.csv_log``; this name stays for the scripts that import it here."""

import sys

import lakegen.observability.csv_log as _moved

sys.modules[__name__] = _moved
