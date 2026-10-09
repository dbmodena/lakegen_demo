"""Moved to ``lakegen.lakes.table_io``; this name stays for the scripts that import it here."""

import sys

import lakegen.lakes.table_io as _moved

sys.modules[__name__] = _moved
