"""Moved to ``lakegen.discovery.ledger``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.ledger as _moved

sys.modules[__name__] = _moved
