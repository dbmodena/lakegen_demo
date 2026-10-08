"""Moved to ``lakegen.discovery.selectors.jev_service`` (the Jev service client); this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.selectors.jev_service as _moved

sys.modules[__name__] = _moved
