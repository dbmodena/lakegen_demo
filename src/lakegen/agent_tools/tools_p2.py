"""Moved to ``lakegen.discovery.tools.judge``; this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.tools.judge as _moved

sys.modules[__name__] = _moved
