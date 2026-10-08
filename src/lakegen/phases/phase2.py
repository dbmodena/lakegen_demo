"""Moved to ``lakegen.discovery.divided`` (the divided architecture's judge over the retrieved candidates (phase 2)); this name stays for the scripts that import it here."""

import sys

import lakegen.discovery.divided as _moved

sys.modules[__name__] = _moved
