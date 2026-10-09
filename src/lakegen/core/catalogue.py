"""Moved to ``lakegen.lakes.catalogue_text``; this name stays for the scripts that import it here."""

import sys

import lakegen.lakes.catalogue_text as _moved

sys.modules[__name__] = _moved
