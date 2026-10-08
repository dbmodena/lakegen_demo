"""Compatibility alias: the Solr client lives in ``lakegen.retrieval.solr_client``.

Both names refer to one module object, so patching an attribute through either
name patches the real module. Remove once nothing imports ``src.client_solr``.
"""

import sys

from lakegen.retrieval import solr_client as _solr_client

sys.modules[__name__] = _solr_client
