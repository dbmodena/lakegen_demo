"""Solr: the client, the indexers, and the portal catalogues behind the lakes' cores.

``client`` talks to a core; ``indexing`` and ``representation`` write the
embedding vectors and the text a table is indexed as; ``portal`` builds the
documents of the UK portal from CKAN metadata; ``catalogue`` and
``catalogue_text`` turn a catalogue document back into a local file and
the metadata the agents read.
"""
