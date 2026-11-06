"""Solr: the client of the lakes' cores and the catalogue entries read from them.

``client`` queries a core (lexical, KNN and cursor queries); ``catalogue``
turns a catalogue document into a local file and the metadata the agents
read. The cores are built by ``solr_builder`` at the repository root.
"""
