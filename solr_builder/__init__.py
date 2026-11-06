"""The Solr builder: everything that constructs the lakes' Solr cores, offline.

LakeGen only queries the cores (``lakegen.retrieval.solr``); it never
imports this package. Run its commands from the repository root with LakeGen's
environment, e.g. ``.venv/bin/python -m solr_builder.index --help``.

Commands:
    build_portal_core: a new portal core from a CKAN/ODS crawl.
    build_uk_fixed: the ``uk_fixed`` core rebuilt from the cleaned UK catalogue.
    index: the table embeddings of a core, dry-run first.

Modules:
    portal: the UK documents built from CKAN metadata and parquet schemas.
    representation: the versioned text a table is embedded as.
    indexing: the embedding indexer and the vector-schema checks.
"""
