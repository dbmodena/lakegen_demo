"""Adapters to the systems outside LakeGen: the language models, Solr and its portal catalogues, Pneuma, the tables on disk.

``llm`` has the chat, embedding and rerank clients and the agent runner,
``solr`` the Solr client, the indexers and the portal documents, ``pneuma``
the Pneuma service client and the Seeker's content scan, ``table_io`` the
readers of the lake's CSV and Parquet files. Nothing here knows the
pipeline: the retrieval modes and the phases build on these.
"""
