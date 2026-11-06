"""The Pneuma retrieval engine: the index builder and the HTTP query service.

It runs in its own environment, ``.venv-pneuma`` (``pneuma_engine/install.sh``):
Pneuma 0.0.4's Chroma stack needs NumPy < 2, while LakeGen needs NumPy >= 2.
LakeGen never imports it; it queries the service over HTTP
(``lakegen.integrations.pneuma.client``).

Modules:
    server: the HTTP service answering ``/query`` and ``/health``.
    bootstrap: builds a resumable index from a lake's tables and metadata.
    indexing: patches to Pneuma's indexing (DuckDB memory, embeddings, Chroma batches).
    metadata: the catalogue text registered as each table's context.
    judge: the schema-constrained LLM relevance judge.
    narration: concurrent column narration and capped row samples.
    oci_client: an OpenAI-shaped client backed by OCI Generative AI.
"""
