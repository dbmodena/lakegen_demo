"""LakeGen: answer a question over a data lake with generated code.

The package is organised in seven areas; an import may only go down this list:

    entrypoints     the API (one question), the ``lakegen`` command, the web app
    pipeline        one question through discovery, coding, evaluation and the answer:
                    the engine and its stages, the phases, the evidence they share,
                    what a run records
    retrieval       the service and the modes that rank the lake's tables, over Solr
                    (client, catalogue) and Pneuma
    llm             the chat models, embeddings, reranker, agent runner and prompts
    config          settings, the lake and model registries, ``ExperimentConfig``, paths
    datastructures  the records the areas exchange: results, outcomes, hits, errors
    utils           general-purpose helpers: table files, naming, table ids, merge limits

The prompts (``prompts/``) and the configuration (``config/``) are files at the
repository root. What is built around LakeGen lives beside ``src/``, never
imported by it: ``pneuma_engine`` (the Pneuma service), ``solr_builder`` (the
Solr cores) and ``experiment_runner`` (specs, batches, the run store and their
analysis).
"""
