"""LakeGen: answer a question over a data lake with generated code.

The package is organised by pipeline phase, with the shared layers below it
and the surfaces above it (an import may only go down this list; each folder
holds at most ten entries):

    interfaces      the API, the ``lakegen`` command, the Chainlit app, the dashboard
    experiments     specs, the run store, executors, services; ``analysis`` reads the runs
    workflow        pipeline (engine), coding, discovery, answer, evaluation,
                    plus observability (recorder, logs) and the prompt templates
    tools           the agents' tools: unified, coder, shared
    retrieval       the registry, the service and the modes (keyword, semantic, hybrid, ...)
    integrations    llm (chat, embeddings, rerank), solr (client, portal), pneuma, table_io
    config          settings, registries, ``ExperimentConfig``, paths
    domain          pure types, errors and result models
"""
