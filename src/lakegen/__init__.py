"""LakeGen: answer a question over a data lake with generated code.

The package is organised by pipeline phase, with the shared layers below it
and the surfaces above it (an import may only go down this list):

    interfaces      the API, the ``lakegen`` command, Chainlit, the dashboard
    experiments     specs, the run store, executors, services
    pipeline        the engine that runs one question; the service and runner
    coding          phase 3: the coder agent, its tools, context and sweep
    discovery       phases 1-2: keywords, table selection, the unified agent
    answer          phase 4: the written answer
    evaluation      judging a code result against its reference
    retrieval       the retrieval modes, Pneuma, the cell scan
    llm             the chat models and the agent runner
    lakes           a lake's tables and catalogue
    config          settings, registries, ``ExperimentConfig``, paths
    domain          pure types, errors and result models

``observability`` (the recorder, logs, manifest) is used by the pipeline and
what runs it; ``analysis`` reads the runs; ``prompts`` holds the templates.
The old module paths (``core``, ``agent_tools``, ``ui``, ``phases`` and the
loose top-level modules) are aliases of the new homes.
"""
