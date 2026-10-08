"""The orchestration of one question: discovery, coding, answer.

``engine`` runs the pipeline for a ``RunContext`` (the settings of the run)
with ``PipelineDeps`` (the phases and resources it calls, injectable). The
service, the CLI and the Chainlit app are drivers of this one engine.
"""
