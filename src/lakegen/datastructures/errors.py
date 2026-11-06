"""Exceptions the orchestration raises across layers.

They live here, chainlit-free, because the discovery phases raise them and
the service imports the phases; the UI re-exports them from ``ui.state``.
"""


class WorkflowCancelled(Exception):
    """Raised when the user clicks Stop in the UI."""


class WorkflowTimedOut(Exception):
    """Raised when a Chainlit interaction expires without an answer."""
