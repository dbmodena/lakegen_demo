"""The chat models: a client per provider, the tool-call fixes they need, and ``get_llm``.

``factory.get_llm(model)`` looks the model up in ``config/models.yaml`` and
builds its provider's client (``providers.oci``, ``providers.ollama``,
``providers.together``), each with LakeGen's token accounting.
``tool_calls`` holds what the providers' tool calls need before an agent can
run them.
"""

from lakegen.llm.factory import get_llm, spec_for

__all__ = ["get_llm", "spec_for"]
