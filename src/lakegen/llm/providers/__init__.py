"""The LLM providers: a protocol, a registry, and the client each builds for a ``ModelSpec``.

An ``LLMProvider`` turns a model's spec (from ``config/models.yaml``) and the
environment's settings into a LlamaIndex client with LakeGen's token
accounting. ``build_llm(spec, settings, callback_manager)`` is the one place
a provider name turns into code; a spec naming an unregistered provider
fails with the registered ones listed.

To add a provider: a module here with a class holding ``name`` and
``build(spec, settings, callback_manager)``, ``@register("name")`` on it,
its import at the bottom of this file, and the models in ``models.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM

from lakegen.config.registry import ModelSpec
from lakegen.config.settings import Settings


@runtime_checkable
class LLMProvider(Protocol):
    """What a provider is: a name and the client it builds for a model's spec.

    Attributes:
        name: The provider's name in config/models.yaml.
    """

    name: str

    def build(self, spec: ModelSpec, settings: Settings, callback_manager: CallbackManager) -> LLM:
        """The client of ``spec``, with LakeGen's token accounting."""
        ...


Factory = Callable[[], LLMProvider]
_REGISTRY: dict[str, Factory] = {}


def register(name: str) -> Callable[[Factory], Factory]:
    """A class decorator that registers a provider under ``name``."""

    def mark(factory: Factory) -> Factory:
        _REGISTRY[name] = factory
        return factory

    return mark


def registered_providers() -> tuple[str, ...]:
    """The registered providers' names, in registration order."""
    return tuple(_REGISTRY)


def provider_for(name: str) -> LLMProvider:
    """The provider of ``name``; an unregistered name lists the registered ones."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown LLM provider {name!r}; registered: {known}") from None
    return factory()


def build_llm(spec: ModelSpec, settings: Settings, callback_manager: CallbackManager) -> LLM:
    """The client of ``spec`` from its provider."""
    return provider_for(spec.provider).build(spec, settings, callback_manager)


# Importing the providers registers them.
from lakegen.llm.providers import oci, oci_dedicated, ollama, together  # noqa: E402, F401
