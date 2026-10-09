"""Every provider models.yaml names is registered; get_llm builds through the registry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lakegen.config.registry import ModelSpec, models
from lakegen.integrations.llm import providers
from lakegen.integrations.llm.providers import (
    LLMProvider,
    build_llm,
    provider_for,
    registered_providers,
)


def _spec(provider: str) -> ModelSpec:
    return ModelSpec(
        name="x", provider=provider, provider_model="x", context_window=None,
        max_output_tokens=10, temperature=0.1, timeout_seconds=None, groq_model=None,
    )


def test_every_configured_provider_is_registered():
    assert set(registered_providers()) == {"oci", "ollama", "together"}
    assert {model.provider for model in models()} <= set(registered_providers())
    for name in registered_providers():
        provider = provider_for(name)
        assert isinstance(provider, LLMProvider) and provider.name == name


def test_an_unregistered_provider_names_the_registered_ones():
    with pytest.raises(ValueError, match="unknown LLM provider 'nope'; registered: oci, ollama, together"):
        build_llm(_spec("nope"), SimpleNamespace(), None)


def test_build_llm_hands_the_spec_to_its_provider(monkeypatch):
    seen = {}

    class Fake:
        name = "fake"

        def build(self, spec, settings, callback_manager):
            seen.update(spec=spec, settings=settings, callback_manager=callback_manager)
            return "client"

    monkeypatch.setitem(providers._REGISTRY, "fake", Fake)
    settings = SimpleNamespace()
    assert build_llm(_spec("fake"), settings, "manager") == "client"
    assert seen == {"spec": _spec("fake"), "settings": settings, "callback_manager": "manager"}
