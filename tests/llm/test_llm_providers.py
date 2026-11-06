"""Every provider models.yaml names is registered; get_llm builds through the registry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lakegen.config.registry import ModelSpec, models
from lakegen.llm import providers
from lakegen.llm.providers import (
    LLMProvider,
    build_llm,
    provider_for,
    registered_providers,
)
from lakegen.llm.providers.together import TogetherLLM


def _spec(provider: str, **options) -> ModelSpec:
    return ModelSpec(
        name="x",
        provider=provider,
        provider_model="x",
        context_window=None,
        max_output_tokens=10,
        temperature=0.1,
        timeout_seconds=None,
        groq_model=None,
        options=options,
    )


def test_every_configured_provider_is_registered():
    assert set(registered_providers()) == {"oci", "oci_dedicated", "ollama", "together"}
    assert {model.provider for model in models()} <= set(registered_providers())
    for name in registered_providers():
        provider = provider_for(name)
        assert isinstance(provider, LLMProvider) and provider.name == name


def test_an_unregistered_provider_names_the_registered_ones():
    with pytest.raises(
        ValueError,
        match="unknown LLM provider 'nope'; registered: oci, oci_dedicated, ollama, together",
    ):
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


def _dedicated_settings(**values) -> SimpleNamespace:
    return SimpleNamespace(
        **{
            "dedicated_api_key": "",
            "dedicated_base_url": "",
            "dedicated_headers": {},
            "dedicated_extra_body": {},
            **values,
        }
    )


def test_the_dedicated_provider_points_the_together_client_at_the_endpoint(monkeypatch):
    monkeypatch.setenv("DEDICATE_OCI_MODEL_2", " ocid1.endpoint.two ")
    settings = _dedicated_settings(
        dedicated_api_key="secret",
        dedicated_base_url="https://host/20231130/actions/v1",
        dedicated_headers={"x-compartment": "c"},
        dedicated_extra_body={"reasoning_effort": "low"},
    )
    spec = ModelSpec(
        name="dedicated.x",
        provider="oci_dedicated",
        provider_model="dedicated.x",
        context_window=128_000,
        max_output_tokens=16_000,
        temperature=0.1,
        timeout_seconds=300,
        groq_model=None,
        options={"model_env": "DEDICATE_OCI_MODEL_2"},
    )
    llm = build_llm(spec, settings, None)
    assert isinstance(llm, TogetherLLM) and llm.tool_choice_supported
    assert (llm.model, llm.api_base, llm.api_key) == (
        "ocid1.endpoint.two",
        "https://host/20231130/actions/v1",
        "secret",
    )
    assert llm.default_headers == {"x-compartment": "c"}
    assert llm.additional_kwargs == {
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
    }
    assert (llm.context_window, llm.max_tokens, llm.temperature, llm.timeout, llm.max_retries) == (
        128_000,
        16_000,
        0.1,
        300,
        0,
    )


def test_the_dedicated_provider_names_what_the_environment_lacks(monkeypatch):
    monkeypatch.delenv("DEDICATE_OCI_MODEL", raising=False)
    with pytest.raises(
        RuntimeError, match="set DEDICATE_OCI_MODEL, DEDICATE_OCI_API_KEY, DEDICATE_OCI_BASE_URL"
    ):
        build_llm(
            _spec("oci_dedicated", model_env="DEDICATE_OCI_MODEL"), _dedicated_settings(), None
        )
    monkeypatch.setenv("DEDICATE_OCI_MODEL", "ocid1.endpoint")
    with pytest.raises(RuntimeError, match="set DEDICATE_OCI_API_KEY, DEDICATE_OCI_BASE_URL"):
        build_llm(
            _spec("oci_dedicated", model_env="DEDICATE_OCI_MODEL"), _dedicated_settings(), None
        )
    with pytest.raises(RuntimeError, match="needs model_env in config/models.yaml"):
        build_llm(
            _spec("oci_dedicated"),
            _dedicated_settings(dedicated_api_key="k", dedicated_base_url="u"),
            None,
        )
