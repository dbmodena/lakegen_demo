"""The chat model of a run: its spec from the registry, its client from the provider."""

from __future__ import annotations

import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import LLM

from lakegen.config.registry import ModelSpec, get_model
from lakegen.config.settings import get_settings
from lakegen.llm.providers import oci, ollama, together


def spec_for(model: str) -> ModelSpec:
    """The registered model, or an OCI model id the registry does not list.

    An unlisted id goes to OCI with the limits it always had: 4,000 output
    tokens and no context size.
    """
    try:
        return get_model(model)
    except KeyError:
        return ModelSpec(
            name=model, provider="oci", provider_model=model, context_window=None,
            max_output_tokens=4000, temperature=0.1, timeout_seconds=None, groq_model=None,
        )


def get_llm(model: str) -> tuple[LLM, TokenCountingHandler]:
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])
    env = get_settings()
    spec = spec_for(model)
    if spec.provider == "ollama":
        llm = ollama.build(spec, env, Settings.callback_manager)
    elif spec.provider == "together":
        llm = together.build(spec, env, Settings.callback_manager)
    else:
        llm = oci.build(spec, env, Settings.callback_manager)
    return llm, token_counter
