"""A local Ollama server through its OpenAI-compatible endpoint, with LakeGen's token accounting."""

from __future__ import annotations

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai_like import OpenAILike

from lakegen.config.registry import ModelSpec, get_model
from lakegen.config.settings import Settings
from lakegen.llm.providers import register
from lakegen.llm.token_usage import estimate_tokens, extract_total_tokens

OLLAMA_CONTEXT_SIZE = get_model("ollama.gpt-oss-20b").context_window


class OllamaLLM(OpenAILike):
    """A local Ollama model with the token accounting LakeGen reads from OCI."""

    _token_usage_total: int = PrivateAttr(default=0)

    @property
    def token_usage_total(self) -> int:
        """Tokens spent since the last reset: reported by the server, else estimated."""
        return self._token_usage_total

    def reset_token_usage(self) -> None:
        """Start counting tokens from zero."""
        self._token_usage_total = 0

    def _record_response(self, messages, kwargs, response) -> None:
        total = extract_total_tokens(getattr(response, "raw", None))
        self._token_usage_total += total or estimate_tokens(
            messages, kwargs, getattr(response.message, "content", "") or ""
        )

    def chat(self, messages, **kwargs):
        """A chat reply, its tokens counted; the ``think`` option is dropped."""
        # See OCIGenAILLM.stream_chat: the OpenAI client rejects ``think``.
        kwargs.pop("think", None)
        response = super().chat(messages, **kwargs)
        self._record_response(messages, kwargs, response)
        return response

    async def achat(self, messages, **kwargs):
        """The async ``chat``."""
        kwargs.pop("think", None)
        response = await super().achat(messages, **kwargs)
        self._record_response(messages, kwargs, response)
        return response

    def stream_chat(self, messages, **kwargs):
        """A streamed chat reply, its tokens counted from the last chunk."""
        kwargs.pop("think", None)
        stream = super().stream_chat(messages, **kwargs)

        def generate():
            last = None
            for chunk in stream:
                last = chunk
                yield chunk
            if last is not None:
                self._record_response(messages, kwargs, last)

        return generate()

    async def astream_chat(self, messages, **kwargs):
        """The async ``stream_chat``."""
        kwargs.pop("think", None)
        stream = await super().astream_chat(messages, **kwargs)

        async def generate():
            last = None
            async for chunk in stream:
                last = chunk
                yield chunk
            if last is not None:
                self._record_response(messages, kwargs, last)

        return generate()


@register("ollama")
class OllamaProvider:
    """The ``ollama`` provider: a local Ollama server (LAKEGEN_OLLAMA_BASE_URL)."""

    name = "ollama"

    def build(
        self, spec: ModelSpec, settings: Settings, callback_manager: CallbackManager
    ) -> OllamaLLM:
        """The Ollama client of ``spec``, on the server's OpenAI-compatible /v1 route."""
        base_url = settings.ollama_base_url
        return OllamaLLM(
            model=spec.provider_model,
            api_base=f"{base_url.rstrip('/')}/v1",
            api_key="ollama",
            context_window=spec.context_window,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=spec.temperature,
            max_tokens=spec.max_output_tokens,
            timeout=spec.timeout_seconds,
            callback_manager=callback_manager,
        )
