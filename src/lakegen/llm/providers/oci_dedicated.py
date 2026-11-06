"""An OCI Dedicated AI Cluster endpoint through its OpenAI-compatible API: the Together client pointed at it.

A dedicated endpoint serves one deployment of open weights (gpt-oss-20b on endpoints 1 and 2, Qwen3.8 27B
on endpoint 3) on the OpenAI chat route,
with an API key as bearer token. Its model id is the endpoint's OCID, which belongs to one deployment,
so it lives in .env rather than in config/models.yaml: each ``dedicated.*`` entry names the variable
that holds it (``model_env``), and the key, base URL, extra headers and extra body are the
DEDICATE_OCI_* settings (``config/settings.py``). The client is ``TogetherLLM``: retries on 429/5xx,
connection errors and replies the transport cut, the reasoning sent back with the turn, one tool call
per turn, tool calls written as text recovered, and ``tool_choice="required"`` honoured when the
selector asks a turn again.
"""

from __future__ import annotations

from llama_index.core.callbacks import CallbackManager

from lakegen.config.registry import ModelSpec
from lakegen.config.settings import Settings, dedicated_model
from lakegen.llm.providers import register
from lakegen.llm.providers.together import TogetherLLM


@register("oci_dedicated")
class OCIDedicatedProvider:
    """The ``oci_dedicated`` provider: an OCI Dedicated AI Cluster endpoint."""

    name = "oci_dedicated"

    def build(
        self, spec: ModelSpec, settings: Settings, callback_manager: CallbackManager
    ) -> TogetherLLM:
        """The ``TogetherLLM`` client of a dedicated endpoint.

        Args:
            spec: The model's spec; ``options["model_env"]`` names the variable holding
                the endpoint's model id.
            settings: The DEDICATE_OCI_* key, base URL, headers and extra body.
            callback_manager: LlamaIndex's callback manager.

        Returns:
            The client.

        Raises:
            RuntimeError: The spec names no ``model_env``, or the model id, key or base
                URL is not set.
        """
        variable = spec.options.get("model_env")
        if not variable:
            raise RuntimeError(
                f"{spec.name} needs model_env in config/models.yaml: the .env variable holding the endpoint's model id"
            )
        needed = {
            variable: dedicated_model(variable),
            "DEDICATE_OCI_API_KEY": settings.dedicated_api_key,
            "DEDICATE_OCI_BASE_URL": settings.dedicated_base_url,
        }
        missing = [name for name, value in needed.items() if not value]
        if missing:
            raise RuntimeError(
                f"{spec.name} needs its dedicated endpoint in .env: set {', '.join(missing)}"
            )
        return TogetherLLM(
            model=needed[variable],
            api_base=settings.dedicated_base_url,
            api_key=settings.dedicated_api_key,
            context_window=spec.context_window,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=spec.temperature,
            max_tokens=spec.max_output_tokens,
            timeout=spec.timeout_seconds,
            max_retries=0,  # TogetherLLM retries with its own backoff
            default_headers=settings.dedicated_headers or None,
            # Streams end with a usage chunk, which the token accounting reads; the extra body
            # (a reasoning_effort, say) rides on every request.
            additional_kwargs={
                "stream_options": {"include_usage": True},
                **settings.dedicated_extra_body,
            },
            callback_manager=callback_manager,
        )
