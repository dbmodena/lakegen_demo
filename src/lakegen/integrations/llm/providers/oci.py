"""OCI Generative AI: the client with LakeGen's tool-call and token fixes, and the account's runtime config."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import oci
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.llms.oci_genai import OCIGenAI
from llama_index.llms.oci_genai.utils import CHAT_MODELS, PROVIDERS, XAIProvider

from lakegen.config.registry import ModelSpec, get_model, models_by_provider
from lakegen.config.settings import Settings, get_settings
from lakegen.integrations.llm import groq_offload
from lakegen.integrations.llm.providers import register
from lakegen.integrations.llm.token_usage import estimate_tokens, extract_total_tokens
from lakegen.integrations.llm.tool_calls import (
    STREAM_END,
    finalized_stream_tool_calls,
    merge_generic_stream_tool_calls,
    next_stream_item,
    parse_tool_call_input,
)

OPENAI_GPT_OSS_MODEL = "openai.gpt-oss-120b"
# The models, their providers and their limits come from config/models.yaml
# (lakegen.config.registry); these names serve the generic chat format below.
GENERIC_CHAT_MODELS = set(models_by_provider("oci"))
OPENAI_GPT_OSS_CONTEXT_SIZE = get_model(OPENAI_GPT_OSS_MODEL).context_window


class OCIGenericProvider(XAIProvider):
    """Use OCI's generic chat format for GPT-OSS and Meta models.

    LlamaIndex 0.7.0 implements the generic OCI request/response format in its
    XAI provider, but does not register OpenAI and registers Meta without tool
    support. OCI's generic endpoint supports tool calls for both models.
    """

    def chat_generation_info(self, response: Any) -> dict[str, Any]:
        chat_response = response.data.chat_response
        info: dict[str, Any] = {
            "finish_reason": chat_response.choices[0].finish_reason,
        }
        usage_total = extract_total_tokens(getattr(chat_response, "usage", None))
        if usage_total:
            info["usage"] = {"total_tokens": usage_total}
        assistant_message = chat_response.choices[0].message
        tool_calls = getattr(assistant_message, "tool_calls", None) or []
        formatted_tool_calls = []
        for tool_call in tool_calls:
            name = getattr(tool_call, "name", None)
            if name is None:
                continue
            arguments = getattr(tool_call, "arguments", None) or getattr(
                tool_call, "parameters", None
            )
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            formatted_tool_calls.append(
                {
                    "toolUseId": getattr(tool_call, "id", None) or uuid.uuid4().hex,
                    "name": name,
                    "input": arguments,
                }
            )
        if formatted_tool_calls:
            info["tool_calls"] = formatted_tool_calls
        return info


class OCIGenAILLM(OCIGenAI):
    """Compatibility fixes for OCI tool calls and async LlamaIndex agents."""

    _token_usage_total: int = PrivateAttr(default=0)

    @property
    def token_usage_total(self) -> int:
        return self._token_usage_total

    def reset_token_usage(self) -> None:
        self._token_usage_total = 0

    def _record_token_usage(self, value: Any) -> int:
        total = extract_total_tokens(value)
        if total:
            self._token_usage_total += total
        return total

    @property
    def metadata(self):
        return super().metadata.model_copy(
            update={"is_function_calling_model": True}
        )

    def _prepare_chat_with_tools(
        self,
        tools: Sequence[Any],
        user_msg: str | ChatMessage | None = None,
        chat_history: list[ChatMessage] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(user_msg, str):
            user_msg = ChatMessage(role=MessageRole.USER, content=user_msg)

        messages = list(chat_history or [])
        if user_msg is not None:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tools,
            **({"tool_choice": "REQUIRED"} if tool_required else {}),
            **kwargs,
        }

    async def achat(self, messages, **kwargs):
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    def get_tool_calls_from_response(
        self,
        response,
        error_on_no_tool_call: bool = True,
        **kwargs,
    ) -> list[ToolSelection]:
        selections = []
        for tool_call in response.message.additional_kwargs.get("tool_calls", []):
            name = tool_call.get("name")
            arguments = parse_tool_call_input(tool_call.get("input"))
            if not name or arguments is None:
                continue
            selections.append(
                ToolSelection(
                    tool_id=tool_call.get("toolUseId") or uuid.uuid4().hex,
                    tool_name=name,
                    tool_kwargs=arguments,
                )
            )

        if not selections and error_on_no_tool_call:
            raise ValueError("Expected at least one complete tool call, but got none.")
        return selections

    def chat(self, messages, **kwargs):
        # Plain calls may go to Groq's free tier instead (LAKEGEN_GROQ_OFFLOAD).
        offloaded = groq_offload.chat(self.model, messages, kwargs, self.temperature)
        if offloaded is not None:
            self._record_token_usage(offloaded.raw)
            return offloaded
        response = super().chat(messages, **kwargs)
        total = self._record_token_usage(response.raw)
        if not total:
            total = self._record_token_usage(response.message.additional_kwargs)
        if not total:
            self._token_usage_total += estimate_tokens(
                messages, kwargs, getattr(response.message, "content", "")
            )
        return response

    def stream_chat(self, messages, **kwargs):
        # ``think`` is an Ollama-specific request option.  Phase 1 and Phase 3
        # use it to request a separate reasoning stream when the backend
        # supports that feature, but OCI rejects unknown request parameters.
        # OCI may still return reasoning fields on its own, so only remove the
        # unsupported request option and continue processing the response.
        kwargs.pop("think", None)
        stream = super().stream_chat(messages, **kwargs)
        def generate():
            accumulated_tool_calls: list[dict[str, Any]] = []
            recorded_for_request = 0
            response_parts: list[str] = []
            for chunk in stream:
                response_parts.append(str(getattr(chunk, "delta", "") or ""))
                raw = chunk.raw if isinstance(chunk.raw, dict) else {}
                raw_data = raw.get("data", "")
                event_data = {}
                if raw_data:
                    try:
                        event_data = json.loads(raw_data)
                        if self.model in GENERIC_CHAT_MODELS:
                            merge_generic_stream_tool_calls(
                                accumulated_tool_calls, event_data
                            )
                    except json.JSONDecodeError:
                        pass
                usage_total = extract_total_tokens(event_data)
                if usage_total > recorded_for_request:
                    self._token_usage_total += usage_total - recorded_for_request
                    recorded_for_request = usage_total
                    chunk.message.additional_kwargs["usage"] = {
                        "total_tokens": usage_total
                    }
                if (
                    self.model in GENERIC_CHAT_MODELS
                    and event_data.get("finishReason") == "tool_calls"
                ):
                    finalized = finalized_stream_tool_calls(
                        accumulated_tool_calls
                    )
                    if finalized:
                        chunk.message.additional_kwargs["tool_calls"] = finalized
                yield chunk
            if recorded_for_request == 0:
                self._token_usage_total += estimate_tokens(
                    messages, kwargs, "".join(response_parts)
                )

        return generate()

    async def acomplete(self, prompt, formatted=False, **kwargs):
        return await asyncio.to_thread(
            self.complete,
            prompt,
            formatted=formatted,
            **kwargs,
        )

    async def astream_chat(self, messages, **kwargs):
        stream = await asyncio.to_thread(self.stream_chat, messages, **kwargs)

        async def generate():
            while True:
                item = await asyncio.to_thread(next_stream_item, stream)
                if item is STREAM_END:
                    break
                yield item

        return generate()

    async def astream_complete(self, prompt, formatted=False, **kwargs):
        stream = await asyncio.to_thread(
            self.stream_complete,
            prompt,
            formatted=formatted,
            **kwargs,
        )

        async def generate():
            while True:
                item = await asyncio.to_thread(next_stream_item, stream)
                if item is STREAM_END:
                    break
                yield item

        return generate()


# Register the models with OCI's GENERIC native chat format until the
# LlamaIndex integration supports their agentic tool calls directly.
CHAT_MODELS.setdefault(OPENAI_GPT_OSS_MODEL, OPENAI_GPT_OSS_CONTEXT_SIZE)
PROVIDERS["openai"] = OCIGenericProvider()
PROVIDERS["meta"] = OCIGenericProvider()


def oci_runtime_config() -> tuple[Path, str, str, str]:
    env = get_settings()
    config_file = Path(env.oci_config_file).expanduser()
    profile = env.oci_profile

    try:
        config = oci.config.from_file(
            file_location=str(config_file),
            profile_name=profile,
        )
        oci.config.validate_config(config)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid OCI configuration in {config_file} (profile {profile!r}): {exc}"
        ) from exc

    compartment_id = (
        env.oci_compartment_id
        or config.get("oci_compartment_id")
        or ""
    ).strip()
    if not compartment_id:
        raise RuntimeError(
            "OCI compartment ID is missing. Set OCI_COMPARTMENT_ID or add "
            "oci_compartment_id to the OCI config profile."
        )

    service_endpoint = (
        env.oci_service_endpoint
        or f"https://inference.generativeai.{config['region']}.oci.oraclecloud.com"
    )
    return config_file, profile, compartment_id, service_endpoint


@register("oci")
class OCIProvider:
    name = "oci"

    def build(self, spec: ModelSpec, settings: Settings, callback_manager: CallbackManager) -> OCIGenAILLM:
        """The OCI client of ``spec``; the account comes from the OCI config file, not from ``settings``."""
        config_file, profile, compartment_id, service_endpoint = oci_runtime_config()
        model_kwargs = {}
        if spec.context_window:
            model_kwargs["context_size"] = spec.context_window
        return OCIGenAILLM(
            model=spec.provider_model,
            compartment_id=compartment_id,
            service_endpoint=service_endpoint,
            auth_type="API_KEY",
            auth_profile=profile,
            auth_file_location=str(config_file),
            temperature=spec.temperature,
            max_tokens=spec.max_output_tokens,
            callback_manager=callback_manager,
            **model_kwargs,
        )
