import asyncio
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import uuid

import oci
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import ChatMessage, LLM, MessageRole
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.llms.oci_genai import OCIGenAI
from llama_index.llms.oci_genai.utils import CHAT_MODELS, PROVIDERS, XAIProvider

from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient
from lakegen.core.table_io import list_table_files
from lakegen.core.token_usage import extract_total_tokens
from lakegen.core.config import LOG_DIR
from lakegen.retrieval import RetrievalConfig, RetrievalRunLogger, TableRetrievalService


OCI_DEFAULT_PROFILE = "DEFAULT"
OCI_MAX_OUTPUT_TOKENS = 4000
OPENAI_GPT_OSS_MODEL = "openai.gpt-oss-120b"
OPENAI_GPT_OSS_CONTEXT_SIZE = 128_000
OPENAI_GPT_OSS_MAX_OUTPUT_TOKENS = 16_000
META_LLAMA_MODEL = "meta.llama-3.3-70b-instruct"
GENERIC_CHAT_MODELS = {OPENAI_GPT_OSS_MODEL, META_LLAMA_MODEL}


class _OCIGenericProvider(XAIProvider):
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


_STREAM_END = object()


def _next_stream_item(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return _STREAM_END


def _merge_generic_stream_tool_calls(
    accumulated: list[dict[str, Any]],
    event_data: dict[str, Any],
) -> None:
    fragments = event_data.get("message", {}).get("toolCalls", [])
    for index, fragment in enumerate(fragments):
        tool_id = fragment.get("id")
        current = next(
            (
                item
                for item in accumulated
                if tool_id is not None and item["toolUseId"] == tool_id
            ),
            None,
        )
        if current is None and index < len(accumulated):
            current = accumulated[index]
        if current is None:
            current = {
                "toolUseId": tool_id or uuid.uuid4().hex,
                "name": fragment.get("name", ""),
                "input": "",
            }
            accumulated.append(current)

        if tool_id:
            current["toolUseId"] = tool_id
        if fragment.get("name"):
            current["name"] = fragment["name"]
        current["input"] += fragment.get("arguments") or ""


def _finalized_stream_tool_calls(
    accumulated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    finalized = []
    for tool_call in accumulated:
        raw_input = (tool_call.get("input") or "").strip() or "{}"
        try:
            json.loads(raw_input)
        except json.JSONDecodeError:
            continue
        if not tool_call.get("name"):
            continue
        finalized.append({**tool_call, "input": raw_input})
    return finalized


def _parse_tool_call_input(raw_input: Any) -> dict[str, Any] | None:
    if isinstance(raw_input, dict):
        return raw_input
    if raw_input is None or not str(raw_input).strip():
        return {}
    try:
        parsed = json.loads(str(raw_input))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class _LakeGenOCIGenAI(OCIGenAI):
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
        user_msg: Optional[Union[str, ChatMessage]] = None,
        chat_history: Optional[list[ChatMessage]] = None,
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
            arguments = _parse_tool_call_input(tool_call.get("input"))
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
        response = super().chat(messages, **kwargs)
        total = self._record_token_usage(response.raw)
        if not total:
            self._record_token_usage(response.message.additional_kwargs)
        return response

    def stream_chat(self, messages, **kwargs):
        stream = super().stream_chat(messages, **kwargs)
        if self.model not in GENERIC_CHAT_MODELS:
            return stream

        def generate():
            accumulated_tool_calls: list[dict[str, Any]] = []
            recorded_for_request = 0
            for chunk in stream:
                raw = chunk.raw if isinstance(chunk.raw, dict) else {}
                raw_data = raw.get("data", "")
                event_data = {}
                if raw_data:
                    try:
                        event_data = json.loads(raw_data)
                        _merge_generic_stream_tool_calls(
                            accumulated_tool_calls,
                            event_data,
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
                if event_data.get("finishReason") == "tool_calls":
                    finalized = _finalized_stream_tool_calls(
                        accumulated_tool_calls
                    )
                    if finalized:
                        chunk.message.additional_kwargs["tool_calls"] = finalized
                yield chunk

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
                item = await asyncio.to_thread(_next_stream_item, stream)
                if item is _STREAM_END:
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
                item = await asyncio.to_thread(_next_stream_item, stream)
                if item is _STREAM_END:
                    break
                yield item

        return generate()


# Register the models with OCI's GENERIC native chat format until the
# LlamaIndex integration supports their agentic tool calls directly.
CHAT_MODELS.setdefault(OPENAI_GPT_OSS_MODEL, OPENAI_GPT_OSS_CONTEXT_SIZE)
PROVIDERS["openai"] = _OCIGenericProvider()
PROVIDERS["meta"] = _OCIGenericProvider()


def _oci_runtime_config() -> tuple[Path, str, str, str]:
    config_file = Path(
        os.environ.get("OCI_CONFIG_FILE", "~/.oci/config")
    ).expanduser()
    profile = os.environ.get("OCI_PROFILE", OCI_DEFAULT_PROFILE)

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
        os.environ.get("OCI_COMPARTMENT_ID")
        or config.get("oci_compartment_id")
        or ""
    ).strip()
    if not compartment_id:
        raise RuntimeError(
            "OCI compartment ID is missing. Set OCI_COMPARTMENT_ID or add "
            "oci_compartment_id to the OCI config profile."
        )

    service_endpoint = (
        os.environ.get("OCI_SERVICE_ENDPOINT")
        or f"https://inference.generativeai.{config['region']}.oci.oraclecloud.com"
    )
    return config_file, profile, compartment_id, service_endpoint


def get_llm(model: str) -> tuple[LLM, TokenCountingHandler]:
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])

    config_file, profile, compartment_id, service_endpoint = _oci_runtime_config()
    max_tokens = (
        OPENAI_GPT_OSS_MAX_OUTPUT_TOKENS
        if model == OPENAI_GPT_OSS_MODEL
        else OCI_MAX_OUTPUT_TOKENS
    )
    model_kwargs = {}
    if model == OPENAI_GPT_OSS_MODEL:
        model_kwargs["context_size"] = OPENAI_GPT_OSS_CONTEXT_SIZE
    llm = _LakeGenOCIGenAI(
        model=model,
        compartment_id=compartment_id,
        service_endpoint=service_endpoint,
        auth_type="API_KEY",
        auth_profile=profile,
        auth_file_location=str(config_file),
        temperature=0.1,
        max_tokens=max_tokens,
        callback_manager=Settings.callback_manager,
        **model_kwargs,
    )

    return llm, token_counter


@lru_cache(maxsize=8)
def get_solr(core):
    return LocalSolrClient(
        core=core,
        base_url=os.environ.get("SOLR_BASE_URL", "http://localhost:8983/solr"),
    )


@lru_cache(maxsize=1)
def get_retrieval_run_logger() -> RetrievalRunLogger:
    return RetrievalRunLogger(LOG_DIR / "retrieval_rankings.jsonl")


def get_table_retrieval_service(
    solr: LocalSolrClient,
    config: RetrievalConfig,
) -> TableRetrievalService:
    return TableRetrievalService(
        solr,
        config,
        observer=get_retrieval_run_logger(),
    )


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    return PromptManager()


def get_all_table_files(table_dir):
    return list_table_files(table_dir)


# Backward-compatible name for callers outside this repository.
get_all_csv_files = get_all_table_files
