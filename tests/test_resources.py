import json
from pathlib import Path

import pytest

from lakegen.core import resources
from lakegen.retrieval import RetrievalConfig, RetrievalMode


def test_oci_runtime_config_uses_profile_and_region(monkeypatch, tmp_path):
    config_file = tmp_path / "config"
    config_file.touch()
    config = {
        "user": "user",
        "fingerprint": "fingerprint",
        "tenancy": "tenancy",
        "region": "eu-frankfurt-1",
        "key_file": str(tmp_path / "key.pem"),
        "oci_compartment_id": "compartment",
    }
    calls = {}

    def fake_from_file(*, file_location, profile_name):
        calls.update(file_location=file_location, profile_name=profile_name)
        return config

    monkeypatch.setenv("OCI_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("OCI_PROFILE", "LAKEGEN")
    monkeypatch.setattr(resources.oci.config, "from_file", fake_from_file)
    monkeypatch.setattr(resources.oci.config, "validate_config", lambda value: None)

    result = resources._oci_runtime_config()

    assert result == (
        config_file,
        "LAKEGEN",
        "compartment",
        "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
    )
    assert calls == {
        "file_location": str(config_file),
        "profile_name": "LAKEGEN",
    }


def test_oci_runtime_config_requires_compartment(monkeypatch, tmp_path):
    monkeypatch.delenv("OCI_COMPARTMENT_ID", raising=False)
    monkeypatch.setattr(
        resources.oci.config,
        "from_file",
        lambda **kwargs: {
            "user": "user",
            "fingerprint": "fingerprint",
            "tenancy": "tenancy",
            "region": "eu-frankfurt-1",
            "key_file": str(tmp_path / "key.pem"),
        },
    )
    monkeypatch.setattr(resources.oci.config, "validate_config", lambda value: None)

    with pytest.raises(RuntimeError, match="compartment ID is missing"):
        resources._oci_runtime_config()


def test_get_solr_honors_configurable_base_url(monkeypatch):
    resources.get_solr.cache_clear()
    monkeypatch.setenv("SOLR_BASE_URL", "http://127.0.0.1:8993/solr")

    client = resources.get_solr("nyc")

    assert client.base_url == "http://127.0.0.1:8993/solr"
    resources.get_solr.cache_clear()


def test_get_solr_serves_a_portal_from_the_configured_core(monkeypatch):
    resources.get_solr.cache_clear()
    monkeypatch.setenv("LAKEGEN_SOLR_CORE_UK", "uk_fixed")

    assert resources.get_solr("uk").core == "uk_fixed"
    assert resources.get_solr("nyc").core == "nyc"
    resources.get_solr.cache_clear()


def test_get_solr_uses_the_portal_name_without_an_override(monkeypatch):
    resources.get_solr.cache_clear()
    monkeypatch.delenv("LAKEGEN_SOLR_CORE_UK", raising=False)
    monkeypatch.setenv("LAKEGEN_SOLR_CORE_NYC", "  ")

    assert resources.get_solr("uk").core == "uk"
    assert resources.get_solr("nyc").core == "nyc"
    resources.get_solr.cache_clear()


def test_get_llm_builds_oci_client(monkeypatch):
    captured = {}
    fake_llm = object()

    def fake_oci_genai(**kwargs):
        captured.update(kwargs)
        return fake_llm

    monkeypatch.setattr(
        resources,
        "_oci_runtime_config",
        lambda: (
            Path("/secure/config"),
            "DEFAULT",
            "compartment",
            "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com",
        ),
    )
    monkeypatch.setattr(resources, "_LakeGenOCIGenAI", fake_oci_genai)
    monkeypatch.setattr(
        resources.tiktoken,
        "encoding_for_model",
        lambda _model: type(
            "OfflineEncoding", (), {"encode": staticmethod(lambda text: list(text))}
        )(),
    )

    llm, token_counter = resources.get_llm("openai.gpt-oss-120b")

    assert llm is fake_llm
    assert token_counter is not None
    assert captured == {
        "model": "openai.gpt-oss-120b",
        "compartment_id": "compartment",
        "service_endpoint": (
            "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"
        ),
        "auth_type": "API_KEY",
        "auth_profile": "DEFAULT",
        "auth_file_location": "/secure/config",
        "temperature": 0.1,
        "max_tokens": 16000,
        "callback_manager": resources.Settings.callback_manager,
        "context_size": 128_000,
    }


@pytest.mark.parametrize(
    ("model", "tag"),
    [("ollama.gpt-oss-20b", "gpt-oss:20b"), ("ollama.qwen3.8-27b", "qwen3.8:27b")],
)
def test_get_llm_builds_local_ollama_client_without_oci(monkeypatch, model, tag):
    def no_oci():
        raise AssertionError("a local model must not need the OCI configuration")

    monkeypatch.setattr(resources, "_oci_runtime_config", no_oci)
    monkeypatch.setenv("LAKEGEN_OLLAMA_BASE_URL", "http://gpu-host:11434/")
    monkeypatch.setattr(
        resources.tiktoken,
        "encoding_for_model",
        lambda _model: type(
            "OfflineEncoding", (), {"encode": staticmethod(lambda text: list(text))}
        )(),
    )

    llm, token_counter = resources.get_llm(model)

    assert isinstance(llm, resources._LakeGenOllamaLLM)
    assert token_counter is not None
    assert llm.model == tag
    assert llm.api_base == "http://gpu-host:11434/v1"
    assert llm.metadata.context_window == resources.OLLAMA_CONTEXT_SIZE
    assert llm.metadata.is_function_calling_model


def test_ollama_chat_drops_think_and_counts_tokens(monkeypatch):
    captured = {}

    def fake_stream_chat(_self, messages, **kwargs):
        captured.update(messages=messages, kwargs=kwargs)
        chunk = type("Chunk", (), {
            "raw": {"usage": {"prompt_tokens": 7, "completion_tokens": 5}},
            "message": type("Message", (), {"content": "done"})(),
        })()
        return iter([chunk])

    monkeypatch.setattr(resources.OpenAILike, "stream_chat", fake_stream_chat)
    llm = resources._LakeGenOllamaLLM(model="gpt-oss:20b", api_key="ollama")

    assert len(list(llm.stream_chat(["message"], think=True, request_id="run-1"))) == 1
    assert captured == {"messages": ["message"], "kwargs": {"request_id": "run-1"}}
    assert llm.token_usage_total == 12
    llm.reset_token_usage()
    assert llm.token_usage_total == 0


def test_gpt_oss_provider_is_registered():
    assert resources.CHAT_MODELS["openai.gpt-oss-120b"] == 128_000
    assert isinstance(resources.PROVIDERS["openai"], resources._OCIGenericProvider)
    assert isinstance(resources.PROVIDERS["meta"], resources._OCIGenericProvider)


def test_oci_stream_drops_unsupported_think_option(monkeypatch):
    captured = {}

    def fake_stream_chat(_self, messages, **kwargs):
        captured.update(messages=messages, kwargs=kwargs)
        return iter(())

    monkeypatch.setattr(resources.OCIGenAI, "stream_chat", fake_stream_chat)
    llm = resources._LakeGenOCIGenAI.model_construct(
        model=resources.OPENAI_GPT_OSS_MODEL
    )

    assert list(llm.stream_chat(["message"], think=True, request_id="run-1")) == []
    assert captured == {
        "messages": ["message"],
        "kwargs": {"request_id": "run-1"},
    }


def test_generic_stream_tool_call_fragments_are_accumulated():
    accumulated = []

    resources._merge_generic_stream_tool_calls(
        accumulated,
        {
            "message": {
                "toolCalls": [
                    {
                        "id": "call-1",
                        "name": "lookup_table",
                        "arguments": "{\"name\":",
                    }
                ]
            }
        },
    )
    resources._merge_generic_stream_tool_calls(
        accumulated,
        {"message": {"toolCalls": [{"arguments": "\"parks\"}"}]}},
    )

    assert accumulated == [
        {
            "toolUseId": "call-1",
            "name": "lookup_table",
            "input": "{\"name\":\"parks\"}",
        }
    ]
    assert resources._finalized_stream_tool_calls(accumulated) == accumulated


def test_tool_call_input_parser_tolerates_empty_and_partial_json():
    assert resources._parse_tool_call_input("") == {}
    assert resources._parse_tool_call_input(None) == {}
    assert resources._parse_tool_call_input('{"name": "parks"}') == {
        "name": "parks"
    }
    assert resources._parse_tool_call_input('{"name":') is None


def test_incomplete_stream_tool_calls_are_not_exposed():
    assert resources._finalized_stream_tool_calls(
        [{"toolUseId": "call-1", "name": "lookup_table", "input": ""}]
    ) == [
        {"toolUseId": "call-1", "name": "lookup_table", "input": "{}"}
    ]
    assert resources._finalized_stream_tool_calls(
        [{"toolUseId": "call-1", "name": "lookup_table", "input": "{\"name\":"}]
    ) == []


def test_retrieval_runs_are_logged_and_captured_for_the_api_csv(monkeypatch):
    persisted = []

    class FakeSolr:
        def select(self, tokens, **params):
            return {
                "response": {
                    "docs": [{"resource_id": "parks", "score": 2.5}]
                }
            }

    monkeypatch.setattr(resources, "get_retrieval_run_logger", lambda: persisted.append)
    service = resources.get_table_retrieval_service(
        FakeSolr(), RetrievalConfig(mode=RetrievalMode.KEYWORD)
    )

    with resources.capture_retrieval_runs(
        {"JOB_ID": "job-1", "SOURCE_PATH": "$.questions[0]", "SOURCE_ID": 7}
    ) as captured:
        service.retrieve(question="Where are the parks?", keywords=["parks"])

    assert len(persisted) == 1
    assert captured[0]["mode"] == "keyword"
    assert captured[0]["hits"][0]["resource_id"] == "parks"
    assert captured[0]["job_id"] == "job-1"
    assert captured[0]["source_path"] == "$.questions[0]"
    assert captured[0]["source_id"] == 7
    assert captured[0]["retrieval_attempt"] == 1
    assert "timestamp" in captured[0]


# --------------------------------------------------------------- Groq offload

from llama_index.core.llms import ChatMessage  # noqa: E402

from lakegen.core import groq_offload  # noqa: E402


class _FakeGroqResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body or {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def _groq_ok(content="answer", finish="stop", total=900):
    return _FakeGroqResponse(body={
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": total - 100, "completion_tokens": 100, "total_tokens": total},
    })


def _offload(tmp_path, monkeypatch, replies, keys=("k1", "k2")):
    sent = []

    def fake_post(url, headers, payload):
        sent.append({"key": headers["Authorization"].split()[-1], **payload})
        return replies.pop(0)

    monkeypatch.setattr(groq_offload, "_post", fake_post)
    return groq_offload.GroqOffload(list(keys), tmp_path / "groq.jsonl"), sent


def _ledger(tmp_path):
    return [json.loads(line) for line in (tmp_path / "groq.jsonl").read_text().splitlines()]


def test_groq_offload_answers_a_plain_call_and_logs_it(tmp_path, monkeypatch):
    offload, sent = _offload(tmp_path, monkeypatch, [_groq_ok()])
    messages = [ChatMessage(role="system", content="judge"), ChatMessage(role="user", content="is it right?")]

    response = offload.chat("openai.gpt-oss-120b", messages, {"think": True}, 0.1)

    assert response.message.content == "answer"
    assert resources.extract_total_tokens(response.raw) == 900
    assert sent[0]["model"] == "openai/gpt-oss-120b"
    assert [m["role"] for m in sent[0]["messages"]] == ["system", "user"]
    assert sent[0]["temperature"] == 0.1
    # Prompt plus requested output stays under the free tier's per-minute cap.
    assert groq_offload.MIN_OUTPUT_TOKENS <= sent[0]["max_completion_tokens"] <= groq_offload.MAX_OUTPUT_TOKENS
    entry = _ledger(tmp_path)[0]
    assert entry["outcome"] == "ok" and entry["key"] == 1 and entry["total_tokens"] == 900


def test_groq_offload_leaves_agent_turns_and_large_prompts_to_oci(tmp_path, monkeypatch):
    offload, sent = _offload(tmp_path, monkeypatch, [])
    plain = [ChatMessage(role="user", content="hi")]
    tool_turn = ChatMessage(role="assistant", content="", additional_kwargs={"tool_calls": [{"name": "search"}]})

    assert offload.chat("openai.gpt-oss-120b", plain, {"tools": ["search"]}, 0.1) is None
    assert offload.chat("openai.gpt-oss-120b", [*plain, tool_turn], {}, 0.1) is None
    assert offload.chat("openai.gpt-oss-120b", [ChatMessage(role="tool", content="rows")], {}, 0.1) is None
    huge = [ChatMessage(role="user", content="word " * 6_000)]
    assert offload.chat("openai.gpt-oss-120b", huge, {}, 0.1) is None
    assert sent == []
    assert [entry["outcome"] for entry in _ledger(tmp_path)] == ["skip_large"]


def test_groq_offload_falls_back_and_rests_a_rate_limited_key(tmp_path, monkeypatch):
    replies = [_FakeGroqResponse(429, {"error": "TPD"}, {"retry-after": "3600"}), _groq_ok("from key 2")]
    offload, sent = _offload(tmp_path, monkeypatch, replies)
    plain = [ChatMessage(role="user", content="hi")]

    assert offload.chat("openai.gpt-oss-120b", plain, {}, 0.1) is None
    assert offload.chat("openai.gpt-oss-120b", plain, {}, 0.1).message.content == "from key 2"
    assert [call["key"] for call in sent] == ["k1", "k2"]
    # Key 1 rests for the hour Groq asked; key 2 has used its minute's room.
    assert offload.chat("openai.gpt-oss-120b", plain, {}, 0.1) is None
    assert [entry["outcome"] for entry in _ledger(tmp_path)] == ["http_429", "ok", "skip_busy"]


def test_groq_offload_rejects_truncated_answers(tmp_path, monkeypatch):
    offload, _ = _offload(tmp_path, monkeypatch, [_groq_ok("half", finish="length")])

    assert offload.chat("openai.gpt-oss-120b", [ChatMessage(role="user", content="hi")], {}, 0.1) is None
    assert _ledger(tmp_path)[0]["outcome"] == "unusable_length"


def test_groq_offload_restores_daily_usage_from_the_ledger(tmp_path, monkeypatch):
    now = groq_offload.datetime.now(groq_offload.timezone.utc).isoformat()
    (tmp_path / "groq.jsonl").write_text(json.dumps(
        {"at": now, "key": 1, "sent": True, "outcome": "ok", "total_tokens": 189_000}) + "\n")
    offload, sent = _offload(tmp_path, monkeypatch, [_groq_ok()], keys=("k1",))

    assert offload.chat("openai.gpt-oss-120b", [ChatMessage(role="user", content="hi")], {}, 0.1) is None
    assert sent == []


def test_groq_offload_is_off_unless_enabled(monkeypatch):
    monkeypatch.delenv("LAKEGEN_GROQ_OFFLOAD", raising=False)
    monkeypatch.setenv("GROP_API_KEY", "secret")
    assert not groq_offload.enabled_for("openai.gpt-oss-120b")
    monkeypatch.setenv("LAKEGEN_GROQ_OFFLOAD", "1")
    assert groq_offload.enabled_for("openai.gpt-oss-120b")
    assert not groq_offload.enabled_for("meta.llama-3.3-70b-instruct")


def test_oci_chat_uses_the_groq_answer_when_offloaded(monkeypatch):
    def no_oci(*_args, **_kwargs):
        raise AssertionError("an offloaded call must not reach OCI")

    monkeypatch.setattr(resources.OCIGenAI, "chat", no_oci)
    answer = groq_offload.ChatResponse(
        message=ChatMessage(role="assistant", content="ok"), raw={"usage": {"total_tokens": 321}})
    monkeypatch.setattr(resources.groq_offload, "chat", lambda *_args: answer)
    llm = resources._LakeGenOCIGenAI.model_construct(model=resources.OPENAI_GPT_OSS_MODEL, temperature=0.1)

    assert llm.chat([ChatMessage(role="user", content="hi")]).message.content == "ok"
    assert llm.token_usage_total == 321


def test_get_llm_builds_together_client_without_oci(monkeypatch):
    def no_oci():
        raise AssertionError("a Together model must not need the OCI configuration")

    monkeypatch.setattr(resources, "_oci_runtime_config", no_oci)
    monkeypatch.setenv("TOGETHERAI_API_KEY", "together-secret")
    monkeypatch.setattr(
        resources.tiktoken,
        "encoding_for_model",
        lambda _model: type(
            "OfflineEncoding", (), {"encode": staticmethod(lambda text: list(text))}
        )(),
    )

    llm, _ = resources.get_llm("together.gpt-oss-120b")

    assert isinstance(llm, resources._LakeGenTogetherLLM)
    assert llm.model == "openai/gpt-oss-120b"
    assert llm.api_base == "https://api.together.xyz/v1"
    assert llm.api_key == "together-secret"
    assert llm.metadata.is_function_calling_model
    assert llm.metadata.context_window == resources.OPENAI_GPT_OSS_CONTEXT_SIZE
    # Streams must end with a usage chunk for the token accounting.
    assert llm.additional_kwargs["stream_options"] == {"include_usage": True}


def test_get_llm_needs_a_together_key(monkeypatch):
    monkeypatch.delenv("TOGETHERAI_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TOGETHERAI_API_KEY"):
        resources.get_llm("together.gpt-oss-120b")


def _together_error(status, headers=None):
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.together.xyz/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    cls = {429: openai.RateLimitError, 400: openai.BadRequestError}.get(status, openai.InternalServerError)
    return cls("limited", response=response, body=None)


def test_together_retries_rate_limits_with_the_suggested_wait(monkeypatch):
    calls, waits = [], []
    replies = [_together_error(429, {"x-ratelimit-reset": "7"}), _together_error(503), "ok"]

    def fake_chat(_self, messages, **kwargs):
        calls.append(messages)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(resources._LakeGenOllamaLLM, "chat", fake_chat)
    monkeypatch.setattr(resources.time, "sleep", waits.append)
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")

    assert llm.chat(["m"]) == "ok"
    assert len(calls) == 3
    assert waits[0] >= 7  # at least Together's x-ratelimit-reset
    assert 0 < waits[1] <= resources.TOGETHER_BACKOFF_MAX_SECONDS


def test_together_does_not_retry_request_errors(monkeypatch):
    def fake_chat(_self, messages, **kwargs):
        raise _together_error(400)

    monkeypatch.setattr(resources._LakeGenOllamaLLM, "chat", fake_chat)
    monkeypatch.setattr(resources.time, "sleep", lambda _s: (_ for _ in ()).throw(AssertionError("no wait")))
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")

    with pytest.raises(Exception, match="limited"):
        llm.chat(["m"])


def test_together_stream_retries_until_the_first_chunk(monkeypatch):
    attempts = []

    def fake_stream_chat(_self, messages, **kwargs):
        attempts.append(1)

        def generate():
            if len(attempts) == 1:
                raise _together_error(429)
            yield "a"
            yield "b"

        return generate()

    monkeypatch.setattr(resources._LakeGenOllamaLLM, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(resources.time, "sleep", lambda _s: None)
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")

    assert list(llm.stream_chat(["m"])) == ["a", "b"]
    assert len(attempts) == 2


def test_together_async_chat_retries_rate_limits(monkeypatch):
    import asyncio

    replies = [_together_error(429), "ok"]
    waits = []

    async def fake_achat(_self, messages, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def fake_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(resources._LakeGenOllamaLLM, "achat", fake_achat)
    monkeypatch.setattr(resources.asyncio, "sleep", fake_sleep)
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")

    assert asyncio.run(llm.achat(["m"])) == "ok"
    assert len(waits) == 1


def test_together_allows_parallel_calls_in_the_request_by_default(monkeypatch):
    monkeypatch.delenv("LAKEGEN_TOGETHER_PARALLEL_TOOL_CALLS", raising=False)
    from llama_index.core.tools import FunctionTool

    def add(a: int, b: int) -> int:
        "Add two numbers."
        return a + b

    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")
    request = llm._prepare_chat_with_tools([FunctionTool.from_defaults(add)], user_msg="hi",
                                           allow_parallel_tool_calls=True)
    assert request["parallel_tool_calls"] is True  # allowed; extra calls are dropped client-side


def test_together_keeps_only_the_first_tool_call():
    from llama_index.core.base.llms.types import ChatResponse, TextBlock, ToolCallBlock

    message = ChatMessage(role="assistant", blocks=[
        TextBlock(text="running"),
        ToolCallBlock(tool_call_id="a", tool_name="run_analysis", tool_kwargs={"code": "df.head()"}),
        ToolCallBlock(tool_call_id="b", tool_name="run_analysis", tool_kwargs={"code": "print(1)"}),
    ], additional_kwargs={"tool_calls": ["first", "second"]})
    response = ChatResponse(message=message)
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")

    calls = llm.get_tool_calls_from_response(response)

    assert [call.tool_id for call in calls] == ["a"]
    assert [b.tool_call_id for b in response.message.blocks if isinstance(b, ToolCallBlock)] == ["a"]
    assert response.message.additional_kwargs["tool_calls"] == ["first"]


def test_together_reasoning_is_kept_and_sent_back(monkeypatch):
    from types import SimpleNamespace

    raw = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(reasoning="check the year column"))])

    def fake_chat(_self, messages, **kwargs):
        from llama_index.core.base.llms.types import ChatResponse
        return ChatResponse(message=ChatMessage(role="assistant", content="ok"), raw=raw)

    monkeypatch.setattr(resources._LakeGenOllamaLLM, "chat", fake_chat)
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")
    reply = llm.chat([ChatMessage(role="user", content="hi")])
    assert reply.message.additional_kwargs["reasoning"] == "check the year column"

    import llama_index.llms.openai.base as openai_base
    dicts = openai_base.to_openai_message_dicts([ChatMessage(role="user", content="hi"), reply.message])
    assert dicts[1]["reasoning"] == "check the year column"
    assert "reasoning" not in dicts[0]
    # Messages without reasoning (every other backend) are serialized as before.
    plain = openai_base.to_openai_message_dicts([ChatMessage(role="assistant", content="x")])
    assert "reasoning" not in plain[0]


def test_together_stream_accumulates_reasoning(monkeypatch):
    from types import SimpleNamespace
    from llama_index.core.base.llms.types import ChatResponse

    def chunk(reasoning):
        raw = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning=reasoning))])
        return ChatResponse(message=ChatMessage(role="assistant", content=""), raw=raw, delta="")

    def fake_stream_chat(_self, messages, **kwargs):
        return iter([chunk("look at "), chunk("the columns"), chunk(None)])

    monkeypatch.setattr(resources._LakeGenOllamaLLM, "stream_chat", fake_stream_chat)
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")
    chunks = list(llm.stream_chat([ChatMessage(role="user", content="hi")]))
    assert chunks[-1].message.additional_kwargs["reasoning"] == "look at the columns"


def test_together_recovers_a_tool_call_that_arrived_as_text():
    from llama_index.core.base.llms.types import ChatResponse, ToolCallBlock

    leaked = ('analysisNeed the columns first.assistantcommentary to=functions.inspect_data json'
              '{\n  "tables": ["r6e8-2fwe.parquet"],\n  "columns": null\n}')
    response = ChatResponse(message=ChatMessage(role="assistant", content=leaked))
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")

    calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)

    assert [(c.tool_name, c.tool_kwargs) for c in calls] == [
        ("inspect_data", {"tables": ["r6e8-2fwe.parquet"], "columns": None})]
    assert response.message.content == ""
    assert response.message.additional_kwargs["reasoning"] == "Need the columns first."
    blocks = [b for b in response.message.blocks if isinstance(b, ToolCallBlock)]
    assert len(blocks) == 1 and blocks[0].tool_call_id.startswith("call_")


def test_together_leaves_plain_answers_and_broken_leaks_alone():
    from llama_index.core.base.llms.types import ChatResponse

    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")
    plain = ChatResponse(message=ChatMessage(role="assistant", content="The answer is 42."))
    broken = ChatResponse(message=ChatMessage(
        role="assistant", content='commentary to=functions.run_analysis json{"code": "import pan'))

    assert llm.get_tool_calls_from_response(plain, error_on_no_tool_call=False) == []
    assert llm.get_tool_calls_from_response(broken, error_on_no_tool_call=False) == []
    assert plain.message.content == "The answer is 42."


def test_together_sends_tool_call_turns_with_empty_content_not_null():
    import llama_index.llms.openai.base as openai_base
    from llama_index.core.base.llms.types import ToolCallBlock

    turn = ChatMessage(role="assistant", blocks=[
        ToolCallBlock(tool_call_id="c1", tool_name="run_code", tool_kwargs={"code": "print(1)"})])
    history = [ChatMessage(role="user", content="go"), turn]

    # Other backends: unchanged (the OpenAI client sends null for a bare tool call).
    assert openai_base.to_openai_message_dicts(history)[1]["content"] is None
    token = resources._TOGETHER_REQUEST.set(True)
    try:
        sent = openai_base.to_openai_message_dicts(history)[1]
    finally:
        resources._TOGETHER_REQUEST.reset(token)
    assert sent["content"] == "" and sent["tool_calls"]


def test_together_waits_for_credit_instead_of_failing(monkeypatch):
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.together.xyz/v1/chat/completions")
    credit = openai.APIStatusError("Credit limit exceeded", response=httpx.Response(402, request=request), body=None)
    replies = [credit, credit, "ok"]
    waits = []

    def fake_chat(_self, messages, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(resources._LakeGenOllamaLLM, "chat", fake_chat)
    monkeypatch.setattr(resources.time, "sleep", waits.append)
    llm = resources._LakeGenTogetherLLM(model="openai/gpt-oss-120b", api_key="k")

    assert llm.chat(["m"]) == "ok"
    assert waits == [resources.TOGETHER_CREDIT_CHECK_SECONDS] * 2
