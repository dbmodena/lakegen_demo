from pathlib import Path

import pytest

from lakegen.core import resources


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


def test_gpt_oss_provider_is_registered():
    assert resources.CHAT_MODELS["openai.gpt-oss-120b"] == 128_000
    assert isinstance(resources.PROVIDERS["openai"], resources._OCIGenericProvider)
    assert isinstance(resources.PROVIDERS["meta"], resources._OCIGenericProvider)


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
