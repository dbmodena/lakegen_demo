from types import SimpleNamespace

from lakegen.core.token_usage import estimate_tokens, extract_total_tokens


def test_extracts_oci_usage_objects_from_raw_response_shape():
    raw_response = {
        "data": SimpleNamespace(
            chat_response=SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=120,
                    completion_tokens=30,
                    total_tokens=150,
                )
            )
        )
    }

    assert extract_total_tokens(raw_response) == 150


def test_extracts_usage_from_oci_stream_event_json():
    raw_event = {
        "data": '{"usage":{"promptTokens":80,"completionTokens":20}}'
    }

    assert extract_total_tokens(raw_event) == 100


def test_extracts_ollama_usage_for_backward_compatibility():
    assert extract_total_tokens(
        {"prompt_eval_count": 40, "eval_count": 10}
    ) == 50


def test_estimates_tokens_when_provider_usage_is_missing():
    assert estimate_tokens("one two three", {"answer": "four"}) > 0
