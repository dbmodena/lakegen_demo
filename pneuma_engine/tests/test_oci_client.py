import json
from types import SimpleNamespace

from pneuma_engine.oci_client import OCICompatOpenAI, OCISettings, _Chat


class _FakeChatEndpoint:
    """An OpenAI-compatible client that records requests and replays answers."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request)))
        text = self.answers.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _endpoint_chat_client(endpoint, **settings):
    # Built without __init__: no OCI config is needed to exercise the chat path.
    client = object.__new__(OCICompatOpenAI)
    client._oci_settings = OCISettings(
        chat_base_url="http://127.0.0.1:11434/v1", retry_initial_delay=0, **settings
    )
    client._chat_client = endpoint
    client._oci_client = None
    return client


def test_pneuma_chat_on_an_openai_endpoint_keeps_the_oci_contract():
    """Judge requests reach the local model with the settings OCI would get."""
    endpoint = _FakeChatEndpoint('{"relevant": true, "reason": "r"}')
    client = _endpoint_chat_client(endpoint)
    schema = {"type": "object", "properties": {"relevant": {"type": "boolean"}}}

    text = client.chat_completion(
        [{"role": "system", "content": "judge"}, {"role": "user", "content": "doc"}],
        model="gpt-4o-mini",
        max_tokens=200,
        temperature=0.0,
        top_p=1.0,
        seed=7,
        json_schema=schema,
        json_schema_name="verdict",
    )

    assert text == '{"relevant": true, "reason": "r"}'
    [request] = endpoint.requests
    # Pneuma's OpenAI default name never reaches the endpoint.
    assert request["model"] == "gpt-oss:20b"
    assert request["max_tokens"] == 1024  # the OCI floor, as for OCI
    assert request["reasoning_effort"] == "low" and request["seed"] == 7
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": schema, "strict": True},
    }


def test_pneuma_narration_through_openai_client_surface_uses_the_endpoint():
    """Pneuma's summarizer calls chat.completions.create on the client it holds."""
    endpoint = _FakeChatEndpoint("", "The borough of the incident.")
    client = _endpoint_chat_client(endpoint, chat_model="local-model")
    client.chat = _Chat(client)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Describe the boro column."}],
        max_tokens=400,
    )

    assert response.choices[0].message.content == "The borough of the incident."
    first, second = endpoint.requests
    assert first["model"] == second["model"] == "local-model"
    # An empty answer is retried with twice the budget and a plain-text nudge.
    assert second["max_tokens"] == 2 * first["max_tokens"]
    assert second["messages"][-1]["content"].endswith(
        "Return the requested brief answer as plain text."
    )
    assert "response_format" not in first
