"""The entities a question names, for the Seeker's content search: extracted by an OCI, dedicated or Together model."""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Protocol

import requests

from lakegen.config.registry import models, models_by_provider
from lakegen.config.settings import dedicated_model, get_settings, together_api_key
from lakegen.llm.rerank import _oci_client

# The reference code's entity-extraction system prompt, verbatim
# (PneumaRetriever.__get_entity_extraction_sys_prompt); the question is the user
# message.
ENTITY_EXTRACTION_PROMPT = """You are an information extraction system.

Your task is to analyze a natural-language query and extract **explicitly mentioned, concrete entities** that are suitable for direct lookup in a structured dataset.

**Extraction rules:**

* Only extract entities that are **specific, named, and canonical**, such as identifiers, symbols, or codes that would typically appear verbatim in a database.
* Do **not** extract general concepts or categories.
* If the query does **not** contain any extractable entities under these rules, return an empty list ({"entities": []}).

**Output requirements:**

* Output **only** a JSON object.
* The JSON object must contain a single key `"entities"` whose value is a list of strings.
* Each string must exactly match how the entity appears in the query.
* Preserve original casing and punctuation.
* Do not include duplicates.
* Do not include any explanation, comments, formatting, or additional text outside the JSON object.

**Output format:**
{"entities": [...]}"""


class EntityExtractor(Protocol):
    """Extracts the entities a question names, for the Seeker's content search."""

    def entities(self, question: str) -> list[str]:
        """The entities the question names, as written; may be empty."""


def _parse_entities(text: str) -> list[str]:
    """The ``entities`` list of a model's JSON reply, strings only."""
    match = re.search(r"\{.*\}", text, re.S)
    decoded = json.loads(match.group(0) if match else text)
    values = decoded.get("entities") if isinstance(decoded, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"no entities list in {text[:200]!r}")
    return [value for value in values if isinstance(value, str) and value.strip()]


@lru_cache(maxsize=512)
def _oci_entities(model_id: str, question: str) -> tuple[str, ...]:
    """The reference extraction on OCI: same prompt, temperature 0, JSON-object reply."""
    from oci.generative_ai_inference import models

    client, compartment = _oci_client()
    response = client.chat(
        models.ChatDetails(
            compartment_id=compartment,
            serving_mode=models.OnDemandServingMode(model_id=model_id),
            chat_request=models.GenericChatRequest(
                messages=[
                    models.SystemMessage(
                        content=[models.TextContent(text=ENTITY_EXTRACTION_PROMPT)]
                    ),
                    models.UserMessage(content=[models.TextContent(text=question)]),
                ],
                # Room for a reasoning model's thinking before the JSON.
                max_tokens=4096,
                temperature=0.0,
                response_format=models.JsonObjectResponseFormat(),
            ),
        )
    )
    message = response.data.chat_response.choices[0].message
    text = "".join(getattr(part, "text", None) or "" for part in message.content or ())
    return tuple(_parse_entities(text))


# Together AI models for the same extraction (pneuma_entity_model "together.*"),
# so Pneuma-Seeker needs no OCI chat model: LakeGen model name -> Together id.
TOGETHER_ENTITY_MODELS = models_by_provider("together")
TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"
# The OCI dedicated endpoints serve the same OpenAI-compatible route: LakeGen
# model name -> the .env variable holding the endpoint's model id.
DEDICATED_ENTITY_MODELS = {
    spec.name: spec.options["model_env"]
    for spec in models()
    if spec.provider == "oci_dedicated" and spec.options.get("model_env")
}


@lru_cache(maxsize=512)
def _together_entities(model: str, question: str) -> tuple[str, ...]:
    """The reference extraction on Together: same prompt, temperature 0, JSON-object reply."""
    api_key = together_api_key()
    if not api_key:
        raise RuntimeError(f"{model} needs a Together AI key: set TOGETHERAI_API_KEY")
    return _chat_entities(TOGETHER_CHAT_URL, api_key, TOGETHER_ENTITY_MODELS[model], question)


@lru_cache(maxsize=512)
def _dedicated_entities(model: str, question: str) -> tuple[str, ...]:
    """The same extraction on a dedicated endpoint (DEDICATE_OCI_* in .env)."""
    settings = get_settings()
    variable = DEDICATED_ENTITY_MODELS[model]
    model_id = dedicated_model(variable)
    missing = [
        name
        for name, value in (
            (variable, model_id),
            ("DEDICATE_OCI_API_KEY", settings.dedicated_api_key),
            ("DEDICATE_OCI_BASE_URL", settings.dedicated_base_url),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"{model} needs its dedicated endpoint in .env: set {', '.join(missing)}"
        )
    return _chat_entities(
        settings.dedicated_base_url + "/chat/completions",
        settings.dedicated_api_key,
        model_id,
        question,
        headers=settings.dedicated_headers,
    )


def _chat_entities(
    url: str,
    api_key: str,
    model_id: str,
    question: str,
    headers: dict[str, str] | None = None,
) -> tuple[str, ...]:
    """One OpenAI-compatible chat call: the prompt, temperature 0, a JSON-object reply, Together's retry rules."""
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
            {"role": "user", "content": question},
        ],
        # Room for a reasoning model's thinking before the JSON, as on OCI.
        "max_tokens": 4096,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        **(headers or {}),
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    started = time.monotonic()
    for attempt in range(10_000):
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 402 and time.monotonic() - started < 86400:
            time.sleep(300)  # credit limit reached: wait for credit, as the LLM client does
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 5:
            # Together's dynamic limits: wait at least its suggested interval.
            try:
                suggested = float(response.headers.get("x-ratelimit-reset") or 0)
            except ValueError:
                suggested = 0.0
            time.sleep(max(suggested, min(60.0, 2.0 * 2**attempt)))
            continue
        response.raise_for_status()
        text = response.json()["choices"][0]["message"].get("content") or ""
        return tuple(_parse_entities(text))
    raise RuntimeError(f"{url} kept refusing the entity extraction")


class TogetherEntityExtractor:
    """The reference entity extraction through Together AI, cached per process.

    Args:
        model: The LakeGen name of a Together model (``together.*``).

    Attributes:
        model: The LakeGen model name.

    Raises:
        ValueError: When ``model`` is not a Together model.
    """

    def __init__(self, model: str) -> None:
        if model not in TOGETHER_ENTITY_MODELS:
            raise ValueError(f"unknown Together entity model {model!r}")
        self.model = model

    def entities(self, question: str) -> list[str]:
        """The entities the question names, as written; may be empty."""
        return list(_together_entities(self.model, question))


class DedicatedEntityExtractor:
    """The same extraction on an OCI dedicated endpoint, cached per process.

    Args:
        model: The LakeGen name of a dedicated endpoint model (``dedicated.*``).

    Attributes:
        model: The LakeGen model name.

    Raises:
        ValueError: When ``model`` is not a dedicated endpoint model.
    """

    def __init__(self, model: str) -> None:
        if model not in DEDICATED_ENTITY_MODELS:
            raise ValueError(f"unknown dedicated entity model {model!r}")
        self.model = model

    def entities(self, question: str) -> list[str]:
        """The entities the question names, as written; may be empty."""
        return list(_dedicated_entities(self.model, question))


def entity_extractor_for(model: str) -> EntityExtractor:
    """Together for ``together.*`` names, the dedicated endpoints for ``dedicated.*``, OCI for OCI model ids."""
    if model in TOGETHER_ENTITY_MODELS:
        return TogetherEntityExtractor(model)
    if model in DEDICATED_ENTITY_MODELS:
        return DedicatedEntityExtractor(model)
    return OCIEntityExtractor(model)


class OCIEntityExtractor:
    """The reference code's entity extraction, one OCI chat call per question.

    Answers are cached per process, so a question retrieved twice is extracted
    once.

    Args:
        model_id: The OCI model id.

    Attributes:
        model_id: The OCI model id.
    """

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def entities(self, question: str) -> list[str]:
        """The entities the question names, as written; may be empty."""
        return list(_oci_entities(self.model_id, question))
