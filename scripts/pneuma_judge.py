"""Structured LLM Judge for Pneuma's hybrid retrieval (paper §5.1).

The paper's judge classifies each document-question pair as relevant or not;
documents judged irrelevant move behind the relevant ones, and each group keeps
its fused order. Pneuma 0.0.4 asks an OpenAI-compatible provider for that
verdict as free text, which breaks it twice:

* ``prompt_openai_llm`` samples at temperature 0.7, while the paper's local
  pipeline decodes greedily, so one document-question pair flips between
  relevant and irrelevant from run to run;
* the answer is read with ``startswith("yes")``, so ``**Yes**`` counts as
  irrelevant.

Here the verdict is a Pydantic model. The provider generates it against the
model's JSON schema in strict mode, greedily; the response is validated; and a
response that does not validate is retried with the error fed back, after the
structured-output clients of orqa: the schema sits in a static system message,
the user message is rebuilt on every retry rather than appended to, and a
failed output is quoted back only as a bounded excerpt. A document that never
yields a valid verdict fails the query, because a guessed verdict would move it
silently.

Neither Pneuma nor a provider SDK is imported: the server supplies a transport,
so the protocol is testable without either.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import time
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Greedy, as the paper's local judge pipeline decodes: Pneuma calls it with
# sampling off and temperature, top_p and top_k unset.
JUDGE_TEMPERATURE = 0.0
# Pneuma's own judge seed, kept for providers that honour one.
JUDGE_SEED = 42
# Room for the JSON verdict plus a reasoning model's hidden reasoning. Pneuma's
# two tokens only ever fitted a bare "Yes".
JUDGE_MAX_TOKENS = 1024
SCHEMA_NAME = "relevance_judgment"

# How much of an unparsable response is quoted back in the retry message.
_FAILED_OUTPUT_EXCERPT_CHARS = 600
# Pneuma's relevance prompts close with this free-text answer instruction.
_FREE_TEXT_INSTRUCTION = "Begin your answer with yes/no."
_JSON_INSTRUCTION = "Answer with the JSON verdict described in the system message."


class RelevanceJudgment(BaseModel):
    """The judge's verdict on one document-question pair.

    ``relevant`` comes first: the paper's judge commits to a bare yes or no, so
    the reason is feedback recorded after that commitment, not reasoning the
    verdict is derived from.
    """

    model_config = ConfigDict(extra="forbid")

    relevant: bool = Field(
        ...,
        description=(
            "Whether the table the document describes is relevant to answer "
            "the question."
        ),
    )
    reason: str = Field(
        ...,
        description="One sentence: what in the document supports or rules out that verdict.",
    )

    @field_validator("reason")
    @classmethod
    def reason_is_stated(cls, value: str) -> str:
        reason = " ".join(value.split())
        if not reason:
            raise ValueError("state what in the document supports or rules out the verdict")
        return reason


class JudgeError(RuntimeError):
    """A document-question pair produced no valid verdict."""


class ChatTransport(Protocol):
    """Sends one schema-constrained chat request and returns the response text."""

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
        max_tokens: int,
    ) -> str: ...


_SCHEMA = RelevanceJudgment.model_json_schema()
# Static for every document, so providers can serve it from the prompt cache.
_SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Return the verdict strictly as JSON exactly matching this schema, with "
        "no text outside the JSON:\n" + json.dumps(_SCHEMA, indent=2)
    ),
}

Node = TypeVar("Node")


def structured_relevance_prompt(pneuma_prompt: str) -> str:
    """Pneuma's relevance prompt with its free-text answer instruction made JSON."""
    if _FREE_TEXT_INSTRUCTION in pneuma_prompt:
        return pneuma_prompt.replace(_FREE_TEXT_INSTRUCTION, _JSON_INSTRUCTION)
    return f"{pneuma_prompt.rstrip()}\n{_JSON_INSTRUCTION}"


def order_by_relevance(
    nodes: Sequence[Node], judgments: Sequence[RelevanceJudgment]
) -> list[Node]:
    """Relevant documents first, then irrelevant ones, each in fused order (§5.1)."""
    if len(nodes) != len(judgments):
        raise ValueError(f"{len(judgments)} verdicts for {len(nodes)} documents")
    pairs = list(zip(nodes, judgments))
    return [node for node, judgment in pairs if judgment.relevant] + [
        node for node, judgment in pairs if not judgment.relevant
    ]


def _json_object(content: str) -> str:
    """The outermost JSON object in ``content``, without fences or prose around it."""
    start, end = content.find("{"), content.rfind("}")
    return content[start : end + 1] if 0 <= start < end else content.strip()


def _json_error_feedback(content: str, error: json.JSONDecodeError) -> str:
    output = content.strip()
    excerpt = output[:_FAILED_OUTPUT_EXCERPT_CHARS]
    truncated = "\n…(truncated)" if len(output) > _FAILED_OUTPUT_EXCERPT_CHARS else ""
    quoted = f"\nYour previous output started:\n{excerpt}{truncated}\n" if excerpt else ""
    return (
        "JSON PARSING ERROR: your previous response was not a valid JSON object.\n"
        f"Parser error: {error}\n{quoted}"
        "Return ONLY the JSON verdict matching the schema in the system message, "
        "with no text before or after it."
    )


def _validation_error_feedback(error: ValidationError) -> str:
    lines = [
        f"  - {' -> '.join(str(part) for part in item['loc']) or '(object)'}: "
        f"{item['msg']} ({item['type']})"
        for item in error.errors()
    ]
    return (
        "SCHEMA VALIDATION ERROR: your JSON did not match the schema on these fields:\n"
        + "\n".join(lines)
        + "\nFix exactly these fields and return the full JSON verdict again."
    )


@dataclass
class StructuredRelevanceJudge:
    """Schema-constrained, greedily decoded judge with retry-with-feedback."""

    transport: ChatTransport
    max_attempts: int = 3
    workers: int = 8
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

    def judge(self, prompt: str) -> RelevanceJudgment:
        """One verdict, retried with the error fed back until it validates."""
        user_content = prompt
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                # Rebuilt on every attempt: the static system message plus the
                # prompt, and at most the latest error, never a growing history.
                content = self.transport(
                    [_SYSTEM_MESSAGE, {"role": "user", "content": user_content}],
                    schema_name=SCHEMA_NAME,
                    schema=_SCHEMA,
                    temperature=JUDGE_TEMPERATURE,
                    seed=JUDGE_SEED,
                    max_tokens=JUDGE_MAX_TOKENS,
                )
                if not content or not content.strip():
                    raise ValueError("the judge returned empty content")
                try:
                    return RelevanceJudgment.model_validate(
                        json.loads(_json_object(content))
                    )
                except json.JSONDecodeError as error:
                    last_error = error
                    user_content = f"{prompt}\n\n{_json_error_feedback(content, error)}"
                except ValidationError as error:
                    last_error = error
                    user_content = f"{prompt}\n\n{_validation_error_feedback(error)}"
            except Exception as error:  # the request itself failed, or came back empty
                last_error = error
            if attempt < self.max_attempts:
                time.sleep(self.retry_delay)
        raise JudgeError(
            f"no valid verdict after {self.max_attempts} attempt(s): {last_error}"
        )

    def judge_all(self, prompts: Sequence[str]) -> list[RelevanceJudgment]:
        """Verdicts for ``prompts`` in their order, judged concurrently.

        Pairs are independent, so the verdicts are the ones a sequential judge
        would give; only the wall time changes. The first pair without a valid
        verdict raises, pending pairs are cancelled, and the query fails.
        """
        if not prompts:
            return []
        pool = ThreadPoolExecutor(max_workers=min(self.workers, len(prompts)))
        try:
            return list(pool.map(self.judge, prompts))
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
