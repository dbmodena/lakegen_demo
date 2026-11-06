import json
import time

import pytest

from pneuma_engine.judge import (
    JudgeError,
    RelevanceJudgment,
    StructuredRelevanceJudge,
    order_by_relevance,
    structured_relevance_prompt,
)


class ScriptedTransport:
    """A judge transport replaying canned responses and recording each request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, messages, **params):
        self.requests.append({"messages": [dict(message) for message in messages], **params})
        return self.responses.pop(0)


def test_structured_judge_decodes_greedily_against_the_verdict_schema():
    transport = ScriptedTransport('{"relevant": true, "reason": "It has FTE posts per region."}')

    judgment = StructuredRelevanceJudge(transport, retry_delay=0).judge("Is it relevant?")

    assert judgment == RelevanceJudgment(relevant=True, reason="It has FTE posts per region.")
    (request,) = transport.requests
    assert request["temperature"] == 0.0
    assert request["schema"] == RelevanceJudgment.model_json_schema()
    assert request["schema"]["additionalProperties"] is False
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert request["messages"][1]["content"] == "Is it relevant?"
    with pytest.raises(ValueError):
        RelevanceJudgment(relevant=True, reason="   ")


def test_structured_judge_feeds_errors_back_until_the_verdict_validates():
    transport = ScriptedTransport(
        "**Yes**, the table is relevant.",
        '{"relevant": "maybe", "reason": "Unsure."}',
        '```json\n{"relevant": false, "reason": "No Asylum rows."}\n```',
    )

    judgment = StructuredRelevanceJudge(transport, retry_delay=0).judge("Is it relevant?")

    assert judgment.relevant is False
    first, after_json_error, after_validation_error = (
        request["messages"] for request in transport.requests
    )
    # Rebuilt on every attempt, never appended to; the system message is fixed.
    assert len(after_json_error) == len(after_validation_error) == 2
    assert after_json_error[0] == after_validation_error[0] == first[0]
    assert "JSON PARSING ERROR" in after_json_error[1]["content"]
    assert "**Yes**, the table is relevant." in after_json_error[1]["content"]
    assert "SCHEMA VALIDATION ERROR" in after_validation_error[1]["content"]
    assert "  - relevant:" in after_validation_error[1]["content"]


def test_structured_judge_raises_rather_than_guessing_a_verdict():
    transport = ScriptedTransport("Yes", "Yes", "Yes")

    with pytest.raises(JudgeError, match="no valid verdict after 3 attempt"):
        StructuredRelevanceJudge(transport, retry_delay=0).judge("Is it relevant?")


def test_judge_all_returns_verdicts_in_document_order_across_workers():
    def transport(messages, **_params):
        prompt = messages[1]["content"]
        index = int(prompt[-1])
        time.sleep(0.01 * (5 - index))  # later documents answer first
        return json.dumps({"relevant": index % 2 == 0, "reason": prompt})

    judge = StructuredRelevanceJudge(transport, workers=4, retry_delay=0)
    judgments = judge.judge_all([f"document {index}" for index in range(5)])

    assert [judgment.reason for judgment in judgments] == [
        f"document {index}" for index in range(5)
    ]
    assert [judgment.relevant for judgment in judgments] == [True, False, True, False, True]


def test_order_by_relevance_moves_irrelevant_documents_behind_in_fused_order():
    def verdict(relevant):
        return RelevanceJudgment(relevant=relevant, reason="r")

    nodes = ["a", "b", "c", "d"]
    verdicts = [verdict(False), verdict(True), verdict(False), verdict(True)]

    assert order_by_relevance(nodes, verdicts) == ["b", "d", "a", "c"]
    with pytest.raises(ValueError):
        order_by_relevance(nodes, verdicts[:1])


def test_structured_relevance_prompt_changes_only_the_answer_instruction():
    pneuma_prompt = (
        "and this question:\n/*\nWhich?\n*/\n"
        "Is the table relevant to answer the question? Begin your answer with yes/no."
    )

    prompt = structured_relevance_prompt(pneuma_prompt)

    assert prompt == (
        "and this question:\n/*\nWhich?\n*/\n"
        "Is the table relevant to answer the question? "
        "Answer with the JSON verdict described in the system message."
    )
