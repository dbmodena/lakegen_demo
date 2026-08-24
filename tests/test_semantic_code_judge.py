from types import SimpleNamespace

from lakegen.semantic_code_judge import judge_semantic_code_result


class FakePromptManager:
    def render(self, *_args, **kwargs):
        return f"Question: {kwargs['question']}"


class FakeLlm:
    def __init__(self, content):
        self.content = content
        self.token_usage_total = 0

    def chat(self, _messages):
        message = SimpleNamespace(content=self.content, additional_kwargs={})
        return SimpleNamespace(message=message, raw={"usage": {"total_tokens": 17}})


def test_semantic_judge_accepts_supported_alternative():
    judgment, tokens = judge_semantic_code_result(
        question="How many records?",
        expected_description="A count",
        reference_result=10,
        selected_tables=["newer.parquet"],
        selected_metadata={"newer.parquet": {"description": "Updated data"}},
        generated_code="print(12)",
        generated_result=12,
        llm=FakeLlm(
            '{"disposition":"alternative_correct","confidence":0.9,'
            '"rationale":"Newer source","requirements_met":["count"],'
            '"requirements_missing":[]}'
        ),
        prompt_manager=FakePromptManager(),
    )

    assert judgment["disposition"] == "alternative_correct"
    assert judgment["confidence"] == 0.9
    assert tokens == 17


def test_semantic_judge_fails_closed_on_invalid_response():
    judgment, tokens = judge_semantic_code_result(
        question="Question",
        expected_description="Expected",
        reference_result=[],
        selected_tables=[],
        selected_metadata={},
        generated_code="print([])",
        generated_result=[],
        llm=FakeLlm("not json"),
        prompt_manager=FakePromptManager(),
    )

    assert judgment["disposition"] == "indeterminate"
    assert judgment["judge_error"].startswith("JSONDecodeError:")
    assert tokens == 0
