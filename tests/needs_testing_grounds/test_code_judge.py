from types import SimpleNamespace

from lakegen.code_judge import judge_production_code_result
from lakegen.value_grounding import GroundingViolation


class FakePromptManager:
    def __init__(self):
        self.kwargs = {}

    def render(self, *_args, **kwargs):
        self.kwargs = kwargs
        return f"Question: {kwargs['question']}"


class FakeLlm:
    def __init__(self, contents, token_usage_total=42):
        self._contents = list(contents) if isinstance(contents, (list, tuple)) else [contents]
        self.token_usage_total = token_usage_total

    def chat(self, _messages):
        content = self._contents.pop(0) if len(self._contents) > 1 else self._contents[0]
        message = SimpleNamespace(content=content, additional_kwargs={})
        return SimpleNamespace(message=message, raw={})


def test_approves_a_correct_result():
    judgment, tokens = judge_production_code_result(
        question="Total amount written off?",
        generated_code="df['Amount written-off'].sum()",
        generated_result=[{"total": 1234.5}],
        real_columns={"t.parquet": ["Amount written-off"]},
        llm=FakeLlm(
            '{"plan_compliance_approval": true, "present_result_approval": true, '
            '"violated_criteria": [], "feedback": "correct", "suggestions": []}'
        ),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["approved"] is True
    assert judgment["plan_compliance_approval"] is True
    assert judgment["present_result_approval"] is True
    assert tokens == 42


def test_rejects_and_reports_violated_criteria():
    judgment, _tokens = judge_production_code_result(
        question="Total amount written off in 2016/17?",
        generated_code="df[df['Financial Year'] == 2016]['Amount written-off'].sum()",
        generated_result=[{"total": 0.0}],
        grounding_violations=[
            GroundingViolation("Financial Year", "Eq", 2016, "HIGH",
                                "'Financial Year' holds fiscal-year-span codes like '201617'"),
        ],
        real_columns={"t.parquet": ["Financial Year", "Amount written-off"]},
        llm=FakeLlm(
            '{"plan_compliance_approval": false, "present_result_approval": false, '
            '"violated_criteria": ["silent_filter_bias"], '
            '"feedback": "wrong fiscal year code", '
            '"suggestions": ["[FIX QUERY] use 201617 not 2016"]}'
        ),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["approved"] is False
    assert judgment["violated_criteria"] == ["silent_filter_bias"]
    assert judgment["suggestions"] == ["[FIX QUERY] use 201617 not 2016"]


def test_grounding_violations_are_passed_into_the_prompt():
    prompt_manager = FakePromptManager()
    judge_production_code_result(
        question="q",
        generated_code="code",
        generated_result=0,
        grounding_violations=[
            GroundingViolation("col", "Eq", "x", "HIGH", "msg"),
        ],
        real_columns={},
        llm=FakeLlm('{"plan_compliance_approval": true, "present_result_approval": true, '
                    '"violated_criteria": [], "feedback": "", "suggestions": []}'),
        prompt_manager=prompt_manager,
    )
    assert "col='x' (HIGH)" in prompt_manager.kwargs["grounding_note"]


def test_no_grounding_violations_produces_empty_note():
    prompt_manager = FakePromptManager()
    judge_production_code_result(
        question="q", generated_code="code", generated_result=0,
        grounding_violations=[], real_columns={},
        llm=FakeLlm('{"plan_compliance_approval": true, "present_result_approval": true, '
                    '"violated_criteria": [], "feedback": "", "suggestions": []}'),
        prompt_manager=prompt_manager,
    )
    assert prompt_manager.kwargs["grounding_note"] == ""


def test_does_not_hallucinate_a_missing_column_when_schema_has_it():
    judgment, _tokens = judge_production_code_result(
        question="Average latitude?",
        generated_code="df['lat'].mean()",
        generated_result=[{"average_latitude": 51.5}],
        real_columns={"t.parquet": ["lat", "lon"]},
        llm=FakeLlm(
            '{"plan_compliance_approval": true, "present_result_approval": true, '
            '"violated_criteria": [], "feedback": "lat column present and used", '
            '"suggestions": []}'
        ),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["approved"] is True


def test_malformed_json_is_repaired_on_second_attempt():
    judgment, _tokens = judge_production_code_result(
        question="q", generated_code="code", generated_result=0,
        real_columns={},
        llm=FakeLlm([
            "not json",
            '{"plan_compliance_approval": true, "present_result_approval": true, '
            '"violated_criteria": [], "feedback": "repaired", "suggestions": []}',
        ]),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["feedback"] == "repaired"
    assert judgment["approved"] is True


def test_fails_open_on_judge_infrastructure_error():
    class BrokenLlm:
        token_usage_total = 0

        def chat(self, _messages):
            raise RuntimeError("timeout")

    judgment, tokens = judge_production_code_result(
        question="q", generated_code="code", generated_result=0,
        real_columns={}, llm=BrokenLlm(), prompt_manager=FakePromptManager(),
    )
    assert judgment["approved"] is True  # fail-open
    assert judgment["plan_compliance_approval"] is True
    assert judgment["present_result_approval"] is True
    assert "timeout" in judgment["judge_error"]
    assert tokens == 0
