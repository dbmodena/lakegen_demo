from types import SimpleNamespace

from lakegen.column_lineage import DerivedColumn, PlanStep
from lakegen.plan_judge import judge_plan_question_fidelity


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


def test_approves_a_faithful_plan():
    steps = [PlanStep(order=1, op="execute", tables=["t.parquet"], reads=["Postcode", "Amount"])]
    judgment, tokens = judge_plan_question_fidelity(
        question="Total amount by postcode?",
        steps=steps,
        selected_tables=["t.parquet"],
        selected_metadata={},
        real_columns={"t.parquet": ["Postcode", "Amount"]},
        llm=FakeLlm(
            '{"approves_question_fidelity": true, "confidence": 0.9, '
            '"rationale": "matches", "missing_requirements": [], "unjustified_steps": []}'
        ),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["approves_question_fidelity"] is True
    assert judgment["confidence"] == 0.9
    assert judgment["judge_error"] == ""
    assert tokens == 42


def test_rejects_with_missing_requirements():
    steps = [PlanStep(order=1, op="execute", tables=["t.parquet"], reads=["Postcode"])]
    judgment, _tokens = judge_plan_question_fidelity(
        question="Total amount written off, grouped by postcode?",
        steps=steps,
        selected_tables=["t.parquet"],
        selected_metadata={},
        real_columns={"t.parquet": ["Postcode", "Amount written-off"]},
        llm=FakeLlm(
            '{"approves_question_fidelity": false, "confidence": 0.8, '
            '"rationale": "amount not read", '
            '"missing_requirements": ["Amount written-off"], "unjustified_steps": []}'
        ),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["approves_question_fidelity"] is False
    assert judgment["missing_requirements"] == ["Amount written-off"]


def test_falls_back_to_code_and_trace_when_no_structured_steps():
    """The empty-plan-artifact-noise fix: when no steps exist, the prompt
    must show the actual code/trace instead of an "(empty plan)" framing."""
    prompt_manager = FakePromptManager()
    judge_plan_question_fidelity(
        question="How many rows?",
        steps=[],
        selected_tables=["t.parquet"],
        selected_metadata={},
        real_columns={"t.parquet": ["A"]},
        generated_code="print(len(df))",
        operation_trace={"used_tables": ["t.parquet"], "used_columns": ["A"]},
        llm=FakeLlm('{"approves_question_fidelity": true, "confidence": 1.0, '
                    '"rationale": "ok", "missing_requirements": [], "unjustified_steps": []}'),
        prompt_manager=prompt_manager,
    )
    plan_section = prompt_manager.kwargs["plan_section"]
    assert "(empty plan)" not in plan_section
    assert "print(len(df))" in plan_section
    assert "used_columns" in plan_section


def test_does_not_hallucinate_a_missing_column_when_schema_has_it():
    """Regression guard for the caught bug where a judge falsely claimed a
    real column was missing. The real schema is passed explicitly; a
    correctly-behaving judge (simulated here) approves rather than
    inventing a missing-column complaint."""
    steps = [PlanStep(order=1, op="execute", tables=["t.parquet"], reads=["lat", "lon"])]
    judgment, _tokens = judge_plan_question_fidelity(
        question="Average latitude?",
        steps=steps,
        selected_tables=["t.parquet"],
        selected_metadata={},
        real_columns={"t.parquet": ["lat", "lon"]},
        llm=FakeLlm(
            '{"approves_question_fidelity": true, "confidence": 0.95, '
            '"rationale": "lat column present and used", '
            '"missing_requirements": [], "unjustified_steps": []}'
        ),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["approves_question_fidelity"] is True
    assert judgment["missing_requirements"] == []


def test_malformed_json_is_repaired_on_second_attempt():
    judgment, _tokens = judge_plan_question_fidelity(
        question="How many rows?",
        steps=[],
        selected_tables=["t.parquet"],
        selected_metadata={},
        real_columns={},
        generated_code="print(1)",
        operation_trace={},
        llm=FakeLlm([
            "not json at all",
            '{"approves_question_fidelity": true, "confidence": 0.5, '
            '"rationale": "repaired", "missing_requirements": [], "unjustified_steps": []}',
        ]),
        prompt_manager=FakePromptManager(),
    )
    assert judgment["approves_question_fidelity"] is True
    assert judgment["rationale"] == "repaired"


def test_fails_open_on_judge_infrastructure_error():
    class BrokenLlm:
        token_usage_total = 0

        def chat(self, _messages):
            raise RuntimeError("connection reset")

    judgment, tokens = judge_plan_question_fidelity(
        question="q", steps=[], selected_tables=[], selected_metadata={},
        real_columns={}, llm=BrokenLlm(), prompt_manager=FakePromptManager(),
    )
    assert judgment["approves_question_fidelity"] is True  # fail-open
    assert "connection reset" in judgment["judge_error"]
    assert tokens == 0
