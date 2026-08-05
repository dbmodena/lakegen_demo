from types import SimpleNamespace

from lakegen.output_validation import AnswerDisposition, validate_answer
from lakegen.retrieval import RetrievalConfig
from lakegen.service import extract_questions, run_question


def test_extracts_queries_old_shape_and_preserves_metadata():
    payload = {
        "model": {
            "SQL": {
                "0": {
                    "data": {
                        "queries": [
                            {"id": 7, "question": "First question?", "code": "SELECT 1"},
                            {"id": None, "question": "Second question?"},
                        ]
                    }
                }
            }
        }
    }

    questions = extract_questions(payload)

    assert [item.question for item in questions] == ["First question?", "Second question?"]
    assert questions[0].source_id == 7
    assert questions[0].path == "$.model.SQL['0'].data.queries[0].question"
    assert questions[0].source_data["code"] == "SELECT 1"
    assert questions[0].log_fields()["SOURCE_CODE"] == "SELECT 1"


def test_extracts_simple_question_lists():
    questions = extract_questions({"questions": [" One? ", {"question": "Two?"}, ""]})

    assert [item.question for item in questions] == ["One?", "Two?"]


def test_extracts_top_level_string_list():
    questions = extract_questions(["One?", " Two? "])

    assert [item.question for item in questions] == ["One?", "Two?"]


def test_does_not_treat_unrelated_strings_as_questions():
    assert extract_questions({"description": "not a question", "tables": ["users"]}) == []


def test_answer_validation_distinguishes_valid_rejected_and_empty():
    assert validate_answer("42 schools").disposition == AnswerDisposition.VALID
    assert validate_answer("  ").disposition == AnswerDisposition.EMPTY
    refusal = validate_answer(
        "The data provided does not contain information about bandwidth."
    )
    assert refusal.disposition == AnswerDisposition.REJECTED


def test_run_question_does_not_mark_synthesized_refusal_completed(monkeypatch, tmp_path):
    runtime = SimpleNamespace(
        model_name="fake",
        solr_core="nyc",
        csv_dir=tmp_path,
        portal_name="NYC",
        retrieval=RetrievalConfig(),
    )
    (tmp_path / "gold.csv").write_text("value\n42\n", encoding="utf-8")
    monkeypatch.setattr("lakegen.service.get_llm", lambda _name: (object(), None))
    monkeypatch.setattr("lakegen.service.get_solr", lambda _core: object())
    monkeypatch.setattr("lakegen.service.get_prompt_manager", object)
    monkeypatch.setattr("lakegen.service.get_all_table_files", lambda _path: ["gold.csv"])
    monkeypatch.setattr(
        "lakegen.service.phase12_agent",
        lambda **_kwargs: (["gold.csv"], ["gold"], {}, "correct table", "trace", 0),
    )
    generated = SimpleNamespace(
        tokens=0,
        clean_code="print(42)",
        code_raw="print(42)",
        rejected_reason="",
        error=None,
        raw_result="42 schools",
    )
    monkeypatch.setattr(
        "lakegen.service.phase3_generate_and_execute", lambda *_args, **_kwargs: generated
    )
    monkeypatch.setattr(
        "lakegen.service.phase4_synthesize",
        lambda *_args: ("The necessary data is not available.", 0),
    )
    monkeypatch.setattr("lakegen.service.save_experiment_log", lambda **_kwargs: None)
    monkeypatch.setattr("lakegen.service.log_retrieval_decision", lambda **_kwargs: None)

    result = run_question("How many schools?", runtime)

    assert result.status == "rejected"
    assert result.answer_disposition == "rejected"
    assert result.pipeline_stages["final_answer"] == "rejected"
