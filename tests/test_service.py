from lakegen.service import extract_questions


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
