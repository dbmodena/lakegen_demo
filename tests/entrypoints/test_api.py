"""LakeGen's API: one question, with its configuration."""

import pytest
from pydantic import ValidationError

import lakegen.entrypoints.api.app as api


def test_query_request_rejects_removed_ollama_url():
    try:
        api.QueryRequest(question="One?", ollama_url="http://localhost:11434")
    except ValidationError:
        pass
    else:
        raise AssertionError("ollama_url must not remain part of the OCI API")


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../experiment.yaml", "config/experiment.txt"],
)
def test_query_request_rejects_all_filesystem_config_paths(path):
    with pytest.raises(ValidationError):
        api.QueryRequest(question="One?", config_path=path)


def test_single_query_passes_the_complete_request_to_csv_logging(monkeypatch):
    captured = {}

    class Result:
        def to_dict(self):
            return {"status": "completed"}

    def fake_run_question(question, runtime, *, question_id, log_context):
        captured.update(
            question=question,
            runtime=runtime,
            question_id=question_id,
            log_context=log_context,
        )
        return Result()

    runtime = object()
    monkeypatch.setattr(api, "make_runtime_settings", lambda **kwargs: runtime)
    monkeypatch.setattr(api, "bootstrap_nltk_data", lambda: None)
    monkeypatch.setattr(api, "run_question", fake_run_question)

    response = api.query_lakegen(
        api.QueryRequest(
            question="Where are the parks?",
            retrieval_mode="hybrid",
            top_k=25,
        )
    )

    assert response == {"status": "completed"}
    assert captured["question"] == "Where are the parks?"
    assert captured["runtime"] is runtime
    assert captured["question_id"] is None
    assert captured["log_context"]["SOURCE_PATH"] == "$.question"
    assert captured["log_context"]["SOURCE_RETRIEVAL_MODE"] == "hybrid"
    assert captured["log_context"]["SOURCE_TOP_K"] == 25
