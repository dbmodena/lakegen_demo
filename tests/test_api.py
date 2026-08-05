import src.api as api
from pydantic import ValidationError


def test_job_persistence_separates_metadata_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOB_DIR", tmp_path)
    job_id = "a" * 32
    job = {"job_id": job_id, "status": "running", "results": []}

    api._persist_job(job)
    api._append_result(job_id, {"question": "One?", "result": {"status": "completed"}})

    with_results = api._snapshot_job(job_id)
    without_results = api._snapshot_job(job_id, include_results=False)
    assert with_results["results"][0]["question"] == "One?"
    assert "results" not in without_results
    assert "results" not in api._job_path(job_id).read_text(encoding="utf-8")


def test_invalid_job_id_cannot_be_used_as_a_path(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOB_DIR", tmp_path)

    assert api._snapshot_job("../not-a-job") is None


def test_query_request_rejects_removed_ollama_url():
    try:
        api.QueryRequest(question="One?", ollama_url="http://localhost:11434")
    except ValidationError:
        pass
    else:
        raise AssertionError("ollama_url must not remain part of the OCI API")


def test_single_query_passes_the_complete_request_to_csv_logging(monkeypatch):
    captured = {}

    class Result:
        def to_dict(self):
            return {"status": "completed"}

    def fake_run_question(question, runtime, *, log_context):
        captured.update(
            question=question,
            runtime=runtime,
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
    assert captured["log_context"]["SOURCE_PATH"] == "$.question"
    assert captured["log_context"]["SOURCE_RETRIEVAL_MODE"] == "hybrid"
    assert captured["log_context"]["SOURCE_TOP_K"] == 25
