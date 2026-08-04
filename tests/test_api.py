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
