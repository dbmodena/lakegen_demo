import src.api as api
import csv
import json
import pytest
from fastapi import HTTPException
from pydantic import ValidationError


class FakeUpload:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content
        self._position = 0
        self.closed = False

    async def read(self, size: int) -> bytes:
        chunk = self._content[self._position:self._position + size]
        self._position += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


def test_batch_source_key_uses_path_when_source_id_is_missing():
    first = {"source_id": None, "source_path": "$.questions[0]"}
    second = {"source_id": None, "source_path": "$.questions[1]"}
    assert api._batch_source_key(first) != api._batch_source_key(second)


def test_benchmark_catalog_queues_its_complete_case_metadata(tmp_path, monkeypatch):
    benchmark = tmp_path / "sample.json"
    benchmark.write_text(
        json.dumps({
            "cases": [{
                "id": "case-1",
                "question": "Which table contains parks?",
                "keywords": ["parks"],
                "relevant_table_ids": ["parks-123"],
                "reference_code": "SELECT 1",
            }]
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "BENCHMARK_DIR", tmp_path)
    monkeypatch.setattr(api, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(api._executor, "submit", lambda *_args: None)

    assert api.list_benchmarks() == {"benchmarks": ["sample.json"]}
    accepted = api.submit_benchmark_batch("sample.json", core="nyc")
    job = api._jobs[accepted.job_id]

    assert accepted.question_count == 1
    assert job["settings"]["resolved_config"]["experiment_id"] == "benchmark-sample"
    assert job["settings"]["resolved_config"]["core"] == "nyc"
    stored = api._load_questions(accepted.job_id)[0]
    assert stored["log_fields"]["SOURCE_RELEVANT_TABLE_IDS"] == ["parks-123"]
    assert stored["log_fields"]["SOURCE_REFERENCE_CODE"] == "SELECT 1"
    api._jobs.pop(accepted.job_id, None)


def test_benchmark_catalog_rejects_paths_outside_its_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BENCHMARK_DIR", tmp_path)

    with pytest.raises(HTTPException) as raised:
        api._load_benchmark("../outside.json")
    assert raised.value.status_code == 404


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


def test_batch_question_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOB_DIR", tmp_path)
    job_id = "b" * 32
    questions = [{"question": "One?", "source_id": "q1"}]

    api._persist_questions(job_id, questions)

    assert api._load_questions(job_id) == questions


def test_run_batch_resumes_after_last_persisted_result(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "JOB_DIR", tmp_path)
    monkeypatch.setattr(api, "bootstrap_nltk_data", lambda: None)
    monkeypatch.setattr(api, "ExperimentConfig", type("Config", (), {
        "model_validate": staticmethod(lambda _settings: object())
    }))
    calls = []

    class FakeResult:
        def to_dict(self):
            return {"status": "completed", "error": "", "tables": []}

    class FakeRunner:
        def __init__(self, _config):
            pass

        def run(self, question, **kwargs):
            calls.append((question, kwargs["log_context"]))
            return FakeResult()

    monkeypatch.setattr(api, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(api, "_append_batch_table_metrics", lambda *_args: False)
    job_id = "c" * 32
    first = {
        "question": "One?",
        "source_path": "$.cases[0].question",
        "source_id": "q1",
        "result": {"status": "completed", "error": "", "tables": []},
    }
    questions = [
        {"question": "One?", "source_path": "$.cases[0].question", "source_id": "q1", "log_fields": {}},
        {"question": "Two?", "source_path": "$.cases[1].question", "source_id": "q2", "log_fields": {}},
    ]
    api._jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "started_at": "start",
        "updated_at": "start",
        "processed": 1,
        "failed": 0,
        "metrics_logged": False,
        "results": [first],
    }

    api._run_batch(job_id, questions, {"resolved_config": {}})

    assert calls == [(
        "Two?",
        {"JOB_ID": job_id, "EXECUTION_ATTEMPT": 1, "IS_FINAL_ATTEMPT": True},
    )]
    assert api._jobs[job_id]["processed"] == 2
    assert len(api._jobs[job_id]["results"]) == 2
    assert api._jobs[job_id]["status"] == "completed"
    assert "code" in api._jobs[job_id]["batch_metrics"]
    api._jobs.pop(job_id, None)


def test_batch_table_metrics_returns_summary_for_final_job_file(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BASE_DIR", tmp_path)
    monkeypatch.setattr(api, "append_benchmark_metrics_log", lambda *_args, **_kwargs: None)
    questions = [{
        "question": "How many parks?",
        "source_id": "case-1",
        "source_path": "$.cases[0].question",
        "log_fields": {"SOURCE_RELEVANT_TABLE_IDS": ["parks"]},
    }]
    results = [{"result": {
        "tables": ["parks.csv"],
        "error": "",
    }}]
    settings = {"resolved_config": {
        "experiment_id": "experiment",
        "core": "nyc",
        "model": "model",
        "discovery_architecture": "unified",
        "retrieval": {
            "mode": "keyword",
            "fusion_method": "weighted",
        },
    }}

    summary = api._append_batch_table_metrics(
        "d" * 32, questions, results, settings
    )

    assert summary["case_count"] == 1
    assert summary["mean_metrics"]["Hit@1"] == 1.0


def test_automatic_coder_sweep_writes_one_csv_row_per_shared_context_variant(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(api, "BASE_DIR", tmp_path)
    questions = [{
        "question": "How many parks?",
        "source_id": "case-1",
        "source_path": "$.cases[0].question",
        "log_fields": {"SOURCE_RELEVANT_TABLE_IDS": ["parks"]},
    }]
    variants = {}
    for level, correct in (("full", True), ("schema_only", True), ("minimal", False)):
        variants[level] = {"code_evaluation": {
            "applicable": True,
            "expected_result_type": "number",
            "generation_success": True,
            "execution_success": True,
            "structured_output_valid": True,
            "result_type_match": True,
            "exact_result_match": correct,
            "pass_at_1": correct,
            "success_within_3": correct,
            "attempt_count": 1,
        }}
    results = [{"result": {
        "tables": ["parks.csv"],
        "error": "",
        "coder_context_experiment": {"variants": variants},
    }}]
    retrieval = {
        "mode": "keyword", "fusion_method": "weighted", "alpha": 0.5,
        "rrf_k": 60, "top_k": 10, "candidate_multiplier": 5,
        "representation_version": "metadata-v1", "embedding_model": "bge-m3",
    }
    settings = {"resolved_config": {
        "experiment_id": "coder-sweep", "core": "nyc", "model": "model",
        "discovery_architecture": "unified", "automatic_test_coder": True,
        "coder_context_level": "full", "retrieval": retrieval,
    }}

    api._append_batch_table_metrics("e" * 32, questions, results, settings)

    with (tmp_path / "logs" / "retrieval_benchmarks_log.csv").open(
        newline="", encoding="utf-8"
    ) as input_file:
        rows = list(csv.DictReader(input_file))
    assert [row["CODER_CONTEXT_LEVEL"] for row in rows] == [
        "full", "schema_only", "minimal"
    ]
    assert [row["EXACT_RESULT_MATCH_RATE"] for row in rows] == [
        "1.0", "1.0", "0.0"
    ]
    assert len({row["JOB_ID"] for row in rows}) == 1
    assert all(row["RECALL_AT_1"] == "1.0" for row in rows)


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


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../experiment.yaml", "config/experiment.txt"],
)
def test_query_request_rejects_all_filesystem_config_paths(path):
    with pytest.raises(ValidationError):
        api.QueryRequest(question="One?", config_path=path)


def test_api_contract_exposes_only_inline_config():
    schema = api.app.openapi()
    properties = schema["components"]["schemas"]["QueryRequest"]["properties"]
    assert "config" in properties
    assert "config_path" not in properties
    multipart = schema["paths"]["/v1/batches/files"]["post"]["requestBody"]["content"]
    assert "multipart/form-data" in multipart


def test_batch_accepts_only_inline_top_level_config():
    config = api._inline_batch_config({
        "config": {"core": "bologna", "retrieval": {"mode": "hybrid"}},
        "questions": ["One?"],
    })
    assert config == {"core": "bologna", "retrieval": {"mode": "hybrid"}}
    resolved = api._resolve_api_config(
        core=api.DEFAULT_CORE,
        model=api.DEFAULT_MODEL,
        retrieval_mode="keyword",
        top_k=10,
        hybrid_alpha=0.5,
        candidate_multiplier=5,
        config_data=config,
        explicit_fields=set(),
    )
    assert resolved.core == "bologna"
    assert resolved.retrieval.mode == "hybrid"
    with pytest.raises(ValueError, match="inline JSON object"):
        api._inline_batch_config({"config": "/tmp/experiment.yaml", "questions": []})


@pytest.mark.asyncio
async def test_multipart_batch_accepts_config_and_questions_files(tmp_path, monkeypatch):
    submitted = {}
    monkeypatch.setattr(api, "JOB_DIR", tmp_path)
    monkeypatch.setattr(
        api._executor,
        "submit",
        lambda function, *args: submitted.update(function=function, args=args),
    )
    config_upload = FakeUpload(
        "experiment.yaml",
        b"experiment_id: uploaded-5\ncore: nyc\n"
        b"discovery_architecture: divided\n"
        b"retrieval:\n  mode: hybrid\n  top_k: 7\n",
    )
    questions_upload = FakeUpload(
        "questions.json", b'{"questions": ["One?", "Two?"]}'
    )
    response = await api.submit_batch_files(config_upload, questions_upload)

    assert response.question_count == 2
    job = api._jobs[response.job_id]
    resolved = job["settings"]["resolved_config"]
    assert resolved["experiment_id"] == "uploaded-5"
    assert resolved["discovery_architecture"] == "divided"
    assert resolved["retrieval"]["mode"] == "hybrid"
    assert resolved["retrieval"]["top_k"] == 7
    assert resolved["interaction_mode"] == "autonomous"
    assert submitted["function"] is api._run_batch
    assert config_upload.closed and questions_upload.closed
    api._jobs.pop(response.job_id, None)


@pytest.mark.parametrize(
    ("config_name", "questions_name", "expected"),
    [
        ("../experiment.yaml", "questions.json", "must not contain a path"),
        ("experiment.txt", "questions.json", "extensions"),
        ("experiment.yaml", "questions.yaml", "extensions"),
    ],
)
@pytest.mark.asyncio
async def test_multipart_batch_rejects_unsafe_or_unsupported_filenames(
    config_name, questions_name, expected
):
    config_upload = FakeUpload(config_name, b"core: nyc")
    questions_upload = FakeUpload(questions_name, b'{"questions": ["One?"]}')

    with pytest.raises(HTTPException) as raised:
        await api.submit_batch_files(config_upload, questions_upload)
    assert raised.value.status_code == 422
    assert expected in raised.value.detail
    assert config_upload.closed and questions_upload.closed


@pytest.mark.asyncio
async def test_multipart_batch_rejects_invalid_content_and_uploads_over_limit(
    monkeypatch,
):
    with pytest.raises(HTTPException) as invalid:
        await api.submit_batch_files(
            FakeUpload("experiment.yaml", b"gates: ["),
            FakeUpload("questions.json", b"not-json"),
        )
    assert invalid.value.status_code == 422

    monkeypatch.setattr(api, "MAX_CONFIG_UPLOAD_BYTES", 4)
    with pytest.raises(HTTPException) as too_large:
        await api.submit_batch_files(
            FakeUpload("experiment.yaml", b"core: nyc"),
            FakeUpload("questions.json", b'{"questions": ["One?"]}'),
        )
    assert too_large.value.status_code == 413


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


def test_dynamic_reference_replaces_declared_gold_and_records_drift(
    tmp_path, monkeypatch
):
    questions = [{
        "log_fields": {
            "SOURCE_ENGINE": "PANDAS",
            "SOURCE_REFERENCE_CODE": "result = 2",
            "SOURCE_TABLE_ALIASES": {"Table_0": "numbers"},
            "SOURCE_EXPECTED_RESULT_TYPE": "number",
            "SOURCE_REFERENCE_RESULT": [{"result": 1}],
        }
    }]
    monkeypatch.setattr(
        api,
        "execute_pandas_reference",
        lambda **_kwargs: {
            "status": "success", "result": 2, "cache_hit": False
        },
    )

    progress = []
    metrics = api._prepare_dynamic_references(
        questions,
        tables_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        progress_callback=progress.append,
    )

    fields = questions[0]["log_fields"]
    assert fields["SOURCE_DECLARED_REFERENCE_RESULT"] == [{"result": 1}]
    assert fields["SOURCE_REFERENCE_RESULT"] == 2
    assert fields["SOURCE_REFERENCE_EXECUTION"]["declared_result_drift"] is True
    assert metrics["execution_success_count"] == 1
    assert metrics["reference_drift_count"] == 1
    assert metrics["stable_reference_count"] == 0
    assert metrics["reference_drift_rate"] == 1.0
    assert metrics["processed_case_count"] == 1
    assert progress[0]["processed_case_count"] == 0
    assert progress[-1]["processed_case_count"] == 1


def test_prevalidated_gold_is_not_executed_again(tmp_path, monkeypatch):
    questions = [{
        "log_fields": {
            "SOURCE_ENGINE": "PANDAS",
            "SOURCE_REFERENCE_CODE": "result = 2",
            "SOURCE_TABLE_ALIASES": {"Table_0": "numbers"},
            "SOURCE_EXPECTED_RESULT_TYPE": "number",
            "SOURCE_REFERENCE_RESULT": 2,
            "SOURCE_DECLARED_REFERENCE_RESULT": 1,
            "SOURCE_GOLD_VALIDATION": {
                "status": "benchmark_ready",
                "deterministic": True,
                "validation_runs": 3,
            },
        }
    }]
    monkeypatch.setattr(
        api,
        "execute_pandas_reference",
        lambda **_kwargs: pytest.fail("prevalidated gold must not be re-executed"),
    )

    metrics = api._prepare_dynamic_references(
        questions,
        tables_dir=tmp_path,
        cache_dir=tmp_path / "cache",
    )

    execution = questions[0]["log_fields"]["SOURCE_REFERENCE_EXECUTION"]
    assert execution["prevalidated"] is True
    assert metrics["prevalidated_case_count"] == 1
    assert metrics["execution_success_count"] == 1
    assert metrics["reference_drift_count"] == 1


def test_metric_ready_batch_appends_table_selection_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BASE_DIR", tmp_path)
    questions = [
        {
            "question": "One?",
            "source_path": "$.cases[0].question",
            "source_id": "q1",
            "log_fields": {"SOURCE_RELEVANT_TABLE_IDS": ["gold-a", "gold-b"]},
        },
        {
            "question": "Two?",
            "source_path": "$.cases[1].question",
            "source_id": "q2",
            "log_fields": {"SOURCE_RELEVANT_TABLE_IDS": ["gold-c"]},
        },
    ]
    results = [
        {"result": {
            "tables": ["gold-a.parquet", "other.parquet", "gold-b.parquet"],
            "error": "",
            "code_evaluation": {
                "applicable": True, "generation_success": True,
                "execution_success": True, "structured_output_valid": True,
                "result_type_match": True, "exact_result_match": True,
                "pass_at_1": True, "success_within_3": True,
                "attempt_count": 1, "column_f1": 1.0, "row_f1": 1.0,
                "cell_accuracy": 1.0,
            },
        }},
        {"result": {
            "tables": ["other.parquet", "gold-c.parquet"], "error": "",
            "code_evaluation": {
                "applicable": True, "generation_success": True,
                "execution_success": False, "structured_output_valid": False,
                "result_type_match": False, "exact_result_match": False,
                "pass_at_1": False, "success_within_3": False,
                "attempt_count": 3, "error_category": "execution_error",
            },
        }},
    ]
    settings = {
        "resolved_config": {
            "experiment_id": "architecture-test",
            "core": "nyc",
            "coder_context_level": "full",
            "retrieval": {
                "mode": "keyword",
                "fusion_method": "weighted",
                "alpha": 0.5,
                "rrf_k": 60,
                "top_k": 10,
                "candidate_multiplier": 5,
                "representation_version": "metadata-v1",
                "embedding_model": "bge-m3",
            },
        }
    }

    assert api._append_batch_table_metrics("job-1", questions, results, settings)

    with (tmp_path / "logs" / "retrieval_benchmarks_log.csv").open(
        newline="", encoding="utf-8"
    ) as input_file:
        row = next(csv.DictReader(input_file))
    assert row["JOB_ID"] == "job-1"
    assert row["BENCHMARK_TYPE"] == "batch-table-selection"
    assert row["EXPERIMENT_ID"] == "architecture-test"
    assert row["RETRIEVAL_MODE"] == "keyword"
    assert row["HYBRID_ALPHA"] == "0.5"
    assert row["STATUS"] == "completed"
    assert row["QUESTION_COUNT"] == "2"
    assert row["RECALL_AT_1"] == "0.25"
    assert row["MRR"] == "0.75"
    assert row["CODER_CONTEXT_LEVEL"] == "full"
    assert row["CODE_EXECUTION_SUCCESS_RATE"] == "0.5"
    assert row["EXACT_RESULT_MATCH_RATE"] == "0.5"
    assert row["PASS_AT_1"] == "0.5"
    assert row["MEAN_CODE_ATTEMPTS"] == "2.0"
    assert json.loads(row["CODE_ERROR_CATEGORIES_JSON"]) == {
        "execution_error": 1
    }
    assert "CASE_METRICS_JSON" not in row


def test_batch_metrics_skip_legacy_questions_without_gold_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BASE_DIR", tmp_path)
    questions = [{
        "question": "Legacy?",
        "source_path": "$.questions[0]",
        "source_id": 0,
        "log_fields": {},
    }]

    assert not api._append_batch_table_metrics(
        "job-legacy", questions, [{"result": {"tables": ["table.parquet"]}}], {}
    )
    assert not (tmp_path / "logs" / "retrieval_benchmarks_log.csv").exists()


@pytest.mark.asyncio
async def test_multipart_batch_propagates_question_file_id(tmp_path, monkeypatch):
    submitted = {}
    monkeypatch.setattr(api, "JOB_DIR", tmp_path)
    monkeypatch.setattr(
        api._executor,
        "submit",
        lambda function, *args: submitted.update(function=function, args=args),
    )

    response = await api.submit_batch_files(
        FakeUpload("experiment.yaml", b"core: nyc\n"),
        FakeUpload(
            "questions.json",
            b'{"questions": [{"id": "file-q-7", "question": "Same?"}]}',
        ),
    )

    questions = submitted["args"][1]
    assert questions[0]["source_id"] == "file-q-7"
    assert questions[0]["log_fields"]["SOURCE_ID"] == "file-q-7"
    api._jobs.pop(response.job_id, None)
