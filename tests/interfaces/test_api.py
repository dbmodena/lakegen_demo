import lakegen.interfaces.api.app as api
from lakegen.experiments import batch, jobs
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
    assert jobs.question_key(first) != jobs.question_key(second)


def test_positive_int_from_env_uses_default_for_invalid_values(monkeypatch):
    monkeypatch.setenv("TEST_LAKEGEN_WORKERS", "0")
    assert batch.positive_int_from_env("TEST_LAKEGEN_WORKERS", 2) == 2
    monkeypatch.setenv("TEST_LAKEGEN_WORKERS", "invalid")
    assert batch.positive_int_from_env("TEST_LAKEGEN_WORKERS", 2) == 2
    monkeypatch.setenv("TEST_LAKEGEN_WORKERS", "3")
    assert batch.positive_int_from_env("TEST_LAKEGEN_WORKERS", 2) == 3


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
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path / "jobs")
    monkeypatch.setattr(batch.EXECUTOR, "submit", lambda *_args: None)

    assert api.list_benchmarks() == {"benchmarks": ["sample.json"]}
    accepted = api.submit_benchmark_batch("sample.json", core="nyc")
    job = jobs.JOBS[accepted.job_id]

    assert accepted.question_count == 1
    assert job["settings"]["resolved_config"]["experiment_id"] == "benchmark-sample"
    assert job["settings"]["resolved_config"]["core"] == "nyc"
    stored = jobs.load_questions(accepted.job_id)[0]
    assert stored["log_fields"]["SOURCE_RELEVANT_TABLE_IDS"] == ["parks-123"]
    assert stored["log_fields"]["SOURCE_REFERENCE_CODE"] == "SELECT 1"
    jobs.JOBS.pop(accepted.job_id, None)


def test_benchmark_catalog_rejects_paths_outside_its_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "BENCHMARK_DIR", tmp_path)

    with pytest.raises(HTTPException) as raised:
        api._load_benchmark("../outside.json")
    assert raised.value.status_code == 404


def test_job_persistence_separates_metadata_and_results(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    job_id = "a" * 32
    job = {"job_id": job_id, "status": "running", "results": []}

    jobs.persist_job(job)
    jobs.append_result(job_id, {"question": "One?", "result": {"status": "completed"}})

    with_results = jobs.snapshot_job(job_id)
    without_results = jobs.snapshot_job(job_id, include_results=False)
    assert with_results["results"][0]["question"] == "One?"
    assert "results" not in without_results
    assert "results" not in jobs.job_path(job_id).read_text(encoding="utf-8")


def test_batch_question_checkpoint_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    job_id = "b" * 32
    questions = [{"question": "One?", "source_id": "q1"}]

    jobs.persist_questions(job_id, questions)

    assert jobs.load_questions(job_id) == questions


def test_run_batch_resumes_after_last_persisted_result(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(batch, "bootstrap_nltk_data", lambda: None)
    monkeypatch.setattr(batch, "ExperimentConfig", type("Config", (), {
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

    monkeypatch.setattr(batch, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(batch, "append_batch_table_metrics", lambda *_args: False)
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
    jobs.JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "started_at": "start",
        "updated_at": "start",
        "processed": 1,
        "failed": 0,
        "metrics_logged": False,
        "results": [first],
    }

    batch.run_batch(job_id, questions, {"resolved_config": {}})

    assert calls == [(
        "Two?",
        {"JOB_ID": job_id, "EXECUTION_ATTEMPT": 1, "IS_FINAL_ATTEMPT": True},
    )]
    assert jobs.JOBS[job_id]["processed"] == 2
    assert len(jobs.JOBS[job_id]["results"]) == 2
    assert jobs.JOBS[job_id]["status"] == "completed"
    assert "code" in jobs.JOBS[job_id]["batch_metrics"]
    jobs.JOBS.pop(job_id, None)


def test_batch_table_metrics_returns_summary_for_final_job_file(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(batch, "append_benchmark_metrics_log", lambda *_args, **_kwargs: None)
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

    summary = batch.append_batch_table_metrics(
        "d" * 32, questions, results, settings
    )

    assert summary["case_count"] == 1
    assert summary["mean_metrics"]["Hit@1"] == 1.0


def test_automatic_coder_sweep_writes_one_csv_row_per_shared_context_variant(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(batch, "LOG_DIR", tmp_path / "logs")
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

    batch.append_batch_table_metrics("e" * 32, questions, results, settings)

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
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)

    assert jobs.snapshot_job("../not-a-job") is None


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
    config = batch.inline_config({
        "config": {"core": "bologna", "retrieval": {"mode": "hybrid"}},
        "questions": ["One?"],
    })
    assert config == {"core": "bologna", "retrieval": {"mode": "hybrid"}}
    resolved = batch.resolve_config(
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
        batch.inline_config({"config": "/tmp/experiment.yaml", "questions": []})


@pytest.mark.asyncio
async def test_multipart_batch_accepts_config_and_questions_files(tmp_path, monkeypatch):
    submitted = {}
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(
        batch.EXECUTOR,
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
    job = jobs.JOBS[response.job_id]
    resolved = job["settings"]["resolved_config"]
    assert resolved["experiment_id"] == "uploaded-5"
    assert resolved["discovery_architecture"] == "divided"
    assert resolved["retrieval"]["mode"] == "hybrid"
    assert resolved["retrieval"]["top_k"] == 7
    assert resolved["interaction_mode"] == "autonomous"
    assert submitted["function"] is batch.run_batch
    assert config_upload.closed and questions_upload.closed
    jobs.JOBS.pop(response.job_id, None)


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
    monkeypatch.setattr(batch, "bootstrap_nltk_data", lambda: None)
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


def test_metric_ready_batch_appends_table_selection_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "LOG_DIR", tmp_path / "logs")
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

    assert batch.append_batch_table_metrics("job-1", questions, results, settings)

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
    monkeypatch.setattr(batch, "LOG_DIR", tmp_path / "logs")
    questions = [{
        "question": "Legacy?",
        "source_path": "$.questions[0]",
        "source_id": 0,
        "log_fields": {},
    }]

    assert not batch.append_batch_table_metrics(
        "job-legacy", questions, [{"result": {"tables": ["table.parquet"]}}], {}
    )
    assert not (tmp_path / "logs" / "retrieval_benchmarks_log.csv").exists()


def test_augmented_metrics_report_alternative_selection_without_changing_strict_metrics():
    questions = [{
        "question": "Alternative?",
        "source_path": "$.cases[0].question",
        "source_id": "q-alt",
        "log_fields": {
            "SOURCE_RELEVANT_TABLE_IDS": ["family___gold"],
            "SOURCE_ACCEPTED_TABLE_ALTERNATIVES": {
                "family___gold": ["alternative"],
            },
        },
    }]
    results = [{"result": {
        "tables": ["alternative.parquet"],
        "ranking": [{
            "attempt": 1, "rank": 1, "resource_id": "alternative",
        }],
        "error": "",
    }}]
    settings = {"resolved_config": {
        "experiment_id": "alternatives",
        "core": "uk",
        "coder_context_level": "full",
        "automatic_test_coder": False,
        "retrieval": {"mode": "hybrid", "fusion_method": "weighted"},
    }}

    metrics = batch.append_batch_table_metrics(
        "job-alt", questions, results, settings, append_log=False
    )

    assert metrics["mean_selection_metrics"]["SelectionHit"] == 0.0
    augmented = metrics["augmented_retrieval"]
    assert augmented["eligible_case_count"] == 1
    assert augmented["mean_retrieval_metrics"]["AlternativeHit@1"] == 1.0
    assert augmented["mean_selection_metrics"]["AlternativeSelectionHit"] == 1.0
    assert augmented["mean_selection_metrics"]["StrictMissRescued"] == 1.0


@pytest.mark.asyncio
async def test_multipart_batch_propagates_question_file_id(tmp_path, monkeypatch):
    submitted = {}
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(
        batch.EXECUTOR,
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
    jobs.JOBS.pop(response.job_id, None)


def test_batch_pause_rests_after_every_n_questions_but_not_after_the_last(monkeypatch):
    assert batch.batch_pause_seconds(5, 10, every=5, seconds=30) == 30.0
    assert batch.batch_pause_seconds(10, 1, every=5, seconds=30) == 30.0
    assert batch.batch_pause_seconds(4, 10, every=5, seconds=30) == 0.0
    assert batch.batch_pause_seconds(5, 0, every=5, seconds=30) == 0.0
    assert batch.batch_pause_seconds(5, 10, every=0, seconds=30) == 0.0
    assert batch.batch_pause_seconds(5, 10, every=5, seconds=0) == 0.0
    monkeypatch.setenv("TEST_LAKEGEN_PAUSE", "-3")
    assert batch.non_negative_float_from_env("TEST_LAKEGEN_PAUSE", 0.0) == 0.0
    monkeypatch.setenv("TEST_LAKEGEN_PAUSE", "invalid")
    assert batch.non_negative_float_from_env("TEST_LAKEGEN_PAUSE", 1.5) == 1.5
    monkeypatch.setenv("TEST_LAKEGEN_PAUSE", "30")
    assert batch.non_negative_float_from_env("TEST_LAKEGEN_PAUSE", 0.0) == 30.0


def test_run_batch_pauses_between_groups_of_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(batch, "bootstrap_nltk_data", lambda: None)
    monkeypatch.setattr(batch, "ExperimentConfig", type("Config", (), {
        "model_validate": staticmethod(lambda _settings: object())
    }))
    monkeypatch.setattr(batch, "BATCH_PAUSE_EVERY", 5)
    monkeypatch.setattr(batch, "BATCH_PAUSE_SECONDS", 30.0)
    events = []
    monkeypatch.setattr(batch.time, "sleep", lambda seconds: events.append(("sleep", seconds)))

    class FakeResult:
        def to_dict(self):
            return {"status": "completed", "error": "", "tables": []}

    class FakeRunner:
        def __init__(self, _config):
            pass

        def run(self, question, **kwargs):
            events.append(("run", question))
            return FakeResult()

    monkeypatch.setattr(batch, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(batch, "append_batch_table_metrics", lambda *_args: False)
    job_id = "d" * 32
    questions = [
        {"question": f"Q{index}?", "source_path": f"$.cases[{index}].question",
         "source_id": f"q{index}", "log_fields": {}}
        for index in range(11)
    ]
    jobs.JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "started_at": None, "updated_at": "start",
        "processed": 0, "failed": 0, "metrics_logged": False, "results": [],
    }

    batch.run_batch(job_id, questions, {"resolved_config": {}})

    # A rest after the 5th and the 10th question, none after the last one.
    assert [index for index, event in enumerate(events) if event[0] == "sleep"] == [5, 11]
    assert [event[1] for event in events if event[0] == "sleep"] == [30.0, 30.0]
    assert jobs.JOBS[job_id]["status"] == "completed"
    assert jobs.JOBS[job_id]["processed"] == 11
    jobs.JOBS.pop(job_id, None)


def test_question_tokens_count_every_phase_and_each_judged_variant():
    result = {
        "tokens": {"p1_p2": 1000, "p3": 600, "p4": 50},
        "code_evaluation": {"semantic_judge_tokens": 70},
        "coder_context_experiment": {"variants": {
            "full": {"code_evaluation": {"semantic_judge_tokens": 70}},
            "minimal": {"code_evaluation": {"semantic_judge_tokens": 30}},
        }},
    }
    assert batch.question_tokens(result) == 1650 + 70 + 30
    # Without variants the top-level evaluation is the only judgement.
    del result["coder_context_experiment"]
    assert batch.question_tokens(result) == 1650 + 70
    assert batch.question_tokens({"status": "failed"}) == 0


def test_token_pause_lasts_as_long_as_the_tokens_take_at_the_rate():
    assert batch.token_pause_seconds(30_000, 5, per_minute=10_000) == 180.0
    assert batch.token_pause_seconds(30_000, 0, per_minute=10_000) == 0.0
    assert batch.token_pause_seconds(30_000, 5, per_minute=0) == 0.0
    assert batch.token_pause_seconds(0, 5, per_minute=10_000) == 0.0


def test_token_rate_file_overrides_the_variable_and_is_reread(tmp_path, monkeypatch):
    rate_file = tmp_path / "tokens_per_minute"
    monkeypatch.setattr(batch, "BATCH_TOKENS_PER_MINUTE", 6_000.0)
    monkeypatch.setattr(batch, "BATCH_TOKENS_PER_MINUTE_FILE", str(rate_file))
    # No file yet, then garbage: the variable holds.
    assert batch.token_pause_seconds(30_000, 5) == 300.0
    rate_file.write_text("fast\n")
    assert batch.token_pause_seconds(30_000, 5) == 300.0
    rate_file.write_text("10000\n")
    assert batch.token_pause_seconds(30_000, 5) == 180.0
    rate_file.write_text("0")
    assert batch.token_pause_seconds(30_000, 5) == 0.0


def test_run_batch_rests_in_proportion_to_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(batch, "bootstrap_nltk_data", lambda: None)
    monkeypatch.setattr(batch, "ExperimentConfig", type("Config", (), {
        "model_validate": staticmethod(lambda _settings: object())
    }))
    monkeypatch.setattr(batch, "BATCH_PAUSE_EVERY", 0)
    monkeypatch.setattr(batch, "BATCH_TOKENS_PER_MINUTE", 6_000.0)
    sleeps = []
    monkeypatch.setattr(batch.time, "sleep", sleeps.append)
    spent = iter([1_000, 3_000, 2_000])

    class FakeResult:
        def __init__(self, tokens):
            self.tokens = tokens

        def to_dict(self):
            return {"status": "completed", "error": "", "tables": [],
                    "tokens": {"p1_p2": self.tokens, "p3": 0, "p4": 0}}

    class FakeRunner:
        def __init__(self, _config):
            pass

        def run(self, question, **kwargs):
            return FakeResult(next(spent))

    monkeypatch.setattr(batch, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(batch, "append_batch_table_metrics", lambda *_args: False)
    job_id = "e" * 32
    questions = [
        {"question": f"Q{index}?", "source_path": f"$.cases[{index}].question",
         "source_id": f"q{index}", "log_fields": {}}
        for index in range(3)
    ]
    jobs.JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "started_at": None, "updated_at": "start",
        "processed": 0, "failed": 0, "metrics_logged": False, "results": [],
    }

    batch.run_batch(job_id, questions, {"resolved_config": {}})

    # 1,000 then 3,000 tokens at 6,000/min; no rest after the last question.
    assert sleeps == [10.0, 30.0]
    jobs.JOBS.pop(job_id, None)


def test_swap_model_replaces_the_job_model_at_any_depth():
    config = {"model": "together.m", "experiment_id": "x",
              "retrieval": {"pneuma_entity_model": "together.m", "mode": "pneuma"},
              "semantic_code_judge_model": "together.m", "tags": ["together.m", "y"]}

    swapped = batch.swap_model(config, "together.m", "oci.m")

    assert swapped == {"model": "oci.m", "experiment_id": "x",
                       "retrieval": {"pneuma_entity_model": "oci.m", "mode": "pneuma"},
                       "semantic_code_judge_model": "oci.m", "tags": ["oci.m", "y"]}
    assert config["model"] == "together.m"


def test_split_fraction_is_stable_and_spread():
    ids = [f"q{index}" for index in range(2_000)]
    share = sum(batch.split_fraction(source_id) < 0.2 for source_id in ids) / len(ids)

    assert batch.split_fraction("21a96975-7ba") == batch.split_fraction("21a96975-7ba")
    assert 0.17 < share < 0.23


def test_run_batch_sends_the_split_share_of_questions_to_the_split_model(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    monkeypatch.setattr(batch, "bootstrap_nltk_data", lambda: None)
    monkeypatch.setattr(batch, "ExperimentConfig", type("Config", (), {
        "model_validate": staticmethod(lambda settings: settings)
    }))
    monkeypatch.setattr(batch, "BATCH_PAUSE_EVERY", 0)
    monkeypatch.setattr(batch, "BATCH_TOKENS_PER_MINUTE", 0.0)
    monkeypatch.setattr(batch, "SPLIT_MODEL", "oci.m")
    monkeypatch.setattr(batch, "SPLIT_SHARE", 0.5)
    ran = []

    class FakeResult:
        def to_dict(self):
            return {"status": "completed", "error": "", "tables": [], "tokens": {}}

    class FakeRunner:
        def __init__(self, config):
            self.config = config

        def run(self, question, **kwargs):
            ran.append((kwargs["question_id"], self.config["model"],
                        self.config["semantic_code_judge_model"]))
            return FakeResult()

    monkeypatch.setattr(batch, "ExperimentRunner", FakeRunner)
    monkeypatch.setattr(batch, "append_batch_table_metrics", lambda *_args: False)
    job_id = "f" * 32
    questions = [
        {"question": f"Q{index}?", "source_path": f"$.cases[{index}].question",
         "source_id": f"q{index}", "log_fields": {}}
        for index in range(40)
    ]
    jobs.JOBS[job_id] = {
        "job_id": job_id, "status": "queued", "started_at": None, "updated_at": "start",
        "processed": 0, "failed": 0, "metrics_logged": False, "results": [],
    }
    settings = {"resolved_config": {"model": "together.m",
                                    "semantic_code_judge_model": "together.m"}}

    batch.run_batch(job_id, questions, settings)

    expected = {f"q{index}": "oci.m" if batch.split_fraction(f"q{index}") < 0.5 else "together.m"
                for index in range(40)}
    assert {question_id: model for question_id, model, _judge in ran} == expected
    assert all(model == judge for _id, model, judge in ran)
    assert {"oci.m", "together.m"} == set(expected.values())
    assert [entry["model"] for entry in jobs.JOBS[job_id]["results"]] == list(expected.values())
    jobs.JOBS.pop(job_id, None)


def test_recovery_requeues_jobs_in_submission_order(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOB_DIR", tmp_path)
    submitted = []
    monkeypatch.setattr(
        batch.EXECUTOR, "submit", lambda _function, job_id, *_args: submitted.append(job_id)
    )
    # The job with the smaller id was submitted later; a completed job is left alone.
    stored = {
        "a" * 32: ("queued", "2026-09-27T10:00:02+00:00"),
        "f" * 32: ("running", "2026-09-27T10:00:01+00:00"),
        "b" * 32: ("completed", "2026-09-27T10:00:00+00:00"),
    }
    for job_id, (status, created_at) in stored.items():
        jobs.persist_job({
            "job_id": job_id, "status": status, "created_at": created_at, "updated_at": created_at,
            "processed": 0, "failed": 0, "settings": {"resolved_config": {}}, "results": [],
        })
        jobs.persist_questions(job_id, [
            {"question": "One?", "source_path": "$.cases[0].question", "source_id": "q1", "log_fields": {}},
        ])

    recovered = batch.recover_incomplete_jobs()

    assert recovered == submitted == ["f" * 32, "a" * 32]
    for job_id in recovered:
        jobs.JOBS.pop(job_id, None)
