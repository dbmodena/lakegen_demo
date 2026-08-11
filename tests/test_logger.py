import csv
import json

import lakegen.core.logger as experiment_logger


def test_custom_csv_filename_uses_the_existing_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_logger, "LOG_DIR", tmp_path)

    experiment_logger.save_experiment_log(
        question="Test question?",
        code="print('ok')",
        result="ok",
        retries=0,
        csv_filename="api_experiments_log.csv",
    )

    csv_path = tmp_path / "api_experiments_log.csv"
    assert csv_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == experiment_logger.API_CSV_LOG_COLUMNS
        row = next(reader)
    assert row["QUESTION"] == "Test question?"
    assert not (tmp_path / "experiments_log.csv").exists()


def test_existing_csv_gains_model_and_architecture_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_logger, "LOG_DIR", tmp_path)
    csv_path = tmp_path / "api_experiments_log.csv"
    csv_path.write_text(
        "ID,TIMESTAMP,QUESTION\n1,2026-01-01 10:00:00,Old question?\n",
        encoding="utf-8",
    )

    experiment_logger.save_experiment_log(
        question="New question?",
        code="print('ok')",
        result="ok",
        retries=0,
        csv_filename="api_experiments_log.csv",
        model="openai.gpt-oss-120b",
        architecture="unified",
    )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert "MODEL" in reader.fieldnames
    assert "ARCHITECTURE" in reader.fieldnames
    assert rows[0]["MODEL"] == ""
    assert rows[0]["ARCHITECTURE"] == ""
    assert rows[1]["MODEL"] == "openai.gpt-oss-120b"
    assert rows[1]["ARCHITECTURE"] == "unified"


def test_api_log_keeps_analytical_fields_and_excludes_large_payloads(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_logger, "LOG_DIR", tmp_path)
    long_result = "x" * 700

    experiment_logger.save_experiment_log(
        question="Test question?",
        code="print('complete')",
        result=long_result,
        retries=0,
        status="completed",
        elapsed_seconds=1.25,
        csv_filename="api_experiments_log.csv",
        extra_fields={
            "JOB_ID": "job-1",
            "SOURCE_DIFFICULTY": "easy",
            "SOURCE_TABLES": [{"name": "Table_0"}],
            "RETRIEVAL_RUNS_JSON": [{"mode": "keyword", "hits": []}],
        },
    )

    with (tmp_path / "api_experiments_log.csv").open(
        newline="", encoding="utf-8"
    ) as csv_file:
        row = next(csv.DictReader(csv_file))

    assert "RAW_RESULT" not in row
    assert "CODE" not in row
    assert "RETRIEVAL_RUNS_JSON" not in row
    assert "SOURCE_TABLES" not in row
    assert row["STATUS"] == "completed"
    assert row["ELAPSED_SECONDS"] == "1.25"
    assert "SOURCE_DIFFICULTY" not in row


def test_full_trace_is_written_to_text_but_never_to_api_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_logger, "LOG_DIR", tmp_path)

    experiment_logger.save_experiment_log(
        question="Trace question?",
        code="print('ok')",
        result="ok",
        retries=0,
        full_trace="ToolCall: search_solr\nToolResult: table.parquet",
        csv_filename="api_experiments_log.csv",
    )

    with (tmp_path / "api_experiments_log.csv").open(
        newline="", encoding="utf-8"
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)

    assert "FULL_TRACE" not in reader.fieldnames
    assert "search_solr" not in " ".join(str(value) for value in row.values())
    text_log = (tmp_path / "experiments_log.txt").read_text(encoding="utf-8")
    assert "=== WORKFLOW TRACE ===" in text_log
    assert "ToolCall: search_solr" in text_log


def test_existing_api_csv_drops_legacy_full_trace_column(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_logger, "LOG_DIR", tmp_path)
    csv_path = tmp_path / "api_experiments_log.csv"
    csv_path.write_text(
        "ID,TIMESTAMP,QUESTION,FULL_TRACE\n"
        "1,2026-01-01 10:00:00,Old question?,legacy tool trace\n",
        encoding="utf-8",
    )

    experiment_logger.save_experiment_log(
        question="New question?",
        code="print('ok')",
        result="ok",
        retries=0,
        csv_filename="api_experiments_log.csv",
    )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert "FULL_TRACE" not in reader.fieldnames
    assert [row["QUESTION"] for row in rows] == ["Old question?", "New question?"]


def test_existing_api_csv_with_large_field_can_evolve_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment_logger, "LOG_DIR", tmp_path)
    csv_path = tmp_path / "api_experiments_log.csv"
    large_trace = "x" * 200_000
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["ID", "AGENT_THINKING"])
        writer.writeheader()
        writer.writerow({"ID": 1, "AGENT_THINKING": large_trace})

    experiment_logger.save_experiment_log(
        question="New question?",
        code="print('ok')",
        result="ok",
        retries=0,
        csv_filename="api_experiments_log.csv",
    )

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert "AGENT_THINKING" not in rows[0]
    assert rows[1]["QUESTION"] == "New question?"
