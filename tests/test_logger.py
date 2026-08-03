import csv

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
        assert reader.fieldnames == experiment_logger.CSV_LOG_COLUMNS
        row = next(reader)
    assert row["QUESTION"] == "Test question?"
    assert not (tmp_path / "experiments_log.csv").exists()
