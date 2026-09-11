import json

import pandas as pd

from lakegen.reference_execution import execute_pandas_reference


def test_pandas_reference_executes_and_uses_snapshot_cache(tmp_path):
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    pd.DataFrame({"value": [1, 2, 3]}).to_parquet(
        tables_dir / "numbers.parquet"
    )
    arguments = {
        "reference_code": "result = Table_0['value'].sum()",
        "table_aliases": {"Table_0": "numbers"},
        "tables_dir": tables_dir,
        "cache_dir": tmp_path / "cache",
    }

    first = execute_pandas_reference(**arguments)
    second = execute_pandas_reference(**arguments)

    assert first["status"] == "success"
    assert first["result"] == 6
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert json.loads(next((tmp_path / "cache").glob("*.json")).read_text())["result"] == 6


def test_pandas_reference_rejects_file_access(tmp_path):
    outcome = execute_pandas_reference(
        reference_code="result = open('/etc/passwd').read()",
        table_aliases={},
        tables_dir=tmp_path,
        cache_dir=tmp_path / "cache",
    )

    assert outcome["status"] == "invalid_reference"
    assert "forbidden reference name" in outcome["error"]
