"""Execute trusted-shape Pandas benchmark references against current tables."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


_FORBIDDEN_NAMES = {
    "__import__", "breakpoint", "compile", "eval", "exec", "globals",
    "help", "input", "locals", "open", "os", "pathlib", "shutil",
    "subprocess", "sys",
}
_FORBIDDEN_ATTRIBUTES = {
    "read_csv", "read_excel", "read_feather", "read_html", "read_json",
    "read_orc", "read_parquet", "read_pickle", "read_sql", "to_clipboard",
    "to_csv", "to_excel", "to_feather", "to_hdf", "to_json", "to_orc",
    "to_parquet", "to_pickle", "to_sql",
}


def _validate_reference_code(code: str) -> ast.Module:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split(".")[0]]
            if any(module not in {"pandas", "numpy"} for module in modules):
                raise ValueError("reference imports are limited to pandas and numpy")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"forbidden reference name {node.id!r}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRIBUTES:
                raise ValueError(f"forbidden reference attribute {node.attr!r}")
    return tree


def _output_expression(tree: ast.Module) -> tuple[ast.Module, ast.expr]:
    if not tree.body:
        raise ValueError("reference_code is empty")
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        return ast.Module(body=tree.body[:-1], type_ignores=[]), last.value
    if isinstance(last, (ast.Assign, ast.AnnAssign)):
        target = last.targets[0] if isinstance(last, ast.Assign) else last.target
        if isinstance(target, ast.Name):
            return tree, ast.Name(id=target.id, ctx=ast.Load())
    for statement in reversed(tree.body):
        if isinstance(statement, ast.Assign):
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                return tree, ast.Name(id=target.id, ctx=ast.Load())
    raise ValueError("could not determine the reference output variable")


def _json_result(value: Any) -> Any:
    import numpy as np
    import pandas as pd

    if isinstance(value, pd.DataFrame):
        return [_json_result(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return [{str(key): _json_result(item) for key, item in value.items()}]
    if isinstance(value, Mapping):
        return {str(key): _json_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_result(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    code = str(payload["reference_code"])
    tree = _validate_reference_code(code)
    body, output = _output_expression(tree)
    namespace: dict[str, Any] = {"pd": pd, "np": np}
    for alias, path in payload["tables"].items():
        namespace[str(alias)] = pd.read_parquet(str(path))
    body = ast.fix_missing_locations(body)
    output_expression = ast.fix_missing_locations(ast.Expression(output))
    exec(compile(body, "<benchmark-reference>", "exec"), namespace, namespace)
    value = eval(
        compile(output_expression, "<benchmark-reference-output>", "eval"),
        namespace,
        namespace,
    )
    return {"status": "success", "result": _json_result(value)}


def _cache_key(reference_code: str, tables: Mapping[str, Path]) -> str:
    digest = hashlib.sha256(reference_code.encode("utf-8"))
    for alias, path in sorted(tables.items()):
        stat = path.stat()
        digest.update(f"{alias}:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def execute_pandas_reference(
    *,
    reference_code: str,
    table_aliases: Mapping[str, str],
    tables_dir: Path,
    cache_dir: Path,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Execute one reference in a subprocess and cache it by code/table snapshot."""

    tables: dict[str, Path] = {}
    for alias, table_id in table_aliases.items():
        candidates = [
            tables_dir / str(table_id),
            tables_dir / f"{table_id}.parquet",
            tables_dir / f"{table_id}.pq",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            return {"status": "invalid_reference", "error": f"missing table {table_id}"}
        tables[str(alias)] = path
    try:
        _validate_reference_code(reference_code)
        key = _cache_key(reference_code, tables)
    except Exception as exc:
        return {"status": "invalid_reference", "error": f"{type(exc).__name__}: {exc}"}

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return {**cached, "cache_hit": True, "cache_key": key}

    payload = {
        "reference_code": reference_code,
        "tables": {alias: str(path) for alias, path in tables.items()},
    }
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        if process.returncode != 0:
            detail = process.stderr.strip().splitlines()[-1] if process.stderr.strip() else "worker failed"
            return {"status": "invalid_reference", "error": detail, "cache_hit": False}
        result = json.loads(process.stdout)
        result["cache_hit"] = False
        cache_path.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
        return {**result, "cache_key": key}
    except subprocess.TimeoutExpired:
        return {"status": "invalid_reference", "error": "reference execution timed out", "cache_hit": False}
    except Exception as exc:
        return {"status": "invalid_reference", "error": f"{type(exc).__name__}: {exc}", "cache_hit": False}


def main() -> None:
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("reference_execution is an internal worker")
    try:
        result = _worker(json.loads(sys.stdin.read()))
        print(json.dumps(result, ensure_ascii=False, default=str))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
