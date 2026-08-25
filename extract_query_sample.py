#!/usr/bin/env python3
"""Create a deterministic, metric-ready sample from generated query JSON.

The generator output contains many nested copies of each question (judge
feedback, plans, retries, and metadata).  This script selects only the actual
query records, keeps successful queries, and resolves aliases such as
``Table_0`` to the real dataset identifiers stored in each group's ``_meta``.

The resulting top-level ``cases`` format can be consumed both by LakeGen's
batch API and by ``lakegen.retrieval.benchmark``.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROOT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lakegen.code_evaluation import evaluate_code_result
from lakegen.reference_execution import execute_pandas_reference


DEFAULT_INPUT = Path("queries/generated_queries_nyc.json")
DEFAULT_OUTPUT = Path("benchmark/100q_nyc.json")
DEFAULT_VALID_OUTPUT = Path("benchmark/all_valid.json")
DEFAULT_REJECTED_OUTPUT = Path("benchmark/rejected.json")
DEFAULT_TABLES_DIR = Path("data/nyc/datasets/parquet")


def _records(payload: Any) -> list[dict[str, Any]]:
    """Flatten canonical query records while retaining their group metadata."""

    records: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        raise ValueError("The input root must be a JSON object")

    for engine, families in payload.items():
        if not isinstance(families, dict):
            continue
        for query_kind, groups in families.items():
            if not isinstance(groups, dict):
                continue
            for group_name, group in groups.items():
                if not isinstance(group, dict):
                    continue
                metadata = group.get("_meta", {})
                table_map = metadata.get("tables", {}) if isinstance(metadata, dict) else {}
                if not isinstance(table_map, dict):
                    table_map = {}

                for record_key, record in group.items():
                    if record_key == "_meta" or not isinstance(record, dict):
                        continue
                    # Canonical records have execution status and a question.
                    # Nested plans and judge attempts do not occur at this level.
                    if not isinstance(record.get("question"), str) or "status" not in record:
                        continue
                    records.append(
                        {
                            "engine": str(engine),
                            "query_kind": str(query_kind),
                            "group": str(group_name),
                            "record_key": str(record_key),
                            "table_map": table_map,
                            "record": record,
                        }
                    )
    return records


def _largest_remainder_quotas(sizes: dict[tuple[str, str], int], count: int) -> dict[tuple[str, str], int]:
    """Allocate a proportional sample without exceeding any stratum."""

    total = sum(sizes.values())
    if count > total:
        raise ValueError(f"Requested {count} cases, but only {total} eligible cases exist")
    exact = {key: count * size / total for key, size in sizes.items()}
    quotas = {key: min(size, int(exact[key])) for key, size in sizes.items()}
    remaining = count - sum(quotas.values())
    order = sorted(sizes, key=lambda key: (exact[key] - quotas[key], sizes[key], key), reverse=True)
    for key in order:
        if remaining == 0:
            break
        if quotas[key] < sizes[key]:
            quotas[key] += 1
            remaining -= 1
    return quotas


def _normalize(item: dict[str, Any]) -> dict[str, Any]:
    record = item["record"]
    table_map = item["table_map"]
    table_details = record.get("tables") if isinstance(record.get("tables"), list) else []

    aliases: list[str] = []
    for table in table_details:
        if isinstance(table, dict) and isinstance(table.get("name"), str):
            aliases.append(table["name"])
        elif isinstance(table, str):
            aliases.append(table)

    unresolved = [alias for alias in aliases if alias not in table_map]
    if unresolved:
        location = f"{item['engine']}.{item['query_kind']}.{item['group']}.{item['record_key']}"
        raise ValueError(f"Unresolved table aliases at {location}: {unresolved}")
    relevant_table_ids = list(dict.fromkeys(str(table_map[alias]) for alias in aliases))
    if not relevant_table_ids:
        raise ValueError(f"Query {record.get('client_id')!r} has no gold tables")

    keywords = record.get("question_keywords") or record.get("plan_keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValueError(f"Query {record.get('client_id')!r} has no fixed keywords")

    case: dict[str, Any] = {
        "id": str(record.get("client_id") or f"{item['engine']}-{item['query_kind']}-{item['group']}-{item['record_key']}"),
        "question": record["question"].strip(),
        "keywords": list(dict.fromkeys(str(keyword) for keyword in keywords if str(keyword).strip())),
        "relevant_table_ids": relevant_table_ids,
        "table_aliases": {alias: table_map[alias] for alias in aliases},
        "tables": table_details,
        "engine": item["engine"],
        "query_kind": item["query_kind"],
        "source_group": item["group"],
        "difficulty": record.get("difficulty"),
        "expected_result_type": record.get("expected_result_type"),
        "expected_result_description": record.get("expected_result_description"),
        "reference_code": record.get("code"),
        "reference_result": record.get("query_result"),
    }
    return {key: value for key, value in case.items() if value is not None}


def _static_reproducibility_reasons(code: str) -> list[str]:
    """Reject explicit randomness/time dependence and unordered SQL limits."""

    reasons: list[str] = []
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return ["invalid_reference_syntax"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ""
        if isinstance(node.func, ast.Name):
            name = node.func.id.casefold()
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr.casefold()
        if name in {"now", "today", "utcnow", "uuid4", "rand", "randn", "choice"}:
            reasons.append("non_deterministic_reference")
        if name == "sample" and not any(
            keyword.arg == "random_state" for keyword in node.keywords
        ):
            reasons.append("non_deterministic_reference")
    if re.search(r"\blimit\s+\d+", code, re.IGNORECASE) and not re.search(
        r"\border\s+by\b", code, re.IGNORECASE
    ):
        reasons.append("limit_without_order_by")
    return list(dict.fromkeys(reasons))


def _equivalent(
    *, expected_result_type: str, reference: Any, actual: Any,
    expected_description: str = "",
) -> bool:
    comparison = evaluate_code_result(
        expected_result_type=expected_result_type,
        reference_result=reference,
        actual_result=actual,
        expected_description=expected_description,
    )
    return bool(
        comparison.get("applicable")
        and (
            comparison.get("exact_result_match")
            or comparison.get("representation_equivalent_match")
        )
    )


def validate_case(
    case: dict[str, Any], *, tables_dir: Path, cache_dir: Path,
    validation_runs: int, timeout_seconds: int = 180,
    require_declared_match: bool = False,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Execute and freeze one independent, deterministic Pandas gold."""

    reasons: list[str] = []
    for field in ("question", "reference_code", "expected_result_type"):
        if not str(case.get(field) or "").strip():
            reasons.append(f"missing_{field}")
    if "reference_result" not in case or case.get("reference_result") is None:
        reasons.append("missing_reference_result")
    if not case.get("relevant_table_ids"):
        reasons.append("missing_relevant_tables")
    aliases = case.get("table_aliases")
    if not isinstance(aliases, dict) or not aliases:
        reasons.append("missing_table_aliases")
    code = str(case.get("reference_code") or "")
    reasons.extend(_static_reproducibility_reasons(code))
    if reasons:
        return None, list(dict.fromkeys(reasons))

    executions: list[dict[str, Any]] = []
    for _run in range(validation_runs):
        execution = execute_pandas_reference(
            reference_code=code,
            table_aliases=aliases,
            tables_dir=tables_dir,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            use_cache=False,
        )
        executions.append(execution)
        if execution.get("status") != "success":
            detail = str(execution.get("error") or "reference execution failed")
            return None, ["reference_execution_error", detail]

    expected_type = str(case["expected_result_type"])
    description = str(case.get("expected_result_description") or "")
    canonical = executions[0]["result"]
    if any(
        not _equivalent(
            expected_result_type=expected_type,
            reference=canonical,
            actual=execution["result"],
            expected_description=description,
        )
        for execution in executions[1:]
    ):
        return None, ["non_deterministic_result"]

    declared_match = _equivalent(
        expected_result_type=expected_type,
        reference=case["reference_result"],
        actual=canonical,
        expected_description=description,
    )
    if require_declared_match and not declared_match:
        return None, ["declared_result_drift"]

    validated = dict(case)
    validated["declared_reference_result"] = case["reference_result"]
    validated["reference_result"] = canonical
    validated["gold_validation"] = {
        "status": "benchmark_ready",
        "execution_success": True,
        "deterministic": True,
        "validation_runs": validation_runs,
        "declared_result_match": declared_match,
        "declared_result_drift": not declared_match,
    }
    return validated, []


def validate_candidates(
    payload: Any, *, tables_dir: Path, validation_runs: int,
    timeout_seconds: int = 180, require_declared_match: bool = False,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate every successful Pandas candidate before sampling."""

    candidates = [
        item for item in _records(payload)
        if item["record"].get("status") == "success"
        and item["engine"].casefold() == "pandas"
    ]
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lakegen-gold-validation-") as tmp:
        cache_dir = Path(tmp)
        for index, item in enumerate(candidates, start=1):
            try:
                case = _normalize(item)
                validated, reasons = validate_case(
                    case,
                    tables_dir=tables_dir,
                    cache_dir=cache_dir,
                    validation_runs=validation_runs,
                    timeout_seconds=timeout_seconds,
                    require_declared_match=require_declared_match,
                )
            except Exception as exc:
                case = {
                    "id": str(item["record"].get("client_id") or item["record_key"]),
                    "question": item["record"].get("question"),
                }
                validated = None
                reasons = ["candidate_normalization_error", f"{type(exc).__name__}: {exc}"]
            if validated is None:
                rejected.append({**case, "rejection_reasons": reasons})
            else:
                valid.append(validated)
            if progress:
                print(
                    f"Gold validation: {index}/{len(candidates)} "
                    f"(valid={len(valid)}, rejected={len(rejected)})",
                    flush=True,
                )
    return valid, rejected


def make_sample(payload: Any, *, count: int, seed: int) -> dict[str, Any]:
    eligible = [
        item
        for item in _records(payload)
        if item["record"].get("status") == "success"
        and item["engine"].casefold() == "pandas"
    ]
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        strata[(item["engine"], item["query_kind"])].append(item)

    quotas = _largest_remainder_quotas({key: len(value) for key, value in strata.items()}, count)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        selected.extend(rng.sample(strata[key], quotas[key]))
    rng.shuffle(selected)

    cases = [_normalize(item) for item in selected]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("The selected client_id values are not unique")

    return {
        "sample_metadata": {
            "source": str(DEFAULT_INPUT),
            "generation_seed": None,
            "generation_seed_note": "Not recorded in the source query file",
            "sampling_seed": seed,
            "count": len(cases),
            "engine_filter": "PANDAS",
            "selection": "proportional_by_query_kind_from_successful_pandas_records",
            "strata": {
                f"{engine}/{kind}": quotas[(engine, kind)]
                for engine, kind in sorted(quotas)
            },
        },
        "cases": cases,
    }


def sample_valid_cases(
    valid_cases: list[dict[str, Any]], *, count: int, seed: int,
    source: str,
) -> dict[str, Any]:
    """Sample only after independent gold validation."""

    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in valid_cases:
        strata[(str(case["engine"]), str(case["query_kind"]))].append(case)
    quotas = _largest_remainder_quotas(
        {key: len(value) for key, value in strata.items()}, count
    )
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(strata):
        selected.extend(rng.sample(strata[key], quotas[key]))
    rng.shuffle(selected)
    return {
        "sample_metadata": {
            "source": source,
            "sampling_seed": seed,
            "count": len(selected),
            "engine_filter": "PANDAS",
            "selection": "proportional_by_query_kind_from_validated_pandas_records",
            "gold_validation": "independent_reference_execution",
            "strata": {
                f"{engine}/{kind}": quotas[(engine, kind)]
                for engine, kind in sorted(quotas)
            },
        },
        "cases": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--valid-output", type=Path, default=DEFAULT_VALID_OUTPUT)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED_OUTPUT)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--require-declared-match", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.validation_runs <= 0:
        parser.error("--validation-runs must be positive")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    valid, rejected = validate_candidates(
        payload,
        tables_dir=args.tables_dir,
        validation_runs=args.validation_runs,
        timeout_seconds=args.timeout_seconds,
        require_declared_match=args.require_declared_match,
        progress=True,
    )
    sample = sample_valid_cases(
        valid, count=args.count, seed=args.seed, source=str(args.input)
    )
    for path, document in (
        (args.valid_output, {"validation_metadata": {
            "source": str(args.input),
            "tables_dir": str(args.tables_dir),
            "validation_runs": args.validation_runs,
            "valid_count": len(valid),
        }, "cases": valid}),
        (args.rejected_output, {"validation_metadata": {
            "source": str(args.input),
            "rejected_count": len(rejected),
        }, "cases": rejected}),
        (args.output, sample),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"Wrote {len(valid)} valid, {len(rejected)} rejected, and "
        f"{len(sample['cases'])} sampled cases.",
        flush=True,
    )


if __name__ == "__main__":
    main()
