#!/usr/bin/env python3
"""Build a UK or NYC benchmark from successful generated questions and code.

The source JSON is authoritative: this tool does not generate questions and
does not execute or otherwise modify the generated reference code.  It selects
100 successful Pandas questions deterministically, preserving their question,
code, expected result and table aliases.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PATHS = {
    "uk": (Path("queries/generated_queries_uk.json"), Path("benchmark/100q_uk.json")),
    "nyc": (Path("queries/generated_queries_nyc.json"), Path("benchmark/100q_nyc.json")),
}
DEFAULT_DATASET = "nyc"
DIFFICULTIES = ("easy", "medium", "hard")


def _records(payload: Any) -> list[dict[str, Any]]:
    """Flatten only top-level generated query records and their group metadata."""

    if not isinstance(payload, dict):
        raise ValueError("The input root must be a JSON object")
    records: list[dict[str, Any]] = []
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
                for record_key, record in group.items():
                    if record_key == "_meta" or not isinstance(record, dict):
                        continue
                    if isinstance(record.get("question"), str) and "status" in record:
                        records.append({
                            "engine": str(engine),
                            "query_kind": str(query_kind),
                            "group": str(group_name),
                            "record_key": str(record_key),
                            "table_map": table_map if isinstance(table_map, dict) else {},
                            "record": record,
                        })
    return records


def _difficulty(value: Any) -> str:
    normalized = str(value or "").strip().casefold()
    aliases = {"intermediate": "medium", "moderate": "medium"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty {value!r}; expected easy, medium, or hard")
    return normalized


def _normalize(item: dict[str, Any]) -> dict[str, Any]:
    """Produce the batch-compatible case without executing generated code."""

    record = item["record"]
    table_map = item["table_map"]
    raw_tables = record.get("tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raise ValueError(f"Query {record.get('client_id')!r} has no tables")
    aliases = [
        table["name"] if isinstance(table, dict) and isinstance(table.get("name"), str) else table
        for table in raw_tables
    ]
    if not all(isinstance(alias, str) and alias for alias in aliases):
        raise ValueError(f"Query {record.get('client_id')!r} has an invalid table alias")
    unresolved = [alias for alias in aliases if alias not in table_map]
    if unresolved:
        raise ValueError(f"Query {record.get('client_id')!r} has unresolved aliases: {unresolved}")
    if not isinstance(record.get("code"), str) or not record["code"].strip():
        raise ValueError(f"Query {record.get('client_id')!r} has no generated code")
    if not record["question"].strip():
        raise ValueError(f"Query {record.get('client_id')!r} has an empty question")
    if record.get("query_result") is None:
        raise ValueError(f"Query {record.get('client_id')!r} has no generated result")
    if not isinstance(record.get("expected_result_type"), str) or not record[
        "expected_result_type"
    ].strip():
        raise ValueError(
            f"Query {record.get('client_id')!r} has no expected result type"
        )

    aliases = list(dict.fromkeys(aliases))
    difficulty = _difficulty(record.get("difficulty"))
    keywords = record.get("question_keywords") or record.get("plan_keywords") or []
    if not isinstance(keywords, list) or not any(str(value).strip() for value in keywords):
        raise ValueError(f"Query {record.get('client_id')!r} has no retrieval keywords")
    return {
        "id": str(record.get("client_id") or f"{item['engine']}-{item['query_kind']}-{item['group']}-{item['record_key']}"),
        "question": record["question"].strip(),
        "keywords": list(dict.fromkeys(str(value) for value in keywords if str(value).strip())),
        "relevant_table_ids": list(dict.fromkeys(str(table_map[alias]) for alias in aliases)),
        "table_aliases": {alias: table_map[alias] for alias in aliases},
        "tables": raw_tables,
        "engine": item["engine"],
        "query_kind": item["query_kind"],
        "source_group": item["group"],
        "difficulty": difficulty,
        "table_scope": "single_table" if len(aliases) == 1 else "multi_table",
        "reference_code": record["code"],
        "reference_result": record.get("query_result"),
        "expected_result_type": record.get("expected_result_type"),
        "expected_result_description": record.get("expected_result_description"),
        "evaluation_contract": record.get("evaluation_contract"),
    }


def _difficulty_quotas(count: int) -> dict[str, int]:
    base, remainder = divmod(count, len(DIFFICULTIES))
    return {difficulty: base + int(index < remainder) for index, difficulty in enumerate(DIFFICULTIES)}


def _select(cases: list[dict[str, Any]], *, count: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    quotas = _difficulty_quotas(count)
    scopes = ("single_table", "multi_table")
    pool: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        pool[(case["difficulty"], case["table_scope"])].append(case)

    # Keep a fixed one-third multi-table quota. This gives each benchmark a
    # consistent amount of join reasoning without depending on its full pool.
    multi_target = (count + 2) // 3
    allocations: list[tuple[int, int, int]] = []
    for easy_multi in range(quotas["easy"] + 1):
        for medium_multi in range(quotas["medium"] + 1):
            hard_multi = multi_target - easy_multi - medium_multi
            proposed = {"easy": easy_multi, "medium": medium_multi, "hard": hard_multi}
            if all(
                0 <= proposed[difficulty] <= len(pool[(difficulty, "multi_table")])
                and quotas[difficulty] - proposed[difficulty] <= len(pool[(difficulty, "single_table")])
                for difficulty in DIFFICULTIES
            ):
                allocations.append((easy_multi, medium_multi, hard_multi))
    if not allocations:
        availability = {
            difficulty: {scope: len(pool[(difficulty, scope)]) for scope in scopes}
            for difficulty in DIFFICULTIES
        }
        raise ValueError(
            "Cannot satisfy the requested difficulty and one-third "
            f"multi-table quotas; available={availability}, quotas={quotas}"
        )

    # Prefer a balanced distribution of multi-table questions across levels.
    allocation = min(allocations, key=lambda values: max(values) - min(values))
    multi_by_difficulty = dict(zip(DIFFICULTIES, allocation, strict=True))
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, int]] = {}
    for difficulty in DIFFICULTIES:
        matrix[difficulty] = {}
        for scope, amount in (
            ("multi_table", multi_by_difficulty[difficulty]),
            ("single_table", quotas[difficulty] - multi_by_difficulty[difficulty]),
        ):
            selected.extend(rng.sample(pool[(difficulty, scope)], amount))
            matrix[difficulty][scope] = amount
    rng.shuffle(selected)
    if len({case["id"] for case in selected}) != len(selected):
        raise ValueError("Selected generated questions do not have unique client_id values")
    return selected, matrix


def build_benchmark(payload: Any, *, count: int = 100, seed: int = 42, source: str = "") -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    rejected: list[str] = []
    for item in _records(payload):
        if item["engine"].casefold() != "pandas" or item["record"].get("status") != "success":
            continue
        try:
            candidates.append(_normalize(item))
        except ValueError as exc:
            rejected.append(str(exc))
    selected, matrix = _select(candidates, count=count, seed=seed)
    # The remaining fields drive retrieval and direct reference/code metrics.
    # Sampling-only and source-navigation metadata stays in sample_metadata.
    output_fields = (
        "id", "question", "keywords", "relevant_table_ids", "table_aliases",
        "engine", "reference_code", "reference_result", "expected_result_type",
        "expected_result_description", "evaluation_contract",
    )
    return {
        "sample_metadata": {
            "source": source,
            "sampling_seed": seed,
            "count": len(selected),
            "engine_filter": "PANDAS",
            "status_filter": "success",
            "selection": "balanced_by_difficulty_and_table_scope",
            "difficulty_quotas": _difficulty_quotas(count),
            "table_scope_requirements": {
                "multi_table": (count + 2) // 3,
                "single_table": count - ((count + 2) // 3),
            },
            "strata": matrix,
            "skipped_invalid_generated_records": len(rejected),
        },
        "cases": [
            {field: case[field] for field in output_fields if case.get(field) is not None}
            for case in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=sorted(DEFAULT_PATHS), default=DEFAULT_DATASET
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    default_input, default_output = DEFAULT_PATHS[args.dataset]
    input_path = args.input or default_input
    output_path = args.output or default_output
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    benchmark = build_benchmark(payload, count=args.count, seed=args.seed, source=str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(benchmark['cases'])} {args.dataset.upper()} benchmark cases to {output_path}.")


if __name__ == "__main__":
    main()
