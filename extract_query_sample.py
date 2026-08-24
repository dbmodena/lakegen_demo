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
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("queries/generated_queries_nyc.json")
DEFAULT_OUTPUT = Path("benchmark/100q_nyc.json")


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
        "table_aliases": table_map,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    sample = make_sample(payload, count=args.count, seed=args.seed)
    sample["sample_metadata"]["source"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sample['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
