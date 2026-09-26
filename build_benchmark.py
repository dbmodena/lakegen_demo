#!/usr/bin/env python3
"""Build a UK or NYC benchmark from successful generated questions and code.

The source JSON is authoritative: this tool does not generate questions and
does not execute or otherwise modify the generated reference code. It selects
successful generated questions deterministically (100 by default, from one or
more source files), preserving their question, code, expected result and table
aliases. --all-multi-table keeps every multi-table question.
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


def _normalize(item: dict[str, Any], *, require_keywords: bool = True) -> dict[str, Any]:
    """Produce the batch-compatible case without executing generated code.

    Reference keywords only feed the retriever-only benchmark (which rejects a
    case without them); agentic runs generate their own, so
    ``require_keywords=False`` keeps such cases.
    """

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
    relevant_table_ids = list(dict.fromkeys(str(table_map[alias]) for alias in aliases))
    retrieval_tables = record.get("retrieval", {}).get("tables", {})
    accepted_table_alternatives: dict[str, list[str]] = {}
    if isinstance(retrieval_tables, dict):
        normalized_contract = {
            str(table_id).rsplit("___", 1)[-1]: details
            for table_id, details in retrieval_tables.items()
            if isinstance(details, dict)
        }
        for table_id in relevant_table_ids:
            resource_id = table_id.rsplit("___", 1)[-1]
            alternatives = normalized_contract.get(resource_id, {}).get(
                "accepted_table_ids", []
            )
            if isinstance(alternatives, list):
                normalized_alternatives = list(dict.fromkeys(
                    str(value) for value in alternatives if str(value).strip()
                ))
                if normalized_alternatives:
                    accepted_table_alternatives[table_id] = normalized_alternatives
    difficulty = _difficulty(record.get("difficulty"))
    keywords = record.get("question_keywords") or record.get("plan_keywords") or []
    if not isinstance(keywords, list) or not any(str(value).strip() for value in keywords):
        if require_keywords:
            raise ValueError(f"Query {record.get('client_id')!r} has no retrieval keywords")
        keywords = []
    return {
        "id": str(record.get("client_id") or f"{item['engine']}-{item['query_kind']}-{item['group']}-{item['record_key']}"),
        "question": record["question"].strip(),
        # None (omitted from the output) when the source has no keywords.
        "keywords": list(dict.fromkeys(str(value) for value in keywords if str(value).strip())) or None,
        "relevant_table_ids": relevant_table_ids,
        "accepted_table_alternatives": accepted_table_alternatives or None,
        "table_aliases": {alias: table_map[alias] for alias in aliases},
        "tables": raw_tables,
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

    # Prefer a one-third multi-table quota, then reduce it when the source pool
    # cannot support that target while preserving the difficulty quotas.
    preferred_multi_target = (count + 2) // 3
    allocations: list[tuple[int, int, int]] = []
    multi_target = preferred_multi_target
    while multi_target >= 0 and not allocations:
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
        multi_target -= 1
    if not allocations:
        availability = {
            difficulty: {scope: len(pool[(difficulty, scope)]) for scope in scopes}
            for difficulty in DIFFICULTIES
        }
        raise ValueError(
            "Cannot satisfy the requested difficulty "
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


def _select_all_multi(
    cases: list[dict[str, Any]], *, count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Keep every multi-table case; fill the rest with single-table ones.

    Single-table cases are spread so that each difficulty gets as close as
    possible to count/3 overall; a difficulty short of single-table cases
    passes its remainder to the others.
    """
    multi = [case for case in cases if case["table_scope"] == "multi_table"]
    if len(multi) > count:
        raise ValueError(
            f"{len(multi)} multi-table cases do not fit in a benchmark of {count}"
        )
    singles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if case["table_scope"] == "single_table":
            singles[case["difficulty"]].append(case)
    quotas = _difficulty_quotas(count)
    multi_by_difficulty = {
        difficulty: sum(case["difficulty"] == difficulty for case in multi)
        for difficulty in DIFFICULTIES
    }
    single_by_difficulty = {difficulty: 0 for difficulty in DIFFICULTIES}
    # Hand out single-table slots one at a time to the difficulty furthest
    # below its quota that still has cases left.
    for _ in range(count - len(multi)):
        open_levels = [
            difficulty for difficulty in DIFFICULTIES
            if single_by_difficulty[difficulty] < len(singles[difficulty])
        ]
        if not open_levels:
            raise ValueError(
                f"Only {len(cases)} valid cases are available for a benchmark of {count}"
            )
        target = max(
            open_levels,
            key=lambda difficulty: quotas[difficulty]
            - multi_by_difficulty[difficulty] - single_by_difficulty[difficulty],
        )
        single_by_difficulty[target] += 1
    rng = random.Random(seed)
    selected = list(multi)
    matrix: dict[str, dict[str, int]] = {}
    for difficulty in DIFFICULTIES:
        selected.extend(rng.sample(singles[difficulty], single_by_difficulty[difficulty]))
        matrix[difficulty] = {
            "multi_table": multi_by_difficulty[difficulty],
            "single_table": single_by_difficulty[difficulty],
        }
    rng.shuffle(selected)
    if len({case["id"] for case in selected}) != len(selected):
        raise ValueError("Selected generated questions do not have unique client_id values")
    return selected, matrix


def _proportional_shares(sizes: dict[Any, int], total: int) -> dict[Any, int]:
    """Split ``total`` over the keys in proportion to ``sizes`` (largest remainder)."""
    population = sum(sizes.values())
    exact = {key: total * size / population for key, size in sizes.items()}
    shares = {key: int(value) for key, value in exact.items()}
    for key in sorted(sizes, key=lambda k: (-(exact[k] - shares[k]), k))[
        :total - sum(shares.values())
    ]:
        shares[key] += 1
    return shares


def _stage_order(
    selected: list[dict[str, Any]],
    *,
    first_stage: int,
    seed: int,
    first_stage_multi: int | None = None,
) -> list[dict[str, Any]]:
    """Reorder cases so the first ``first_stage`` are a stratified subsample.

    Each (difficulty, table scope) stratum gets its proportional share of the
    first stage (largest remainder), so a run stopped after the first stage
    still reflects the composition of the whole benchmark. With
    ``first_stage_multi`` the first stage instead holds exactly that many
    multi-table cases (the rest single-table), each scope still split
    proportionally by difficulty.
    """
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in selected:
        strata[(case["difficulty"], case["table_scope"])].append(case)
    keys = sorted(strata)
    if first_stage_multi is None:
        shares = _proportional_shares(
            {key: len(strata[key]) for key in keys}, first_stage
        )
    else:
        shares = {}
        for scope, amount in (
            ("multi_table", first_stage_multi),
            ("single_table", first_stage - first_stage_multi),
        ):
            scope_keys = [key for key in keys if key[1] == scope]
            available = sum(len(strata[key]) for key in scope_keys)
            if amount > available:
                raise ValueError(
                    f"first stage needs {amount} {scope} cases, only {available} available"
                )
            shares.update(_proportional_shares(
                {key: len(strata[key]) for key in scope_keys}, amount
            ) if amount else {key: 0 for key in scope_keys})
    rng = random.Random(seed)
    first: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for key in keys:
        members = strata[key]
        chosen = set(id(case) for case in rng.sample(members, shares[key]))
        first.extend(case for case in members if id(case) in chosen)
        rest.extend(case for case in members if id(case) not in chosen)
    rng.shuffle(first)
    rng.shuffle(rest)
    return first + rest


def build_benchmark(
    payload: Any,
    *,
    count: int = 100,
    seed: int = 42,
    source: str = "",
    all_multi_table: bool = False,
    first_stage: int | None = None,
    first_stage_multi: int | None = None,
    require_keywords: bool = True,
) -> dict[str, Any]:
    """Sample a benchmark; ``payload`` may also be a list of source payloads."""
    payloads = payload if isinstance(payload, list) else [payload]
    candidates: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen_questions: set[tuple[str, tuple[str, ...]]] = set()
    # Equivalent questions produced through different engines count once.
    # Sorting makes the retained reference case deterministic without imposing
    # an engine requirement on the benchmark. With several sources, earlier
    # ones win on duplicates.
    for item in (
        record
        for single_payload in payloads
        for record in sorted(
            _records(single_payload),
            key=lambda value: (
                value["engine"], value["query_kind"], value["group"], value["record_key"],
            ),
        )
    ):
        if item["record"].get("status") != "success":
            continue
        try:
            candidate = _normalize(item, require_keywords=require_keywords)
        except ValueError as exc:
            rejected.append(str(exc))
            continue
        identity = (
            candidate["question"],
            tuple(sorted(candidate["relevant_table_ids"])),
        )
        if identity not in seen_questions:
            seen_questions.add(identity)
            candidates.append(candidate)
    selector = _select_all_multi if all_multi_table else _select
    selected, matrix = selector(candidates, count=count, seed=seed)
    stages = None
    if first_stage is not None:
        if not 0 < first_stage < len(selected):
            raise ValueError("first_stage must be between 1 and count - 1")
        if first_stage_multi is not None and not 0 <= first_stage_multi <= first_stage:
            raise ValueError("first_stage_multi must be between 0 and first_stage")
        selected = _stage_order(
            selected, first_stage=first_stage, seed=seed,
            first_stage_multi=first_stage_multi,
        )
        stages = [first_stage, len(selected) - first_stage]
    elif first_stage_multi is not None:
        raise ValueError("first_stage_multi requires first_stage")
    # The remaining fields drive retrieval and direct reference/code metrics.
    # Sampling-only and source-navigation metadata stays in sample_metadata.
    output_fields = (
        "id", "question", "keywords", "relevant_table_ids", "table_aliases",
        "accepted_table_alternatives",
        "reference_code", "reference_result", "expected_result_type",
        "expected_result_description", "evaluation_contract",
    )
    return {
        "sample_metadata": {
            "source": source,
            "sampling_seed": seed,
            "count": len(selected),
            "status_filter": "success",
            "selection": (
                "all_multi_table_then_single_by_difficulty" if all_multi_table
                else "balanced_by_difficulty_and_table_scope"
            ),
            "candidate_count": len(candidates),
            "requires_reference_keywords": require_keywords,
            "difficulty_quotas": _difficulty_quotas(count),
            "table_scope_requirements": {
                "multi_table": sum(row["multi_table"] for row in matrix.values()),
                "single_table": sum(row["single_table"] for row in matrix.values()),
            },
            "strata": matrix,
            # Case counts of consecutive run stages; the first is a
            # stratified subsample of the whole benchmark.
            **({"stages": stages} if stages else {}),
            **(
                {"first_stage_multi_table": first_stage_multi}
                if stages and first_stage_multi is not None else {}
            ),
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
    parser.add_argument(
        "--input", type=Path, nargs="+",
        help="One or more generated-query files; duplicates keep the first file's case",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--all-multi-table", action="store_true",
        help="Include every multi-table case and fill the rest with single-table ones",
    )
    parser.add_argument(
        "--first-stage", type=int,
        help="Order cases so the first N form a stratified subsample (run in stages)",
    )
    parser.add_argument(
        "--first-stage-multi", type=int,
        help="Exact number of multi-table cases in the first stage (rest single-table)",
    )
    parser.add_argument(
        "--allow-missing-keywords", action="store_true",
        help="Keep cases without reference keywords (unusable by the retriever-only benchmark)",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    default_input, default_output = DEFAULT_PATHS[args.dataset]
    input_paths = args.input or [default_input]
    output_path = args.output or default_output
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    try:
        benchmark = build_benchmark(
            payloads, count=args.count, seed=args.seed,
            source=", ".join(str(path) for path in input_paths),
            all_multi_table=args.all_multi_table,
            first_stage=args.first_stage,
            first_stage_multi=args.first_stage_multi,
            require_keywords=not args.allow_missing_keywords,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(benchmark['cases'])} {args.dataset.upper()} benchmark cases to {output_path}.")


if __name__ == "__main__":
    main()
