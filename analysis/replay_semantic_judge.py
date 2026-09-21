"""Replay the semantic judge from saved batch results without running retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from lakegen.core.resources import get_llm, get_prompt_manager
from lakegen.semantic_code_judge import judge_semantic_code_result


BASE_DIR = Path(__file__).resolve().parents[1]


def _load_cases(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload) if isinstance(payload, Mapping) else payload
    return {str(case["id"]): case for case in cases}


def _metadata_index(path: Path) -> dict[str, dict[str, Any]]:
    datasets = json.loads(path.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        package = {
            key: dataset.get(key)
            for key in ("id", "title", "name", "description", "notes", "metadata_modified")
            if dataset.get(key) not in (None, "")
        }
        for resource in dataset.get("resources", []):
            if not isinstance(resource, Mapping) or not resource.get("id"):
                continue
            resource_id = str(resource["id"])
            index[resource_id] = {
                **package,
                "resource": {
                    key: resource.get(key)
                    for key in ("id", "name", "description", "format", "last_modified")
                    if resource.get(key) not in (None, "")
                },
            }
    return index


def _selected_metadata(
    tables: list[str], metadata: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for table in tables:
        resource_id = Path(table).stem.rsplit("___", 1)[-1]
        selected[table] = metadata.get(resource_id, {})
    return selected


def _generated_result(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _deterministic_only(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evaluation.items()
        if not key.startswith("semantic_")
        and key not in {"semantic_judgment", "supported_correct"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Completed source batch job ID")
    parser.add_argument("--context", default="full", choices=("full", "schema_only", "minimal"))
    parser.add_argument("--questions", type=Path, default=BASE_DIR / "benchmark/5q_uk_new.json")
    parser.add_argument(
        "--metadata", type=Path,
        default=BASE_DIR / "data/uk/metadata/metadata_retrieved_cleaned.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results_path = BASE_DIR / ".lakegen_jobs" / f"{args.job_id}.results.jsonl"
    output_path = args.output or Path(f"/tmp/{args.job_id}.judge-replay.jsonl")
    cases = _load_cases(args.questions)
    metadata = _metadata_index(args.metadata)
    records = [json.loads(line) for line in results_path.read_text().splitlines() if line]
    model = records[0]["result"]["configuration"]["semantic_code_judge_model"]
    llm = get_llm(model)[0]
    prompt_manager = get_prompt_manager()

    with output_path.open("w", encoding="utf-8") as output:
        for record in records:
            result = record["result"]
            case = cases[str(record["source_id"])]
            variant = result["coder_context_experiment"]["variants"][args.context]
            evaluation = variant["code_evaluation"]
            if evaluation.get("exact_result_match") is True:
                continue
            tables = list(result.get("tables", []))
            judgment, tokens = judge_semantic_code_result(
                question=result["question"],
                expected_description=str(case.get("expected_result_description") or ""),
                reference_result=case.get("reference_result"),
                selected_tables=tables,
                selected_metadata=_selected_metadata(tables, metadata),
                generated_code=str(variant.get("code") or ""),
                generated_result=_generated_result(variant.get("raw_result")),
                deterministic_evaluation=_deterministic_only(evaluation),
                llm=llm,
                prompt_manager=prompt_manager,
            )
            replay = {
                "source_id": record["source_id"],
                "question": result["question"],
                "context": args.context,
                "previous_disposition": evaluation.get("semantic_correctness"),
                "replay_disposition": judgment["disposition"],
                "tokens": tokens,
                "judgment": judgment,
            }
            output.write(json.dumps(replay, ensure_ascii=False, default=str) + "\n")
            output.flush()
            print(
                record["source_id"], evaluation.get("semantic_correctness"),
                "->", judgment["disposition"], f"({tokens} tokens)", flush=True,
            )
    print(f"Replay written to {output_path}")


if __name__ == "__main__":
    main()
