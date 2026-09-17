"""HTTP API for single and batch LakeGen queries.

Run locally with::

    uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _path in (_SRC_DIR, _ROOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lakegen.core.bootstrap import bootstrap_nltk_data
from lakegen.core.config import BASE_DIR
from lakegen.service import (
    QuestionSource,
    extract_questions,
    make_runtime_settings,
    run_question,
)
from lakegen.ui.state import (
    MODEL_OPTIONS,
    SOLR_CORE_OPTIONS,
    SOLR_CORE_PORTAL_NAMES,
)
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalMode, check_embedding_health
from lakegen.retrieval.benchmark import append_benchmark_metrics_log
from lakegen.retrieval.evaluation import evaluate_ranking, mean_metrics
from lakegen.code_evaluation import summarize_code_evaluations
from lakegen.experiment_config import (
    CoderContextLevel,
    DiscoveryArchitecture,
    ExperimentConfig,
    load_experiment_config,
    parse_experiment_config_document,
)
from lakegen.runner import ExperimentRunner


DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_CORE = SOLR_CORE_OPTIONS[0]
MAX_BATCH_QUESTIONS = 10_000
MAX_CONFIG_UPLOAD_BYTES = 1_000_000
MAX_QUESTIONS_UPLOAD_BYTES = 25_000_000
JOB_DIR = BASE_DIR / ".lakegen_jobs"
BENCHMARK_DIR = BASE_DIR / "benchmark"
# Use Uvicorn's configured application logger so batch progress is visible in
# the same terminal as the HTTP access log when the API is run with Uvicorn.
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="LakeGen API",
    version="1.0.0",
    description="Query LakeGen directly or submit JSON question batches.",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRequest(StrictModel):
    question: str = Field(min_length=1)
    core: str = DEFAULT_CORE
    model: str = DEFAULT_MODEL
    retrieval_mode: RetrievalMode = RetrievalMode.KEYWORD
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=1000)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(default=5, ge=1, le=100)
    discovery_architecture: DiscoveryArchitecture = DiscoveryArchitecture.UNIFIED
    experiment_id: str = Field(default="default", min_length=1)
    seed: int = Field(default=0, ge=0)
    config: dict[str, Any] | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class BatchAccepted(StrictModel):
    job_id: str
    status: str
    question_count: int
    status_url: str


_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lakegen-batch")
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_workflow_lock = threading.Lock()


def _resolve_api_config(
    *,
    core: str,
    model: str,
    retrieval_mode: RetrievalMode | str,
    top_k: int,
    hybrid_alpha: float,
    candidate_multiplier: int,
    discovery_architecture: DiscoveryArchitecture | str = DiscoveryArchitecture.UNIFIED,
    experiment_id: str = "default",
    seed: int = 0,
    config_data: dict[str, Any] | None = None,
    explicit_fields: set[str] | None = None,
) -> ExperimentConfig:
    explicit = (
        {
            "core", "model", "retrieval_mode", "top_k", "hybrid_alpha",
            "candidate_multiplier", "discovery_architecture", "experiment_id", "seed",
        }
        if explicit_fields is None
        else explicit_fields
    )
    possible = {
        "core": core,
        "model": model,
        "retrieval.mode": retrieval_mode,
        "retrieval.top_k": top_k,
        "retrieval.alpha": hybrid_alpha,
        "retrieval.candidate_multiplier": candidate_multiplier,
        "discovery_architecture": discovery_architecture,
        "experiment_id": experiment_id,
        "seed": seed,
        "interaction_mode": "autonomous",
    }
    aliases = {
        "retrieval_mode": "retrieval.mode",
        "top_k": "retrieval.top_k",
        "hybrid_alpha": "retrieval.alpha",
        "candidate_multiplier": "retrieval.candidate_multiplier",
    }
    override_keys = {aliases.get(name, name) for name in explicit}
    override_keys.add("interaction_mode")
    overrides = {key: value for key, value in possible.items() if key in override_keys}
    return load_experiment_config(
        data_override=config_data,
        overrides=overrides,
    )


def _inline_batch_config(payload: Any) -> dict[str, Any] | None:
    """Read only an inline top-level config from the batch envelope."""

    if not isinstance(payload, dict) or "config" not in payload:
        return None
    config = payload["config"]
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("batch config must be an inline JSON object")
    return config


def _benchmark_file(name: str) -> Path:
    """Return an application-owned benchmark JSON file without path traversal."""

    filename = Path(name).name
    if filename != name or Path(filename).suffix.casefold() != ".json":
        raise HTTPException(
            status_code=404,
            detail="Benchmark must be the name of a JSON file in the benchmark catalog.",
        )
    path = BENCHMARK_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Benchmark not found.")
    return path


def _load_benchmark(name: str) -> Any:
    path = _benchmark_file(name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Invalid benchmark JSON in %s", path.name)
        raise HTTPException(status_code=500, detail="Benchmark JSON is invalid.") from exc


def _upload_suffix(upload: UploadFile, allowed: set[str], label: str) -> str:
    filename = upload.filename or ""
    if not filename or Path(filename).name != filename or "\\" in filename:
        raise ValueError(f"{label} filename must not contain a path")
    suffix = Path(filename).suffix.casefold()
    if suffix not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must use one of these extensions: {expected}")
    return suffix


async def _read_upload(upload: UploadFile, *, limit: int, label: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while chunk := await upload.read(64 * 1024):
        size += len(chunk)
        if size > limit:
            raise HTTPException(
                status_code=413,
                detail=f"{label} exceeds the {limit}-byte upload limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _results_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.results.jsonl"


def _questions_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.questions.json"


def _valid_job_id(job_id: str) -> bool:
    return len(job_id) == 32 and all(character in "0123456789abcdef" for character in job_id)


def _persist_job(job: dict[str, Any]) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    target = _job_path(job["job_id"])
    temporary = target.with_suffix(".tmp")
    metadata = {key: value for key, value in job.items() if key != "results"}
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)


def _persist_questions(job_id: str, questions: list[dict[str, Any]]) -> None:
    """Persist the immutable batch input used to resume interrupted jobs."""

    JOB_DIR.mkdir(parents=True, exist_ok=True)
    target = _questions_path(job_id)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(target)


def _load_questions(job_id: str) -> list[dict[str, Any]] | None:
    if not _valid_job_id(job_id):
        return None
    path = _questions_path(job_id)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else None


def _append_result(job_id: str, result: dict[str, Any]) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    with _results_path(job_id).open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")


def _table_id(value: Any) -> str:
    """Normalize a selected table filename to the gold resource identifier."""

    name = Path(str(value)).name
    suffix = Path(name).suffix.casefold()
    return Path(name).stem if suffix in {".csv", ".parquet", ".pq"} else name


def _resource_id(value: Any) -> str:
    """Return the resource part shared by benchmark gold IDs and rankings."""
    return _table_id(value).rsplit("___", 1)[-1]


def _retrieval_ranking(result: dict[str, Any]) -> list[str]:
    """Return the final retrieval attempt in rank order, with selection fallback."""
    entries = [entry for entry in result.get("ranking", []) if isinstance(entry, dict)]
    if not entries:
        return list(dict.fromkeys(_resource_id(value) for value in result.get("tables", [])))
    final_attempt = max(int(entry.get("attempt", 1)) for entry in entries)
    final_entries = sorted(
        (entry for entry in entries if int(entry.get("attempt", 1)) == final_attempt),
        key=lambda entry: int(entry.get("rank", 0)),
    )
    return list(dict.fromkeys(
        _resource_id(entry["resource_id"])
        for entry in final_entries
        if entry.get("resource_id")
    ))


def _append_batch_table_metrics(
    job_id: str,
    questions: list[dict[str, Any]],
    results: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    append_log: bool = True,
) -> dict[str, Any] | None:
    """Log end-to-end table-selection metrics for metric-ready batches.

    A batch is metric-ready only when every question supplies non-empty
    ``relevant_table_ids`` metadata.  Older query documents are skipped rather
    than producing partial or misleading aggregate metrics.
    """

    if not questions or len(questions) != len(results):
        return None

    case_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    successful_metric_rows: list[dict[str, float]] = []
    selection_metric_rows: list[dict[str, float]] = []
    expansion_query_count = 0
    expansion_contribution_count = 0
    for source, entry in zip(questions, results, strict=True):
        gold = source.get("log_fields", {}).get("SOURCE_RELEVANT_TABLE_IDS")
        if not isinstance(gold, list) or not gold:
            return None
        relevant = list(dict.fromkeys(_resource_id(value) for value in gold))
        result = entry.get("result", {})
        ranking = _retrieval_ranking(result)
        metrics = evaluate_ranking(ranking, relevant, k_values=(1, 5, 10, 15, 20))
        metric_rows.append(metrics)
        selected = list(dict.fromkeys(
            _resource_id(value) for value in result.get("tables", [])
        ))
        relevant_set = set(relevant)
        selected_set = set(selected)
        for k in (1, 5, 10, 15, 20):
            metrics[f"FullCoverage@{k}"] = float(
                relevant_set.issubset(set(ranking[:k]))
            )
        matches = len(relevant_set & selected_set)
        selection_metric_rows.append({
            "SelectionHit": float(matches > 0),
            "SelectionRecall": matches / len(relevant_set),
            "SelectionPrecision": matches / len(selected_set) if selected_set else 0.0,
            "ExactSelection": float(selected_set == relevant_set),
        })
        if result.get("discovery", {}).get("expansion_used"):
            expansion_query_count += 1
            ranks = {resource_id: rank for rank, resource_id in enumerate(ranking, 1)}
            contributed = any(
                resource_id in relevant_set
                and resource_id in selected_set
                and ranks.get(resource_id, 0) > 10
                for resource_id in selected_set
            )
            expansion_contribution_count += int(contributed)
        error = str(result.get("error") or "")
        if not error:
            successful_metric_rows.append(metrics)
        case_rows.append({
            "case_id": str(source.get("source_id") or source.get("source_path")),
            "question": source["question"],
            "relevant_table_ids": relevant,
            "ranking": ranking,
            "metrics": metrics,
            "gold_table_count": len(relevant_set),
            "selected_table_count": len(selected_set),
            "selection_metrics": selection_metric_rows[-1],
            "selection_outcome": (
                "exact" if selected_set == relevant_set
                else "miss" if not matches
                else "over_selection" if relevant_set.issubset(selected_set)
                else "partial"
            ),
            "expansion_used": bool(result.get("discovery", {}).get("expansion_used")),
            "elapsed_seconds": float(result.get("elapsed_seconds") or 0.0),
            "error": error,
        })

    resolved = settings["resolved_config"]
    retrieval = resolved["retrieval"]
    base_label = str(resolved.get("experiment_id") or "batch")
    automatic_test_coder = bool(resolved.get("automatic_test_coder"))
    levels = (
        list(CoderContextLevel)
        if automatic_test_coder
        else [CoderContextLevel(resolved.get("coder_context_level", "full"))]
    )
    experiments: dict[str, Any] = {}
    query_elapsed = [
        float(entry.get("result", {}).get("elapsed_seconds") or 0.0)
        for entry in results
    ]
    token_keys = ("p1_p2", "p3", "p4")
    token_totals = {
        key: sum(int(entry.get("result", {}).get("tokens", {}).get(key, 0) or 0) for entry in results)
        for key in token_keys
    }
    coder_contexts = ("full", "schema_only", "minimal")
    coder_token_totals = {
        context: sum(
            int(
                entry.get("result", {}).get("coder_context_experiment", {})
                .get("variants", {}).get(context, {}).get("tokens", 0) or 0
            )
            for entry in results
        )
        for context in coder_contexts
    }
    coder_token_totals["unattributed"] = max(
        0, token_totals["p3"] - sum(coder_token_totals.values())
    )
    performance_metrics = {
        "query_count": len(results),
        "total_query_elapsed_seconds": sum(query_elapsed),
        "mean_query_elapsed_seconds": statistics.mean(query_elapsed),
        "median_query_elapsed_seconds": statistics.median(query_elapsed),
        "p90_query_elapsed_seconds": sorted(query_elapsed)[
            max(0, math.ceil(0.9 * len(query_elapsed)) - 1)
        ],
        "token_totals": token_totals,
        "mean_tokens_per_query": {
            key: total / len(results) for key, total in token_totals.items()
        },
        "coder_token_totals_by_context": coder_token_totals,
        "mean_coder_tokens_per_query_by_context": {
            key: total / len(results) for key, total in coder_token_totals.items()
        },
    }
    for level in levels:
        label = (
            f"{base_label}-coder-{level.value}"
            if automatic_test_coder
            else base_label
        )
        code_metrics = summarize_code_evaluations(
            results,
            coder_context_level=(level.value if automatic_test_coder else None),
        )
        cardinality_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for case, result_entry in zip(case_rows, results, strict=True):
            count = int(case["gold_table_count"])
            bucket = str(count) if count < 4 else "4+"
            cardinality_groups.setdefault(bucket, []).append((case, result_entry))
        table_count_analysis: dict[str, Any] = {}
        for bucket in ("1", "2", "3", "4+"):
            group = cardinality_groups.get(bucket, [])
            if not group:
                continue
            group_cases = [case for case, _ in group]
            evaluations = []
            for _, result_entry in group:
                result = result_entry.get("result", {})
                if automatic_test_coder:
                    evaluation = (
                        result.get("coder_context_experiment", {})
                        .get("variants", {}).get(level.value, {})
                        .get("code_evaluation", {})
                    )
                else:
                    evaluation = result.get("code_evaluation", {})
                if evaluation.get("applicable"):
                    evaluations.append(evaluation)
            elapsed = sorted(case["elapsed_seconds"] for case in group_cases)
            p90_index = max(0, math.ceil(0.9 * len(elapsed)) - 1)
            outcomes = {
                outcome: sum(case["selection_outcome"] == outcome for case in group_cases)
                for outcome in ("exact", "partial", "over_selection", "miss")
            }
            expansion_count = sum(case["expansion_used"] for case in group_cases)
            table_count_analysis[bucket] = {
                "case_count": len(group_cases),
                "mean_gold_table_count": sum(case["gold_table_count"] for case in group_cases) / len(group_cases),
                "retrieval": mean_metrics([case["metrics"] for case in group_cases]),
                "selection": mean_metrics([case["selection_metrics"] for case in group_cases]),
                "selection_outcomes": outcomes,
                "code_applicable_count": len(evaluations),
                "exact_result_match_rate": sum(bool(e.get("exact_result_match")) for e in evaluations) / len(evaluations) if evaluations else 0.0,
                "supported_result_rate": sum(bool(e.get("supported_correct")) for e in evaluations) / len(evaluations) if evaluations else 0.0,
                "execution_success_rate": sum(bool(e.get("execution_success")) for e in evaluations) / len(evaluations) if evaluations else 0.0,
                "mean_elapsed_seconds": statistics.mean(elapsed),
                "median_elapsed_seconds": statistics.median(elapsed),
                "p90_elapsed_seconds": elapsed[p90_index],
                "expansion_usage_rate": expansion_count / len(group_cases),
            }
        experiments[label] = {
            "config": {
                **retrieval,
                "mode": retrieval["mode"],
                "fusion_method": retrieval["fusion_method"],
                "coder_context_level": level.value,
                "semantic_code_judge_model": resolved.get(
                    "semantic_code_judge_model", ""
                ),
            },
            "mean_metrics": mean_metrics(metric_rows),
            "mean_metrics_successful_queries": mean_metrics(
                successful_metric_rows
            ),
            "mean_selection_metrics": mean_metrics(selection_metric_rows),
            "expansion_metrics": {
                "query_count": expansion_query_count,
                "gold_selected_beyond_initial_count": expansion_contribution_count,
                "gold_selected_beyond_initial_rate": (
                    expansion_contribution_count / expansion_query_count
                    if expansion_query_count else 0.0
                ),
            },
            "successful_case_count": len(successful_metric_rows),
            "failed_case_count": len(case_rows) - len(successful_metric_rows),
            "code_metrics": code_metrics,
            "table_count_analysis": table_count_analysis,
            "performance_metrics": performance_metrics,
            "cases": case_rows,
        }
    report = {
        "benchmark_type": "batch-table-selection",
        "created_at": _now(),
        "case_count": len(case_rows),
        "experiments": experiments,
    }
    if append_log:
        append_benchmark_metrics_log(
            report,
            BASE_DIR / "logs" / "retrieval_benchmarks_log.csv",
            run_id=job_id,
            core=str(resolved["core"]),
            source_path=questions[0]["source_path"],
            source_job_ids={label: job_id for label in experiments},
            model=str(resolved.get("model") or ""),
            architecture=str(resolved.get("discovery_architecture") or ""),
            portal_name=str(
                resolved.get("portal_name")
                or SOLR_CORE_PORTAL_NAMES.get(str(resolved["core"]), resolved["core"])
            ),
        )
    experiment_report = next(iter(experiments.values()))
    return {
        "case_count": len(case_rows),
        "successful_case_count": experiment_report["successful_case_count"],
        "failed_case_count": experiment_report["failed_case_count"],
        "mean_metrics": experiment_report["mean_metrics"],
        "mean_metrics_successful_queries": experiment_report[
            "mean_metrics_successful_queries"
        ],
        "mean_selection_metrics": experiment_report["mean_selection_metrics"],
        "expansion_metrics": experiment_report["expansion_metrics"],
        "table_count_analysis": experiment_report["table_count_analysis"],
        "performance_metrics": experiment_report["performance_metrics"],
    }


def _snapshot_job(job_id: str, *, include_results: bool = True) -> dict[str, Any] | None:
    if not _valid_job_id(job_id):
        return None
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            snapshot = job if include_results else {
                key: value for key, value in job.items() if key != "results"
            }
            return json.loads(json.dumps(snapshot))
    path = _job_path(job_id)
    if path.is_file():
        job = json.loads(path.read_text(encoding="utf-8"))
        if include_results:
            results_path = _results_path(job_id)
            job["results"] = []
            if results_path.is_file():
                for line in results_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        job["results"].append(json.loads(line))
        return job
    return None


def _update_job(job_id: str, **changes: Any) -> None:
    with _jobs_lock:
        job = _jobs[job_id]
        job.update(changes)
        job["updated_at"] = _now()
        snapshot = json.loads(json.dumps(job))
    _persist_job(snapshot)


def _batch_source_key(source: dict[str, Any]) -> str:
    """Return a stable per-question key even when benchmark IDs are absent."""
    source_id = source.get("source_id")
    return str(source_id) if source_id is not None else str(source.get("source_path"))


def _run_batch(job_id: str, questions: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    with _jobs_lock:
        current = _jobs[job_id]
        existing_results = list(current.get("results", []))
        started_at = current.get("started_at") or _now()
    attempts: dict[str, int] = {}
    for entry in existing_results:
        source_key = _batch_source_key(entry)
        attempts[source_key] = attempts.get(source_key, 0) + 1
    completed_source_ids = set(attempts)
    pending_questions = [
        source for source in questions
        if _batch_source_key(source) not in completed_source_ids
    ]
    _update_job(
        job_id,
        status="running",
        started_at=started_at,
        processed=len(completed_source_ids),
    )
    try:
        config = ExperimentConfig.model_validate(settings["resolved_config"])
        runner = ExperimentRunner(config)
        nltk_error = bootstrap_nltk_data()
        if nltk_error:
            raise RuntimeError(nltk_error)
        retrieval_config = getattr(config, "retrieval", None)
        if (
            retrieval_config is not None
            and RetrievalMode(retrieval_config.mode)
            in (RetrievalMode.SEMANTIC, RetrievalMode.HYBRID)
        ):
            health = check_embedding_health(
                retrieval_config.embedding_model,
                retrieval_config.embedding_base_url,
            )
            logger.info("Embedding health check passed: %s", health)

        for source in pending_questions:
            source_key = _batch_source_key(source)
            execution_attempt = attempts.get(source_key, 0) + 1
            with _workflow_lock:
                query_result = runner.run(
                    source["question"],
                    question_id=source["source_id"],
                    log_context={
                        "JOB_ID": job_id,
                        "EXECUTION_ATTEMPT": execution_attempt,
                        "IS_FINAL_ATTEMPT": True,
                        **source["log_fields"],
                    },
                ).to_dict()
            entry = {
                "question": source["question"],
                "source_path": source["source_path"],
                "source_id": source["source_id"],
                "result": query_result,
            }
            _append_result(job_id, entry)
            with _jobs_lock:
                job = _jobs[job_id]
                job["results"].append(entry)
                completed_source_ids.add(source_key)
                job["processed"] = len(completed_source_ids)
                if query_result["status"] != "completed":
                    job["failed"] += 1
                job["updated_at"] = _now()
                snapshot = json.loads(json.dumps(job))
            _persist_job(snapshot)

        with _jobs_lock:
            completed_results = json.loads(json.dumps(_jobs[job_id]["results"]))
            metrics_logged = bool(_jobs[job_id].get("metrics_logged"))
            existing_batch_metrics = json.loads(json.dumps(
                _jobs[job_id].get("batch_metrics", {})
            ))
        table_metrics = None
        if not metrics_logged:
            try:
                table_metrics = _append_batch_table_metrics(
                    job_id, questions, completed_results, settings
                )
                if table_metrics:
                    _update_job(job_id, metrics_logged=True)
            except Exception:
                logger.exception("Could not persist batch table-selection metrics")
        elif existing_batch_metrics:
            table_metrics = existing_batch_metrics.get("table_selection")
        automatic_test_coder = bool(
            settings["resolved_config"].get("automatic_test_coder")
        )
        code_metrics = (
            {
                level.value: summarize_code_evaluations(
                    completed_results, coder_context_level=level.value
                )
                for level in CoderContextLevel
            }
            if automatic_test_coder
            else summarize_code_evaluations(completed_results)
        )
        _update_job(
            job_id,
            status="completed",
            finished_at=_now(),
            batch_metrics={
                "table_selection": table_metrics,
                "code": code_metrics,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=_now(),
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/query")
def query_lakegen(request: QueryRequest) -> dict[str, Any]:
    try:
        explicit_fields = set(request.model_fields_set)
        if request.config is None:
            explicit_fields.update({
                "core", "model", "retrieval_mode", "top_k", "hybrid_alpha",
                "candidate_multiplier", "discovery_architecture", "experiment_id",
                "seed",
            })
        config = _resolve_api_config(
            core=request.core,
            model=request.model,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            hybrid_alpha=request.hybrid_alpha,
            candidate_multiplier=request.candidate_multiplier,
            discovery_architecture=request.discovery_architecture,
            experiment_id=request.experiment_id,
            seed=request.seed,
            config_data=request.config,
            explicit_fields=explicit_fields,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    nltk_error = bootstrap_nltk_data()
    if nltk_error:
        raise HTTPException(status_code=503, detail=nltk_error)
    with _workflow_lock:
        source = QuestionSource(
            question=request.question,
            path="$.question",
            source_data=request.model_dump(mode="json"),
        )
        return ExperimentRunner(config).run(
            request.question,
            log_context=source.log_fields(),
            runtime_factory=make_runtime_settings,
            executor=run_question,
        ).to_dict()


@app.post(
    "/v1/batches",
    response_model=BatchAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_batch(
    payload: Annotated[
        Any,
        Body(
            description=(
                "A queries_old-style JSON document, a list of question objects, "
                "a simple list of question strings, or an envelope with an inline "
                "config object and a questions/queries list."
            )
        ),
    ],
    core: Annotated[str | None, Query()] = None,
    model: Annotated[str | None, Query()] = None,
    retrieval_mode: Annotated[RetrievalMode | None, Query()] = None,
    top_k: Annotated[int | None, Query(ge=1, le=1000)] = None,
    hybrid_alpha: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    candidate_multiplier: Annotated[int | None, Query(ge=1, le=100)] = None,
    discovery_architecture: Annotated[DiscoveryArchitecture | None, Query()] = None,
    experiment_id: Annotated[str | None, Query(min_length=1)] = None,
    seed: Annotated[int | None, Query(ge=0)] = None,
) -> BatchAccepted:
    try:
        inline_config = _inline_batch_config(payload)
        supplied = {
            name for name, value in {
                "core": core,
                "model": model,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "hybrid_alpha": hybrid_alpha,
                "candidate_multiplier": candidate_multiplier,
                "discovery_architecture": discovery_architecture,
                "experiment_id": experiment_id,
                "seed": seed,
            }.items() if value is not None
        }
        if inline_config is None:
            supplied.update({
                "core", "model", "retrieval_mode", "top_k", "hybrid_alpha",
                "candidate_multiplier", "discovery_architecture", "experiment_id",
                "seed",
            })
        resolved_config = _resolve_api_config(
            core=core or DEFAULT_CORE,
            model=model or DEFAULT_MODEL,
            retrieval_mode=retrieval_mode or RetrievalMode.KEYWORD,
            top_k=top_k if top_k is not None else DEFAULT_TOP_K,
            hybrid_alpha=hybrid_alpha if hybrid_alpha is not None else 0.5,
            candidate_multiplier=candidate_multiplier or 5,
            discovery_architecture=discovery_architecture or DiscoveryArchitecture.UNIFIED,
            experiment_id=experiment_id or "default",
            seed=seed if seed is not None else 0,
            config_data=inline_config,
            explicit_fields=supplied,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _enqueue_batch(payload, resolved_config)


@app.get("/v1/benchmarks")
def list_benchmarks() -> dict[str, list[str]]:
    """List the JSON benchmark inputs bundled with this LakeGen instance."""

    if not BENCHMARK_DIR.is_dir():
        return {"benchmarks": []}
    return {"benchmarks": sorted(path.name for path in BENCHMARK_DIR.glob("*.json"))}


@app.post(
    "/v1/benchmarks/{benchmark_name}/batches",
    response_model=BatchAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_benchmark_batch(
    benchmark_name: str,
    core: Annotated[str | None, Query()] = None,
    model: Annotated[str | None, Query()] = None,
    retrieval_mode: Annotated[RetrievalMode | None, Query()] = None,
    top_k: Annotated[int | None, Query(ge=1, le=1000)] = None,
    hybrid_alpha: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    candidate_multiplier: Annotated[int | None, Query(ge=1, le=100)] = None,
    discovery_architecture: Annotated[DiscoveryArchitecture | None, Query()] = None,
    experiment_id: Annotated[str | None, Query(min_length=1)] = None,
    seed: Annotated[int | None, Query(ge=0)] = None,
) -> BatchAccepted:
    """Queue one catalog benchmark while retaining every query metadata field."""

    payload = _load_benchmark(benchmark_name)
    try:
        supplied = {
            name for name, value in {
                "core": core,
                "model": model,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "hybrid_alpha": hybrid_alpha,
                "candidate_multiplier": candidate_multiplier,
                "discovery_architecture": discovery_architecture,
                "experiment_id": experiment_id,
                "seed": seed,
            }.items() if value is not None
        }
        resolved_config = _resolve_api_config(
            core=core or DEFAULT_CORE,
            model=model or DEFAULT_MODEL,
            retrieval_mode=retrieval_mode or RetrievalMode.KEYWORD,
            top_k=top_k if top_k is not None else DEFAULT_TOP_K,
            hybrid_alpha=hybrid_alpha if hybrid_alpha is not None else 0.5,
            candidate_multiplier=candidate_multiplier or 5,
            discovery_architecture=(
                discovery_architecture or DiscoveryArchitecture.UNIFIED
            ),
            experiment_id=experiment_id or f"benchmark-{Path(benchmark_name).stem}",
            seed=seed if seed is not None else 0,
            config_data=_inline_batch_config(payload),
            explicit_fields=supplied | {"experiment_id"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _enqueue_batch(payload, resolved_config)


def _enqueue_batch(payload: Any, resolved_config: ExperimentConfig) -> BatchAccepted:
    """Validate questions and enqueue a batch using an already resolved config."""

    extracted = extract_questions(payload)
    if not extracted:
        raise HTTPException(
            status_code=422,
            detail="No questions found. Expected question fields or a questions/queries list.",
        )
    if len(extracted) > MAX_BATCH_QUESTIONS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch has {len(extracted)} questions; maximum is {MAX_BATCH_QUESTIONS}.",
        )

    job_id = uuid.uuid4().hex
    questions = [
        {
            "question": item.question,
            "source_path": item.path,
            "source_id": item.source_id,
            "log_fields": item.log_fields(),
        }
        for item in extracted
    ]
    now = _now()
    job: dict[str, Any] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "question_count": len(questions),
        "processed": 0,
        "failed": 0,
        "metrics_logged": False,
        "settings": {
            "resolved_config": resolved_config.model_dump(mode="json"),
        },
        "results": [],
        "error": "",
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _persist_questions(job_id, questions)
    _persist_job(job)
    _executor.submit(_run_batch, job_id, questions, job["settings"])

    return BatchAccepted(
        job_id=job_id,
        status="queued",
        question_count=len(questions),
        status_url=f"/v1/batches/{job_id}",
    )


def _recover_incomplete_jobs() -> list[str]:
    """Resume persisted queued/running jobs from their last durable result."""

    if not JOB_DIR.is_dir():
        return []
    recovered: list[str] = []
    for path in sorted(JOB_DIR.glob("*.json")):
        job_id = path.stem
        if not _valid_job_id(job_id):
            continue
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("status") not in {"queued", "running"}:
                continue
            questions = _load_questions(job_id)
            if not questions:
                logger.warning(
                    "Cannot resume job %s because its question checkpoint is missing",
                    job_id,
                )
                continue
            job = _snapshot_job(job_id) or metadata
            results = job.get("results", [])
            unique_results = {
                _batch_source_key(entry): entry for entry in results
            }
            job["results"] = list(unique_results.values())
            job["processed"] = len(unique_results)
            job["failed"] = sum(
                entry.get("result", {}).get("status") != "completed"
                for entry in unique_results.values()
            )
            job["status"] = "queued"
            job["updated_at"] = _now()
            with _jobs_lock:
                _jobs[job_id] = job
            _persist_job(job)
            _executor.submit(_run_batch, job_id, questions, job["settings"])
            recovered.append(job_id)
        except Exception:
            logger.exception("Could not recover batch job %s", job_id)
    return recovered


@app.on_event("startup")
def recover_incomplete_jobs() -> None:
    _recover_incomplete_jobs()


@app.post(
    "/v1/batches/files",
    response_model=BatchAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_batch_files(
    config: Annotated[
        UploadFile,
        File(description="Experiment configuration in YAML or JSON format."),
    ],
    questions: Annotated[
        UploadFile,
        File(description="Question batch in any JSON shape accepted by /v1/batches."),
    ],
) -> BatchAccepted:
    """Upload configuration and questions as two independent multipart files."""

    try:
        config_suffix = _upload_suffix(
            config, {".json", ".yaml", ".yml"}, "config"
        )
        _upload_suffix(questions, {".json"}, "questions")
        config_content = await _read_upload(
            config,
            limit=MAX_CONFIG_UPLOAD_BYTES,
            label="config",
        )
        questions_content = await _read_upload(
            questions,
            limit=MAX_QUESTIONS_UPLOAD_BYTES,
            label="questions",
        )
        try:
            config_text = config_content.decode("utf-8")
            questions_text = questions_content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("uploaded files must be UTF-8 encoded") from exc
        config_data = parse_experiment_config_document(
            config_text,
            suffix=config_suffix,
        )
        try:
            questions_payload = json.loads(questions_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid questions JSON: {exc}") from exc
        resolved_config = _resolve_api_config(
            core=DEFAULT_CORE,
            model=DEFAULT_MODEL,
            retrieval_mode=RetrievalMode.KEYWORD,
            top_k=DEFAULT_TOP_K,
            hybrid_alpha=0.5,
            candidate_multiplier=5,
            config_data=config_data,
            explicit_fields=set(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await config.close()
        await questions.close()

    return _enqueue_batch(questions_payload, resolved_config)


@app.get("/v1/batches/{job_id}")
def get_batch(
    job_id: str,
    include_results: Annotated[bool, Query()] = True,
) -> dict[str, Any]:
    job = _snapshot_job(job_id, include_results=include_results)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return job
