"""HTTP API for single and batch LakeGen queries.

Run locally with::

    uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
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
from lakegen.config.settings import (
    get_settings, non_negative_float_or_default, positive_int_or_default, read_var,
)
from lakegen.core import paths
from lakegen.core.config import BASE_DIR, LOG_DIR
from lakegen.service import (
    QuestionSource,
    extract_questions,
    make_runtime_settings,
    run_question,
)
from lakegen.runtime import (
    MODEL_OPTIONS,
    SOLR_CORE_OPTIONS,
    SOLR_CORE_PORTAL_NAMES,
)
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalMode, check_embedding_health
from lakegen.retrieval.benchmark import append_benchmark_metrics_log
from lakegen.analysis.metrics import batch_cases, batch_report, report_summary
from lakegen.analysis.results import final_ranking
from lakegen.code_evaluation import summarize_code_evaluations
from lakegen.domain.tables import resource_id, table_id
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
# LAKEGEN_JOB_DIR gives an API instance its own job folder, so a second run
# neither shows nor resumes the jobs another instance left unfinished.
JOB_DIR = paths.jobs_dir()
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


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive worker limit without making API startup fragile."""

    return read_var(name, positive_int_or_default, default)


# A workflow may already parallelise table scans internally. Keep the safe
# historical default, while allowing a bounded number of questions/batches in flight.
WORKFLOW_WORKERS = _positive_int_from_env("LAKEGEN_WORKFLOW_WORKERS", 1)
_executor = ThreadPoolExecutor(
    max_workers=WORKFLOW_WORKERS, thread_name_prefix="lakegen-batch"
)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_workflow_lock = threading.BoundedSemaphore(WORKFLOW_WORKERS)


def _non_negative_float_from_env(name: str, default: float) -> float:
    return read_var(name, non_negative_float_or_default, default)


# A breather inside long batches: after every LAKEGEN_BATCH_PAUSE_EVERY finished
# questions a job sleeps LAKEGEN_BATCH_PAUSE_SECONDS before its next question,
# which keeps a run of hundreds of questions under the provider's rate limits.
# Either value at 0 (the default) disables it. The pause is taken outside the
# workflow lock, so it never holds a slot another job's question could use.
BATCH_PAUSE_EVERY = int(_non_negative_float_from_env("LAKEGEN_BATCH_PAUSE_EVERY", 0))
BATCH_PAUSE_SECONDS = _non_negative_float_from_env("LAKEGEN_BATCH_PAUSE_SECONDS", 0.0)
# A rest sized by usage: after each question a job sleeps as long as that
# question's tokens take at LAKEGEN_BATCH_TOKENS_PER_MINUTE, so the job's
# average rate stays at that figure however heavy single questions are. It adds
# to the fixed pause above; 0 (the default) disables it.
BATCH_TOKENS_PER_MINUTE = _non_negative_float_from_env("LAKEGEN_BATCH_TOKENS_PER_MINUTE", 0.0)
# A file holding the rate overrides it and is read before every rest, so a long
# run can be sped up or slowed down without a restart; a missing or unreadable
# file falls back to the variable.
BATCH_TOKENS_PER_MINUTE_FILE = get_settings().batch_tokens_per_minute_file


def _tokens_per_minute() -> float:
    """The batch token rate now: the rate file's number, else the variable."""
    if BATCH_TOKENS_PER_MINUTE_FILE:
        try:
            value = float(Path(BATCH_TOKENS_PER_MINUTE_FILE).read_text().strip())
        except (OSError, ValueError):
            return BATCH_TOKENS_PER_MINUTE
        if value >= 0:
            return value
    return BATCH_TOKENS_PER_MINUTE


def _batch_pause_seconds(
    finished_in_run: int,
    remaining: int,
    *,
    every: int | None = None,
    seconds: float | None = None,
) -> float:
    """How long a batch rests now: ``seconds`` after each ``every`` questions, never after the last."""
    every = BATCH_PAUSE_EVERY if every is None else every
    seconds = BATCH_PAUSE_SECONDS if seconds is None else seconds
    if every <= 0 or seconds <= 0 or finished_in_run <= 0 or remaining <= 0:
        return 0.0
    return float(seconds) if finished_in_run % every == 0 else 0.0


def _question_tokens(result: dict[str, Any]) -> int:
    """LLM tokens one finished question spent: every phase plus the code judge.

    The judge is counted per coder-context variant when the variants ran, since
    each variant is judged; the top-level evaluation repeats one of them.
    """
    spent = sum(int(value or 0) for value in (result.get("tokens") or {}).values())
    variants = (result.get("coder_context_experiment") or {}).get("variants") or {}
    evaluations = (
        [variant.get("code_evaluation") or {} for variant in variants.values()]
        if variants
        else [result.get("code_evaluation") or {}]
    )
    return spent + sum(
        int(evaluation.get("semantic_judge_tokens") or 0) for evaluation in evaluations
    )


def _token_pause_seconds(
    tokens: int, remaining: int, *, per_minute: float | None = None
) -> float:
    """How long ``tokens`` take at the batch token rate; never after the last question."""
    per_minute = _tokens_per_minute() if per_minute is None else per_minute
    if per_minute <= 0 or tokens <= 0 or remaining <= 0:
        return 0.0
    return tokens * 60.0 / per_minute


# A share of a batch's questions on a second model: with LAKEGEN_SPLIT_MODEL set
# and LAKEGEN_SPLIT_SHARE above 0, a question whose id hashes below the share runs
# with that model wherever the job's config names the job's model (pipeline, code
# judge, Pneuma entity extraction). The hash depends on the question id only, so
# a question gets the same model in every job of a grid. Off by default.
SPLIT_MODEL = get_settings().split_model
SPLIT_SHARE = get_settings().split_share


def _split_fraction(source_id: Any) -> float:
    """A stable number in [0, 1) for a question id."""
    digest = hashlib.sha256(str(source_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def _swap_model(value: Any, model: str, replacement: str) -> Any:
    """``value`` with every string equal to ``model`` replaced, at any depth."""
    if isinstance(value, dict):
        return {key: _swap_model(item, model, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [_swap_model(item, model, replacement) for item in value]
    return replacement if value == model else value


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


# The normalisers and the ranking reader moved to the package; the old names
# stay for the analyses that call them (experiments/paper_analysis/records.py).
_table_id = table_id
_resource_id = resource_id
_retrieval_ranking = final_ranking


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

    cases = batch_cases(questions, results)
    if cases is None:
        return None
    resolved = settings["resolved_config"]
    report = batch_report(cases, results, resolved, created_at=_now())
    if append_log:
        append_benchmark_metrics_log(
            report,
            LOG_DIR / "retrieval_benchmarks_log.csv",
            run_id=job_id,
            core=str(resolved["core"]),
            source_path=questions[0]["source_path"],
            source_job_ids={label: job_id for label in report["experiments"]},
            model=str(resolved.get("model") or ""),
            architecture=str(resolved.get("discovery_architecture") or ""),
            portal_name=str(
                resolved.get("portal_name")
                or SOLR_CORE_PORTAL_NAMES.get(str(resolved["core"]), resolved["core"])
            ),
        )
    return report_summary(report)


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
        job_model = settings["resolved_config"].get("model")
        split_runner = None
        if SPLIT_MODEL and SPLIT_SHARE > 0 and job_model and SPLIT_MODEL != job_model:
            split_runner = ExperimentRunner(ExperimentConfig.model_validate(
                _swap_model(settings["resolved_config"], job_model, SPLIT_MODEL)
            ))
            logger.info(
                "Batch %s: %d of %d pending questions on %s, the rest on %s",
                job_id,
                sum(_split_fraction(q["source_id"]) < SPLIT_SHARE for q in pending_questions),
                len(pending_questions), SPLIT_MODEL, job_model,
            )
        nltk_error = bootstrap_nltk_data()
        if nltk_error:
            raise RuntimeError(nltk_error)
        retrieval_config = getattr(config, "retrieval", None)
        if (
            retrieval_config is not None
            and RetrievalMode(retrieval_config.mode).uses_embeddings
        ):
            health = check_embedding_health(
                retrieval_config.embedding_model,
                retrieval_config.embedding_base_url,
            )
            logger.info("Embedding health check passed: %s", health)

        finished_in_run = 0
        for source in pending_questions:
            source_key = _batch_source_key(source)
            execution_attempt = attempts.get(source_key, 0) + 1
            on_split = (
                split_runner is not None
                and _split_fraction(source["source_id"]) < SPLIT_SHARE
            )
            with _workflow_lock:
                query_result = (split_runner if on_split else runner).run(
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
            if split_runner is not None:
                entry["model"] = SPLIT_MODEL if on_split else job_model
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
            finished_in_run += 1
            remaining = len(pending_questions) - finished_in_run
            tokens = _question_tokens(query_result)
            pause = _batch_pause_seconds(finished_in_run, remaining) + _token_pause_seconds(
                tokens, remaining
            )
            if pause:
                logger.info(
                    "Batch %s: pausing %.0fs after %d questions (last one spent %d tokens)",
                    job_id, pause, finished_in_run, tokens,
                )
                time.sleep(pause)

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

    def submitted_at(path: Path) -> tuple[str, str]:
        try:
            created = json.loads(path.read_text(encoding="utf-8")).get("created_at")
        except (OSError, json.JSONDecodeError):
            created = None
        return str(created or ""), path.stem

    # Oldest first, so the queue a restart rebuilds is the queue that was
    # submitted; file names are random ids and say nothing about order.
    job_paths = [path for path in JOB_DIR.glob("*.json") if _valid_job_id(path.stem)]
    for path in sorted(job_paths, key=submitted_at):
        job_id = path.stem
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
