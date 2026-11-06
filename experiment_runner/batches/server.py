"""The batch server: LakeGen's API plus batches of questions run in the background.

    .venv/bin/python -m experiment_runner.batches --port 8002

It answers what LakeGen's API answers (``/health``, ``/v1/query``) and queues
batches: a JSON list of questions (``/v1/batches``), a benchmark file of
``benchmark/`` (``/v1/benchmarks/{name}/batches``) or an uploaded config and
question file (``/v1/batches/files``); ``/v1/batches/{job_id}`` follows a job.
Jobs a previous process left unfinished resume at startup. Experiment specs
submit their cells here (``execution.api_url``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, status

from experiment_runner.batches import batch, jobs
from lakegen.config.experiment import (
    DiscoveryArchitecture,
    ExperimentConfig,
    parse_experiment_config_document,
)
from lakegen.config.paths import BASE_DIR
from lakegen.entrypoints.api.app import QUERY_FIELDS, StrictModel, router
from lakegen.entrypoints.api.submission import DEFAULT_CORE, DEFAULT_MODEL, resolve_config
from lakegen.pipeline.service import extract_questions
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalMode

MAX_BATCH_QUESTIONS = 10_000
MAX_CONFIG_UPLOAD_BYTES = 1_000_000
MAX_QUESTIONS_UPLOAD_BYTES = 25_000_000
BENCHMARK_DIR = BASE_DIR / "benchmark"
# Uvicorn's application logger, so batch progress shows in the same terminal
# as the HTTP access log.
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="LakeGen batch server",
    version="1.0.0",
    description="Query LakeGen directly or submit JSON question batches.",
)
app.include_router(router)


class BatchAccepted(StrictModel):
    """The reply to a batch submission: the queued job and where to follow it.

    Attributes:
        job_id: The batch job's id.
        status: The job's status (``"queued"``).
        question_count: How many questions the job holds.
        status_url: The URL that reports the job's progress and results.
    """

    job_id: str
    status: str
    question_count: int
    status_url: str


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
    """Queue a batch of questions; query parameters override the batch's inline config."""
    try:
        inline_config = batch.inline_config(payload)
        supplied = {
            name
            for name, value in {
                "core": core,
                "model": model,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "hybrid_alpha": hybrid_alpha,
                "candidate_multiplier": candidate_multiplier,
                "discovery_architecture": discovery_architecture,
                "experiment_id": experiment_id,
                "seed": seed,
            }.items()
            if value is not None
        }
        if inline_config is None:
            supplied.update(QUERY_FIELDS)
        resolved_config = resolve_config(
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
            name
            for name, value in {
                "core": core,
                "model": model,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "hybrid_alpha": hybrid_alpha,
                "candidate_multiplier": candidate_multiplier,
                "discovery_architecture": discovery_architecture,
                "experiment_id": experiment_id,
                "seed": seed,
            }.items()
            if value is not None
        }
        resolved_config = resolve_config(
            core=core or DEFAULT_CORE,
            model=model or DEFAULT_MODEL,
            retrieval_mode=retrieval_mode or RetrievalMode.KEYWORD,
            top_k=top_k if top_k is not None else DEFAULT_TOP_K,
            hybrid_alpha=hybrid_alpha if hybrid_alpha is not None else 0.5,
            candidate_multiplier=candidate_multiplier or 5,
            discovery_architecture=(discovery_architecture or DiscoveryArchitecture.UNIFIED),
            experiment_id=experiment_id or f"benchmark-{Path(benchmark_name).stem}",
            seed=seed if seed is not None else 0,
            config_data=batch.inline_config(payload),
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

    questions = [
        {
            "question": item.question,
            "source_path": item.path,
            "source_id": item.source_id,
            "log_fields": item.log_fields(),
        }
        for item in extracted
    ]
    job = batch.enqueue(questions, resolved_config)
    return BatchAccepted(
        job_id=job["job_id"],
        status="queued",
        question_count=len(questions),
        status_url=f"/v1/batches/{job['job_id']}",
    )


@app.on_event("startup")
def recover_incomplete_jobs() -> None:
    """Resume the batch jobs a previous process left unfinished."""
    batch.recover_incomplete_jobs()


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
        config_suffix = _upload_suffix(config, {".json", ".yaml", ".yml"}, "config")
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
        resolved_config = resolve_config(
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
    """Report a batch job's progress, with its results unless told otherwise."""
    job = jobs.snapshot_job(job_id, include_results=include_results)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return job
