"""HTTP API for single and batch LakeGen queries.

Run locally with::

    python -m lakegen.interfaces.api --port 8000

or ``uvicorn lakegen.interfaces.api.app:app``; ``src/api.py`` still answers
to ``uvicorn src.api:app``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lakegen.config.bootstrap import bootstrap_nltk_data
from lakegen.config.experiment import (
    DiscoveryArchitecture,
    ExperimentConfig,
    parse_experiment_config_document,
)
from lakegen.config.paths import BASE_DIR
from lakegen.config.runtime import (
    MODEL_OPTIONS,
    SOLR_CORE_OPTIONS,
)
from lakegen.domain.tables import resource_id, table_id
from lakegen.experiments import batch, jobs
from lakegen.experiments.analysis.results import final_ranking
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalMode
from lakegen.workflow.pipeline.runner import ExperimentRunner
from lakegen.workflow.pipeline.service import (
    QuestionSource,
    extract_questions,
    make_runtime_settings,
    run_question,
)

DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_CORE = SOLR_CORE_OPTIONS[0]
MAX_BATCH_QUESTIONS = 10_000
MAX_CONFIG_UPLOAD_BYTES = 1_000_000
MAX_QUESTIONS_UPLOAD_BYTES = 25_000_000
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


# The normalisers and the ranking reader moved to the package; the old names
# stay for the analyses that call them (experiments/paper_analysis/records.py).
_table_id = table_id
_resource_id = resource_id
_retrieval_ranking = final_ranking


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
        config = batch.resolve_config(
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
    with batch.WORKFLOW_LOCK:
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
        inline_config = batch.inline_config(payload)
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
        resolved_config = batch.resolve_config(
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
        resolved_config = batch.resolve_config(
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
        resolved_config = batch.resolve_config(
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
    job = jobs.snapshot_job(job_id, include_results=include_results)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return job
