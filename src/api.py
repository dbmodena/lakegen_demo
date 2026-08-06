"""HTTP API for single and batch LakeGen queries.

Run locally with::

    uv run uvicorn src.api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, HTTPException, Query, status
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
from lakegen.ui.state import MODEL_OPTIONS, SOLR_CORE_OPTIONS
from lakegen.retrieval import RetrievalMode
from lakegen.experiment_config import (
    DiscoveryArchitecture,
    ExperimentConfig,
    load_experiment_config,
)
from lakegen.runner import ExperimentRunner


DEFAULT_MODEL = MODEL_OPTIONS[0]
DEFAULT_CORE = SOLR_CORE_OPTIONS[0]
MAX_BATCH_QUESTIONS = 10_000
JOB_DIR = BASE_DIR / ".lakegen_jobs"

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
    top_k: int = Field(default=10, ge=1, le=1000)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_multiplier: int = Field(default=5, ge=1, le=100)
    discovery_architecture: DiscoveryArchitecture = DiscoveryArchitecture.UNIFIED
    experiment_id: str = Field(default="default", min_length=1)
    seed: int = Field(default=0, ge=0)
    config_path: str | None = None
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
    config_path: str | None = None,
    config_data: dict[str, Any] | None = None,
    explicit_fields: set[str] | None = None,
) -> ExperimentConfig:
    explicit = explicit_fields or {
        "core", "model", "retrieval_mode", "top_k", "hybrid_alpha",
        "candidate_multiplier", "discovery_architecture", "experiment_id", "seed",
    }
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
        config_path,
        data_override=config_data,
        overrides=overrides,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _results_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.results.jsonl"


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


def _append_result(job_id: str, result: dict[str, Any]) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    with _results_path(job_id).open("a", encoding="utf-8") as output:
        output.write(json.dumps(result, ensure_ascii=False) + "\n")


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


def _run_batch(job_id: str, questions: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    _update_job(job_id, status="running", started_at=_now())
    try:
        config = ExperimentConfig.model_validate(settings["resolved_config"])
        runner = ExperimentRunner(config)
        nltk_error = bootstrap_nltk_data()
        if nltk_error:
            raise RuntimeError(nltk_error)

        for index, source in enumerate(questions):
            with _workflow_lock:
                query_result = runner.run(
                    source["question"],
                    question_id=source["source_id"],
                    log_context={
                        "JOB_ID": job_id,
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
                job["processed"] = index + 1
                if query_result["status"] != "completed":
                    job["failed"] += 1
                job["updated_at"] = _now()
                snapshot = json.loads(json.dumps(job))
            _persist_job(snapshot)

        _update_job(job_id, status="completed", finished_at=_now())
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
        if request.config_path is None and request.config is None:
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
            config_path=request.config_path,
            config_data=request.config,
            explicit_fields=explicit_fields,
        )
    except (OSError, ValueError) as exc:
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
                "or a simple list of question strings."
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
    config_path: Annotated[str | None, Query()] = None,
) -> BatchAccepted:
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
        if config_path is None:
            supplied.update({
                "core", "model", "retrieval_mode", "top_k", "hybrid_alpha",
                "candidate_multiplier", "discovery_architecture", "experiment_id",
                "seed",
            })
        resolved_config = _resolve_api_config(
            core=core or DEFAULT_CORE,
            model=model or DEFAULT_MODEL,
            retrieval_mode=retrieval_mode or RetrievalMode.KEYWORD,
            top_k=top_k or 10,
            hybrid_alpha=hybrid_alpha if hybrid_alpha is not None else 0.5,
            candidate_multiplier=candidate_multiplier or 5,
            discovery_architecture=discovery_architecture or DiscoveryArchitecture.UNIFIED,
            experiment_id=experiment_id or "default",
            seed=seed if seed is not None else 0,
            config_path=config_path,
            explicit_fields=supplied,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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
        "settings": {
            "resolved_config": resolved_config.model_dump(mode="json"),
        },
        "results": [],
        "error": "",
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _persist_job(job)
    _executor.submit(_run_batch, job_id, questions, job["settings"])

    return BatchAccepted(
        job_id=job_id,
        status="queued",
        question_count=len(questions),
        status_url=f"/v1/batches/{job_id}",
    )


@app.get("/v1/batches/{job_id}")
def get_batch(
    job_id: str,
    include_results: Annotated[bool, Query()] = True,
) -> dict[str, Any]:
    job = _snapshot_job(job_id, include_results=include_results)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found.")
    return job
