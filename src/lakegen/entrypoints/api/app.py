"""HTTP API for LakeGen: answer one question.

Run locally with::

    python -m lakegen.entrypoints.api --port 8000

or ``uvicorn lakegen.entrypoints.api.app:app``; ``src/api.py`` still answers
to ``uvicorn src.api:app``. Batches of questions are the experiment harness's
(``experiment_runner.batches``), whose server includes this ``router``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lakegen.config.experiment import DiscoveryArchitecture
from lakegen.config.startup import bootstrap_nltk_data
from lakegen.entrypoints.api.submission import (
    DEFAULT_CORE,
    DEFAULT_MODEL,
    WORKFLOW_LOCK,
    resolve_config,
)
from lakegen.pipeline.runner import ExperimentRunner
from lakegen.pipeline.service import (
    QuestionSource,
    make_runtime_settings,
    run_question,
)
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalMode

# Every field of a query; a query without an inline config sets them all.
QUERY_FIELDS = frozenset(
    {
        "core",
        "model",
        "retrieval_mode",
        "top_k",
        "hybrid_alpha",
        "candidate_multiplier",
        "discovery_architecture",
        "experiment_id",
        "seed",
    }
)

router = APIRouter()


class StrictModel(BaseModel):
    """A request or response body that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class QueryRequest(StrictModel):
    """One question to answer, with the configuration to answer it with.

    Only the fields set explicitly override the inline ``config``; without a
    ``config``, every field applies.

    Attributes:
        question: The question, stripped; it must not be blank.
        core: The Solr core to search.
        model: The pipeline's model.
        retrieval_mode: The retrieval mode (``retrieval.mode``).
        top_k: How many tables retrieval returns (``retrieval.top_k``).
        hybrid_alpha: The lexical weight of weighted hybrid fusion (``retrieval.alpha``).
        candidate_multiplier: The per-branch pool size as a multiple of ``top_k``.
        discovery_architecture: The discovery architecture.
        experiment_id: The experiment id the run records.
        seed: The seed.
        config: An inline experiment configuration.
    """

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
        """The question without surrounding whitespace.

        Args:
            value: The question as sent.

        Returns:
            The stripped question.

        Raises:
            ValueError: When the question is blank.
        """
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the API is up."""
    return {"status": "ok"}


@router.post("/v1/query")
def query_lakegen(request: QueryRequest) -> dict[str, Any]:
    """Answer one question now, one query at a time, and return the run's result."""
    try:
        explicit_fields = set(request.model_fields_set)
        if request.config is None:
            explicit_fields.update(QUERY_FIELDS)
        config = resolve_config(
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
    with WORKFLOW_LOCK:
        source = QuestionSource(
            question=request.question,
            path="$.question",
            source_data=request.model_dump(mode="json"),
        )
        return (
            ExperimentRunner(config)
            .run(
                request.question,
                log_context=source.log_fields(),
                runtime_factory=make_runtime_settings,
                executor=run_question,
            )
            .to_dict()
        )


app = FastAPI(
    title="LakeGen API",
    version="1.0.0",
    description="Answer a question over a lake of tables.",
)
app.include_router(router)
