from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Mapping, Sequence


from lakegen.prompts.prompt_manager import PromptManager
from lakegen.retrieval.solr_client import LocalSolrClient, resolve_solr_core
from lakegen.core.table_io import list_table_files
from lakegen.config.settings import get_settings
from lakegen.core.config import LOG_DIR
from lakegen.retrieval import RetrievalConfig, RetrievalRunLogger, TableRetrievalService
from lakegen.retrieval.models import RetrievalRun


# The LLM clients live in lakegen.llm; these two names are still read here by
# lakegen.retrieval.embeddings, lakegen.retrieval.rerank and the surfaces.
from lakegen.llm.factory import get_llm  # noqa: F401
from lakegen.llm.providers.oci import oci_runtime_config as _oci_runtime_config  # noqa: F401

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def get_solr(core):
    return LocalSolrClient(
        core=resolve_solr_core(core),
        base_url=get_settings().solr_base_url,
    )


@lru_cache(maxsize=1)
def get_retrieval_run_logger() -> RetrievalRunLogger:
    return RetrievalRunLogger(LOG_DIR / "retrieval_rankings.jsonl")


_retrieval_capture: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "lakegen_retrieval_capture", default=None
)
_retrieval_context: ContextVar[dict[str, Any]] = ContextVar(
    "lakegen_retrieval_context", default={}
)
_retrieval_attempt: ContextVar[int] = ContextVar(
    "lakegen_retrieval_attempt", default=0
)


@contextmanager
def capture_retrieval_runs(context: Mapping[str, Any] | None = None):
    """Collect retrieval JSON records produced by the current workflow."""

    captured: list[dict[str, Any]] = []
    token = _retrieval_capture.set(captured)
    context_token = _retrieval_context.set(dict(context or {}))
    attempt_token = _retrieval_attempt.set(0)
    try:
        yield captured
    finally:
        _retrieval_capture.reset(token)
        _retrieval_context.reset(context_token)
        _retrieval_attempt.reset(attempt_token)


def _log_and_capture_retrieval_run(run: RetrievalRun) -> None:
    context = _retrieval_context.get()
    attempt = _retrieval_attempt.get() + 1
    _retrieval_attempt.set(attempt)
    run.job_id = context.get("JOB_ID")
    run.source_path = context.get("SOURCE_PATH")
    run.source_id = context.get("SOURCE_ID")
    run.execution_attempt = context.get("EXECUTION_ATTEMPT")
    run.experiment_id = context.get("EXPERIMENT_ID")
    run.retrieval_attempt = attempt
    get_retrieval_run_logger()(run)
    captured = _retrieval_capture.get()
    if captured is not None:
        payload = run.to_log_dict()
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        captured.append(payload)


def make_retrieval_run_observer(
    context: Mapping[str, Any] | None,
    captured: list[dict[str, Any]],
) -> Callable[[RetrievalRun], None]:
    """Build a thread-safe observer with explicit workflow context.

    Agent tools may run in a worker context where ``ContextVar`` values are not
    propagated.  Binding the context and destination here keeps agentic
    retrieval telemetry attached to the correct job without changing retrieval.
    """

    bound_context = dict(context or {})
    lock = Lock()
    attempt = 0

    def observe(run: RetrievalRun) -> None:
        nonlocal attempt
        with lock:
            attempt += 1
            run.job_id = bound_context.get("JOB_ID")
            run.source_path = bound_context.get("SOURCE_PATH")
            run.source_id = bound_context.get("SOURCE_ID")
            run.execution_attempt = bound_context.get("EXECUTION_ATTEMPT")
            run.experiment_id = bound_context.get("EXPERIMENT_ID")
            run.retrieval_attempt = attempt
            get_retrieval_run_logger()(run)
            payload = run.to_log_dict()
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            captured.append(payload)

    return observe


def log_retrieval_decision(
    *,
    question: str,
    selected_tables: list[str],
    reason: str,
    context: Mapping[str, Any] | None = None,
    mode: str | None = None,
    keywords: Sequence[str] = (),
    retrieval_attempt: int | None = None,
) -> None:
    context = dict(context or {})
    get_retrieval_run_logger().log_payload(
        {
            "event_type": "table_selection",
            "question": question,
            "job_id": context.get("JOB_ID"),
            "source_path": context.get("SOURCE_PATH"),
            "source_id": context.get("SOURCE_ID"),
            "execution_attempt": context.get("EXECUTION_ATTEMPT"),
            "experiment_id": context.get("EXPERIMENT_ID"),
            "mode": mode,
            "keywords": list(keywords),
            "retrieval_attempt": retrieval_attempt,
            "selected_tables": list(selected_tables),
            "decision": "accepted" if selected_tables else "rejected",
            "reason": reason,
        }
    )


def get_table_retrieval_service(
    solr: LocalSolrClient,
    config: RetrievalConfig,
    table_dir: str | Path | None = None,
    *,
    observer: Callable[[RetrievalRun], None] | None = None,
) -> TableRetrievalService:
    return TableRetrievalService(
        solr,
        config,
        observer=observer or _log_and_capture_retrieval_run,
        table_dir=str(table_dir) if table_dir is not None else None,
    )


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    return PromptManager()


def get_all_table_files(table_dir):
    return list_table_files(table_dir)


# Backward-compatible name for callers outside this repository.
get_all_csv_files = get_all_table_files
