"""The log of the retrieval runs of a question, and the capture of them for its record.

Every run of a ``TableRetrievalService`` reaches ``log_and_capture_retrieval_run``
(the service's default observer): it stamps the run with the current job and
question (set with ``capture_retrieval_runs``), appends it to the lake-wide
``retrieval_rankings.jsonl`` and keeps a copy for the question's result.
``make_retrieval_run_observer`` binds that context explicitly for the agent
tools, which run on threads where the context variables do not reach.
``log_retrieval_decision`` records which of the retrieved tables a run kept.

Experiment scripts that must not touch the production log replace
``log_and_capture_retrieval_run`` on this module; the factory reads it at
call time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from threading import Lock
from typing import Any

from lakegen.config.locations import LOG_DIR
from lakegen.retrieval.evaluation import RetrievalRunLogger
from lakegen.retrieval.models import RetrievalRun


@lru_cache(maxsize=1)
def get_retrieval_run_logger() -> RetrievalRunLogger:
    return RetrievalRunLogger(LOG_DIR / "retrieval_rankings.jsonl")


_retrieval_capture: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "lakegen_retrieval_capture", default=None
)
_retrieval_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "lakegen_retrieval_context", default=None
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


def log_and_capture_retrieval_run(run: RetrievalRun) -> None:
    context = _retrieval_context.get() or {}
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
        payload["timestamp"] = datetime.now(UTC).isoformat()
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
            payload["timestamp"] = datetime.now(UTC).isoformat()
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
