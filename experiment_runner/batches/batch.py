"""The batch loop: every question of a job through the autonomous workflow, paced and recorded.

``run_batch`` answers the pending questions of a job one by one (resuming
after the results already on disk), rests between questions as the pacing
settings say, splits a share of them onto a second model when configured,
and closes the job with its table-selection and code metrics. ``enqueue``
creates a job and queues its run on the worker pool; ``recover_incomplete_jobs``
re-queues the jobs a previous process left queued or running.
A submission's configuration is resolved as LakeGen's API resolves a direct
query (``lakegen.entrypoints.api.submission``), and a batch question takes the
same worker slot a direct query does.

The batch server's endpoints (``server``) and the local executor of
experiment specs both submit through here; nothing here depends on the web
framework.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from experiment_runner.batches import jobs
from experiment_runner.batches.jobs import (
    JOBS,
    JOBS_LOCK,
    append_result,
    load_questions,
    new_job,
    now,
    persist_job,
    question_key,
    register,
    snapshot_job,
    update_job,
    valid_job_id,
)
from experiment_runner.batches.metrics import batch_cases, batch_report, report_summary
from lakegen.config.experiment import (
    CoderContextLevel,
    ExperimentConfig,
)
from lakegen.config.paths import LOG_DIR
from lakegen.config.runtime import SOLR_CORE_PORTAL_NAMES
from lakegen.config.settings import (
    get_settings,
    non_negative_float_or_default,
    read_var,
)
from lakegen.config.startup import bootstrap_nltk_data
from lakegen.entrypoints.api.submission import WORKFLOW_LOCK, WORKFLOW_WORKERS
from lakegen.pipeline.evaluation.code import summarize_code_evaluations
from lakegen.pipeline.runner import ExperimentRunner
from lakegen.retrieval import RetrievalMode, check_embedding_health
from lakegen.retrieval.benchmark import append_benchmark_metrics_log

# Batch progress goes to Uvicorn's application logger, so it shows in the same
# terminal as the HTTP access log when the API runs under Uvicorn.
logger = logging.getLogger("uvicorn.error")


# Batches run on as many workers as single questions may run at once; the lock
# is the API's, so a batch question and a direct query share the limit.
EXECUTOR = ThreadPoolExecutor(max_workers=WORKFLOW_WORKERS, thread_name_prefix="lakegen-batch")


def non_negative_float_from_env(name: str, default: float) -> float:
    """Read a non-negative pacing setting, falling back to ``default`` when unset or invalid."""
    return read_var(name, non_negative_float_or_default, default)


# A breather inside long batches: after every LAKEGEN_BATCH_PAUSE_EVERY finished
# questions a job sleeps LAKEGEN_BATCH_PAUSE_SECONDS before its next question,
# which keeps a run of hundreds of questions under the provider's rate limits.
# Either value at 0 (the default) disables it. The pause is taken outside the
# workflow lock, so it never holds a slot another job's question could use.
BATCH_PAUSE_EVERY = int(non_negative_float_from_env("LAKEGEN_BATCH_PAUSE_EVERY", 0))
BATCH_PAUSE_SECONDS = non_negative_float_from_env("LAKEGEN_BATCH_PAUSE_SECONDS", 0.0)
# A rest sized by usage: after each question a job sleeps as long as that
# question's tokens take at LAKEGEN_BATCH_TOKENS_PER_MINUTE, so the job's
# average rate stays at that figure however heavy single questions are. It adds
# to the fixed pause above; 0 (the default) disables it.
BATCH_TOKENS_PER_MINUTE = non_negative_float_from_env("LAKEGEN_BATCH_TOKENS_PER_MINUTE", 0.0)
# A file holding the rate overrides it and is read before every rest, so a long
# run can be sped up or slowed down without a restart; a missing or unreadable
# file falls back to the variable.
BATCH_TOKENS_PER_MINUTE_FILE = get_settings().batch_tokens_per_minute_file


def tokens_per_minute() -> float:
    """The batch token rate now: the rate file's number, else the variable."""
    if BATCH_TOKENS_PER_MINUTE_FILE:
        try:
            value = float(Path(BATCH_TOKENS_PER_MINUTE_FILE).read_text().strip())
        except (OSError, ValueError):
            return BATCH_TOKENS_PER_MINUTE
        if value >= 0:
            return value
    return BATCH_TOKENS_PER_MINUTE


def batch_pause_seconds(
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


def question_tokens(result: dict[str, Any]) -> int:
    """LLM tokens one finished question spent: every phase plus the code judge.

    The judge is counted per coder-context variant when the variants ran, since
    each variant is judged; the top-level evaluation repeats one of them.

    Args:
        result: The question's result as a dict.

    Returns:
        The tokens of every phase plus those of the semantic code judge.
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


def token_pause_seconds(tokens: int, remaining: int, *, per_minute: float | None = None) -> float:
    """How long ``tokens`` take at the batch token rate; never after the last question."""
    per_minute = tokens_per_minute() if per_minute is None else per_minute
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


def split_fraction(source_id: Any) -> float:
    """A stable number in [0, 1) for a question id."""
    digest = hashlib.sha256(str(source_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0x100000000


def swap_model(value: Any, model: str, replacement: str) -> Any:
    """``value`` with every string equal to ``model`` replaced, at any depth."""
    if isinstance(value, dict):
        return {key: swap_model(item, model, replacement) for key, item in value.items()}
    if isinstance(value, list):
        return [swap_model(item, model, replacement) for item in value]
    return replacement if value == model else value


def inline_config(payload: Any) -> dict[str, Any] | None:
    """Read only an inline top-level config from the batch envelope.

    Args:
        payload: The batch document as submitted.

    Returns:
        The ``config`` object, or None when the envelope has none.

    Raises:
        ValueError: When ``config`` is neither an object nor null.
    """
    if not isinstance(payload, dict) or "config" not in payload:
        return None
    config = payload["config"]
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("batch config must be an inline JSON object")
    return config


def append_batch_table_metrics(
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

    Args:
        job_id: The job's id, the run id of the log rows.
        questions: The job's question records.
        results: The job's results lines.
        settings: The job's settings, holding its ``resolved_config``.
        append_log: Whether to append the report to the retrieval benchmarks log.

    Returns:
        The report's summary (``report_summary``), or None when the batch is
        not metric-ready.
    """
    cases = batch_cases(questions, results)
    if cases is None:
        return None
    resolved = settings["resolved_config"]
    report = batch_report(cases, results, resolved, created_at=now())
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


def _start(
    job_id: str,
    questions: list[dict[str, Any]],
) -> tuple[dict[str, int], set[str], list[dict[str, Any]]]:
    """Mark the job running and say which of its questions are left.

    The results already on disk are finished questions, so a resumed job
    goes on after its last durable result.

    Args:
        job_id: The job's id.
        questions: The job's question records.

    Returns:
        The results on disk counted by question key, the keys of the finished
        questions, and the pending question records in job order.
    """
    with JOBS_LOCK:
        current = JOBS[job_id]
        existing_results = list(current.get("results", []))
        started_at = current.get("started_at") or now()
    attempts: dict[str, int] = {}
    for entry in existing_results:
        source_key = question_key(entry)
        attempts[source_key] = attempts.get(source_key, 0) + 1
    completed_source_ids = set(attempts)
    pending_questions = [
        source for source in questions if question_key(source) not in completed_source_ids
    ]
    update_job(
        job_id,
        status="running",
        started_at=started_at,
        processed=len(completed_source_ids),
    )
    return attempts, completed_source_ids, pending_questions


def _split_runner(
    job_id: str,
    settings: dict[str, Any],
    job_model: Any,
    pending_questions: list[dict[str, Any]],
) -> ExperimentRunner | None:
    """The runner on ``SPLIT_MODEL`` for the split share, or None when the job runs on one model.

    Args:
        job_id: The job's id, for the log line.
        settings: The job's settings, holding its ``resolved_config``.
        job_model: The model the job's configuration names.
        pending_questions: The questions still to run, for the log line.

    Returns:
        A runner whose configuration names the split model wherever the job's
        named the job's model, or None.
    """
    if not (SPLIT_MODEL and SPLIT_SHARE > 0 and job_model and SPLIT_MODEL != job_model):
        return None
    split_runner = ExperimentRunner(
        ExperimentConfig.model_validate(
            swap_model(settings["resolved_config"], job_model, SPLIT_MODEL)
        )
    )
    logger.info(
        "Batch %s: %d of %d pending questions on %s, the rest on %s",
        job_id,
        sum(split_fraction(q["source_id"]) < SPLIT_SHARE for q in pending_questions),
        len(pending_questions),
        SPLIT_MODEL,
        job_model,
    )
    return split_runner


def _check_runtime(config: ExperimentConfig) -> None:
    """Fail before the first question when NLTK's data or the embedding service is missing.

    Args:
        config: The job's configuration; the embeddings are checked only when
            its retrieval mode uses them.

    Raises:
        RuntimeError: When NLTK's data cannot be bootstrapped.
    """
    nltk_error = bootstrap_nltk_data()
    if nltk_error:
        raise RuntimeError(nltk_error)
    retrieval_config = getattr(config, "retrieval", None)
    if retrieval_config is not None and RetrievalMode(retrieval_config.mode).uses_embeddings:
        health = check_embedding_health(
            retrieval_config.embedding_model,
            retrieval_config.embedding_base_url,
        )
        logger.info("Embedding health check passed: %s", health)


def _run_question(
    job_id: str,
    source: dict[str, Any],
    runner: ExperimentRunner,
    execution_attempt: int,
) -> dict[str, Any]:
    """Answer one question under the workflow lock; its results line.

    Args:
        job_id: The job's id, logged with the run.
        source: The question record.
        runner: The runner to answer it with.
        execution_attempt: The attempt number logged with the run.

    Returns:
        The results line: the question, its source and the result as a dict.
    """
    with WORKFLOW_LOCK:
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
    return {
        "question": source["question"],
        "source_path": source["source_path"],
        "source_id": source["source_id"],
        "result": query_result,
    }


def _record_result(
    job_id: str,
    entry: dict[str, Any],
    source_key: str,
    completed_source_ids: set[str],
) -> None:
    """Append a results line to disk and to the job, and persist the job.

    Args:
        job_id: The job's id.
        entry: The results line.
        source_key: The question's key.
        completed_source_ids: The keys of the finished questions; the key is added.
    """
    append_result(job_id, entry)
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["results"].append(entry)
        completed_source_ids.add(source_key)
        job["processed"] = len(completed_source_ids)
        if entry["result"]["status"] != "completed":
            job["failed"] += 1
        job["updated_at"] = now()
        snapshot = json.loads(json.dumps(job))
    persist_job(snapshot)


def _rest(job_id: str, finished_in_run: int, remaining: int, query_result: dict[str, Any]) -> None:
    """Sleep the fixed pause and the token-rate rest due after a question, if any.

    Args:
        job_id: The job's id, for the log line.
        finished_in_run: The questions this run has finished.
        remaining: The questions this run has left.
        query_result: The last question's result, whose tokens size the rest.
    """
    tokens = question_tokens(query_result)
    pause = batch_pause_seconds(finished_in_run, remaining) + token_pause_seconds(tokens, remaining)
    if pause:
        logger.info(
            "Batch %s: pausing %.0fs after %d questions (last one spent %d tokens)",
            job_id,
            pause,
            finished_in_run,
            tokens,
        )
        time.sleep(pause)


def _complete(job_id: str, questions: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    """Mark the job completed with its table-selection and code metrics.

    The table-selection metrics are logged once per job: a resumed job that
    logged them already keeps its stored ones. A failure to log them is
    logged and leaves them out.

    Args:
        job_id: The job's id.
        questions: The job's question records.
        settings: The job's settings, holding its ``resolved_config``.
    """
    with JOBS_LOCK:
        completed_results = json.loads(json.dumps(JOBS[job_id]["results"]))
        metrics_logged = bool(JOBS[job_id].get("metrics_logged"))
        existing_batch_metrics = json.loads(json.dumps(JOBS[job_id].get("batch_metrics", {})))
    table_metrics = None
    if not metrics_logged:
        try:
            table_metrics = append_batch_table_metrics(
                job_id, questions, completed_results, settings
            )
            if table_metrics:
                update_job(job_id, metrics_logged=True)
        except Exception:
            logger.exception("Could not persist batch table-selection metrics")
    elif existing_batch_metrics:
        table_metrics = existing_batch_metrics.get("table_selection")
    automatic_test_coder = bool(settings["resolved_config"].get("automatic_test_coder"))
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
    update_job(
        job_id,
        status="completed",
        finished_at=now(),
        batch_metrics={
            "table_selection": table_metrics,
            "code": code_metrics,
        },
    )


def run_batch(job_id: str, questions: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    """Answer a job's pending questions one by one, resting as paced, then close the job.

    With a split model configured, the questions whose id hashes below
    ``SPLIT_SHARE`` run on it and their results line names the model each ran
    on. Any failure marks the job failed with the error (its traceback goes to
    the log); the results already written stay.

    Args:
        job_id: The job's id; the job is registered in ``JOBS``.
        questions: The job's question records, finished ones included.
        settings: The job's settings, holding its ``resolved_config``.
    """
    attempts, completed_source_ids, pending_questions = _start(job_id, questions)
    try:
        config = ExperimentConfig.model_validate(settings["resolved_config"])
        runner = ExperimentRunner(config)
        job_model = settings["resolved_config"].get("model")
        split_runner = _split_runner(job_id, settings, job_model, pending_questions)
        _check_runtime(config)

        finished_in_run = 0
        for source in pending_questions:
            source_key = question_key(source)
            on_split = (
                split_runner is not None and split_fraction(source["source_id"]) < SPLIT_SHARE
            )
            entry = _run_question(
                job_id,
                source,
                split_runner if on_split else runner,
                attempts.get(source_key, 0) + 1,
            )
            if split_runner is not None:
                entry["model"] = SPLIT_MODEL if on_split else job_model
            _record_result(job_id, entry, source_key, completed_source_ids)
            finished_in_run += 1
            _rest(
                job_id, finished_in_run, len(pending_questions) - finished_in_run, entry["result"]
            )

        _complete(job_id, questions, settings)
    except Exception as exc:
        logger.exception("Batch %s failed", job_id)
        update_job(
            job_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=now(),
        )


def enqueue(questions: list[dict[str, Any]], resolved_config: ExperimentConfig) -> dict[str, Any]:
    """Register a batch of question records and queue its run; the job as created.

    A question record holds ``question``, ``source_path``, ``source_id`` and
    its ``log_fields`` (what the API extracts from a batch document).

    Args:
        questions: The batch's question records.
        resolved_config: The configuration the job runs with.

    Returns:
        The job record as registered.
    """
    job = new_job(uuid.uuid4().hex, questions, resolved_config.model_dump(mode="json"))
    register(job, questions)
    EXECUTOR.submit(run_batch, job["job_id"], questions, job["settings"])
    return job


def recover_incomplete_jobs() -> list[str]:
    """Resume persisted queued/running jobs from their last durable result."""
    if not jobs.JOB_DIR.is_dir():
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
    job_paths = [path for path in jobs.JOB_DIR.glob("*.json") if valid_job_id(path.stem)]
    for path in sorted(job_paths, key=submitted_at):
        job_id = path.stem
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("status") not in {"queued", "running"}:
                continue
            questions = load_questions(job_id)
            if not questions:
                logger.warning(
                    "Cannot resume job %s because its question checkpoint is missing",
                    job_id,
                )
                continue
            job = snapshot_job(job_id) or metadata
            results = job.get("results", [])
            unique_results = {question_key(entry): entry for entry in results}
            job["results"] = list(unique_results.values())
            job["processed"] = len(unique_results)
            job["failed"] = sum(
                entry.get("result", {}).get("status") != "completed"
                for entry in unique_results.values()
            )
            job["status"] = "queued"
            job["updated_at"] = now()
            with JOBS_LOCK:
                JOBS[job_id] = job
            persist_job(job)
            EXECUTOR.submit(run_batch, job_id, questions, job["settings"])
            recovered.append(job_id)
        except Exception:
            logger.exception("Could not recover batch job %s", job_id)
    return recovered
