"""The batch loop: every question of a job through the autonomous workflow, paced and recorded.

``run_batch`` answers the pending questions of a job one by one (resuming
after the results already on disk), rests between questions as the pacing
settings say, splits a share of them onto a second model when configured,
and closes the job with its table-selection and code metrics. ``enqueue``
creates a job and queues its run on the worker pool; ``recover_incomplete_jobs``
re-queues the jobs a previous process left queued or running.
``resolve_config`` turns a submission's fields and inline config into the
experiment configuration a job records.

The API's endpoints and the local executor of experiment specs both submit
through here; nothing here depends on the web framework.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lakegen.analysis.metrics import batch_cases, batch_report, report_summary
from lakegen.config.bootstrap import bootstrap_nltk_data
from lakegen.config.experiment import (
    CoderContextLevel,
    DiscoveryArchitecture,
    ExperimentConfig,
    load_experiment_config,
)
from lakegen.config.locations import LOG_DIR
from lakegen.config.runtime import SOLR_CORE_PORTAL_NAMES
from lakegen.config.settings import (
    get_settings,
    non_negative_float_or_default,
    positive_int_or_default,
    read_var,
)
from lakegen.evaluation.code import summarize_code_evaluations
from lakegen.experiments import jobs
from lakegen.experiments.jobs import (
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
from lakegen.retrieval import RetrievalMode, check_embedding_health
from lakegen.retrieval.benchmark import append_benchmark_metrics_log
from lakegen.runner import ExperimentRunner

# Batch progress goes to Uvicorn's application logger, so it shows in the same
# terminal as the HTTP access log when the API runs under Uvicorn.
logger = logging.getLogger("uvicorn.error")


def positive_int_from_env(name: str, default: int) -> int:
    """Read a positive worker limit without making API startup fragile."""

    return read_var(name, positive_int_or_default, default)


# A workflow may already parallelise table scans internally. Keep the safe
# historical default, while allowing a bounded number of questions/batches in flight.
WORKFLOW_WORKERS = positive_int_from_env("LAKEGEN_WORKFLOW_WORKERS", 1)
EXECUTOR = ThreadPoolExecutor(
    max_workers=WORKFLOW_WORKERS, thread_name_prefix="lakegen-batch"
)
WORKFLOW_LOCK = threading.BoundedSemaphore(WORKFLOW_WORKERS)


def non_negative_float_from_env(name: str, default: float) -> float:
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


def token_pause_seconds(
    tokens: int, remaining: int, *, per_minute: float | None = None
) -> float:
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


def resolve_config(
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


def inline_config(payload: Any) -> dict[str, Any] | None:
    """Read only an inline top-level config from the batch envelope."""

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


def run_batch(job_id: str, questions: list[dict[str, Any]], settings: dict[str, Any]) -> None:
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
        source for source in questions
        if question_key(source) not in completed_source_ids
    ]
    update_job(
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
                swap_model(settings["resolved_config"], job_model, SPLIT_MODEL)
            ))
            logger.info(
                "Batch %s: %d of %d pending questions on %s, the rest on %s",
                job_id,
                sum(split_fraction(q["source_id"]) < SPLIT_SHARE for q in pending_questions),
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
            source_key = question_key(source)
            execution_attempt = attempts.get(source_key, 0) + 1
            on_split = (
                split_runner is not None
                and split_fraction(source["source_id"]) < SPLIT_SHARE
            )
            with WORKFLOW_LOCK:
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
            append_result(job_id, entry)
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["results"].append(entry)
                completed_source_ids.add(source_key)
                job["processed"] = len(completed_source_ids)
                if query_result["status"] != "completed":
                    job["failed"] += 1
                job["updated_at"] = now()
                snapshot = json.loads(json.dumps(job))
            persist_job(snapshot)
            finished_in_run += 1
            remaining = len(pending_questions) - finished_in_run
            tokens = question_tokens(query_result)
            pause = batch_pause_seconds(finished_in_run, remaining) + token_pause_seconds(
                tokens, remaining
            )
            if pause:
                logger.info(
                    "Batch %s: pausing %.0fs after %d questions (last one spent %d tokens)",
                    job_id, pause, finished_in_run, tokens,
                )
                time.sleep(pause)

        with JOBS_LOCK:
            completed_results = json.loads(json.dumps(JOBS[job_id]["results"]))
            metrics_logged = bool(JOBS[job_id].get("metrics_logged"))
            existing_batch_metrics = json.loads(json.dumps(
                JOBS[job_id].get("batch_metrics", {})
            ))
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
        update_job(
            job_id,
            status="completed",
            finished_at=now(),
            batch_metrics={
                "table_selection": table_metrics,
                "code": code_metrics,
            },
        )
    except Exception as exc:
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
            unique_results = {
                question_key(entry): entry for entry in results
            }
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
