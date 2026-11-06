"""TARGET-oriented ranking evaluation and trace logging."""

from __future__ import annotations

import csv
import json
import math
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lakegen.datastructures.retrieval import RetrievalRun


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items))


def evaluate_ranking(
    retrieved_ids: Sequence[str],
    relevant: Iterable[str] | Mapping[str, float],
    *,
    k_values: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Hit@k, Recall@k, MRR, and binary/graded nDCG@k.

    Args:
        retrieved_ids: The ranking, best first; repeated ids count once.
        relevant: The relevant ids (grade 1 each) or a mapping of id to grade;
            ids with a grade that is not finite and positive are dropped.
        k_values: The cut-offs to report.

    Returns:
        ``MRR`` and, for each k, ``Hit@k``, ``Recall@k`` and ``nDCG@k``.

    Raises:
        ValueError: When no relevant id is left, or a k is not positive.
    """
    ranking = _unique(retrieved_ids)
    if isinstance(relevant, Mapping):
        grades = {str(key): float(value) for key, value in relevant.items()}
    else:
        grades = {key: 1.0 for key in _unique(relevant)}
    grades = {key: value for key, value in grades.items() if math.isfinite(value) and value > 0}
    if not grades:
        raise ValueError("At least one relevant document is required")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    result: dict[str, float] = {}
    first_relevant = next(
        (rank for rank, doc_id in enumerate(ranking, 1) if doc_id in grades),
        None,
    )
    result["MRR"] = 1.0 / first_relevant if first_relevant is not None else 0.0

    ideal_grades = sorted(grades.values(), reverse=True)
    for k in k_values:
        prefix = ranking[:k]
        matches = sum(doc_id in grades for doc_id in prefix)
        result[f"Hit@{k}"] = float(matches > 0)
        result[f"Recall@{k}"] = matches / len(grades)
        dcg = sum(
            (2.0 ** grades.get(doc_id, 0.0) - 1.0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(prefix, 1)
        )
        ideal_dcg = sum(
            (2.0**grade - 1.0) / math.log2(rank + 1)
            for rank, grade in enumerate(ideal_grades[:k], 1)
        )
        result[f"nDCG@{k}"] = dcg / ideal_dcg if ideal_dcg else 0.0
    return result


def mean_metrics(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """The mean of each metric present in every row; empty for no rows."""
    if not rows:
        return {}
    keys = set.intersection(*(set(row) for row in rows))
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in sorted(keys)}


class RetrievalRunLogger:
    """Append full rankings as JSONL for retriever-only evaluation.

    Calling the logger with a run appends it as a ``retrieval_run`` event. Writes
    are serialised with a lock, so threads can share one logger.

    Args:
        path: The JSONL file; its directory is created on the first write.

    Attributes:
        path: The JSONL file.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def __call__(self, run: RetrievalRun) -> None:
        """Append ``run`` as a ``retrieval_run`` event."""
        payload: dict[str, Any] = run.to_log_dict()
        payload["event_type"] = "retrieval_run"
        self.log_payload(payload)

    def log_payload(self, payload: Mapping[str, Any]) -> None:
        """Append ``payload`` as one JSON line, stamped with the current UTC time."""
        payload = dict(payload)
        payload["timestamp"] = datetime.now(UTC).isoformat()
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line + "\n")


BENCHMARK_LOG_COLUMNS = [
    "ID",
    "TIMESTAMP",
    "JOB_ID",
    "SOURCE_PATH",
    "MODEL",
    "ARCHITECTURE",
    "CORE",
    "PORTAL_NAME",
    "STATUS",
    "BENCHMARK_TYPE",
    "EXPERIMENT_ID",
    "QUESTION_COUNT",
    "SUCCESSFUL_QUERIES",
    "FAILED_QUERIES",
    "RETRIEVAL_MODE",
    "TOP_K",
    "HYBRID_ALPHA",
    "CANDIDATE_MULTIPLIER",
    "REPRESENTATION_VERSION",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "VECTOR_FIELD",
    "LEXICAL_QUERY_FIELDS",
    "MISSING_SIGNAL_POLICY",
    "FUSION_METHOD",
    "RRF_K",
    "HIT_AT_1",
    "HIT_AT_5",
    "HIT_AT_10",
    "HIT_AT_15",
    "HIT_AT_20",
    "RECALL_AT_1",
    "RECALL_AT_5",
    "RECALL_AT_10",
    "RECALL_AT_15",
    "RECALL_AT_20",
    "MRR",
    "NDCG_AT_1",
    "NDCG_AT_5",
    "NDCG_AT_10",
    "NDCG_AT_15",
    "NDCG_AT_20",
    "CODER_CONTEXT_LEVEL",
    "CODE_APPLICABLE_CASES",
    "CODE_EXECUTION_SUCCESS_RATE",
    "EXACT_RESULT_MATCH_RATE",
    "PASS_AT_1",
    "SUCCESS_WITHIN_3",
    "MEAN_CODE_ATTEMPTS",
    "CODE_CASE_COUNT_BY_RESULT_TYPE_JSON",
    "EXACT_RESULT_MATCH_RATE_BY_TYPE_JSON",
    "CODE_ERROR_CATEGORIES_JSON",
    "SEMANTIC_CODE_JUDGE_MODEL",
    "SUPPORTED_RESULT_RATE",
    "AMBIGUOUS_RESULT_RATE",
    "EVALUATION_DISPOSITIONS_JSON",
    "SEMANTIC_JUDGE_CASE_COUNT",
    "SEMANTIC_JUDGE_TOTAL_TOKENS",
    "SEMANTIC_JUDGE_PARSE_SUCCESS_RATE",
    "SEMANTIC_JUDGE_EVIDENCE_COVERAGE",
    "SEMANTIC_JUDGE_DOWNGRADE_RATE",
    "SEMANTIC_JUDGE_BLOCKING_OBJECTIONS",
    "SEMANTIC_JUDGE_OUTCOMES_JSON",
    "SEMANTIC_JUDGE_HARDCODING_RISKS_JSON",
]
_BENCHMARK_LOG_LOCK = threading.Lock()


def append_benchmark_metrics_log(
    report: dict[str, Any],
    path: Path,
    *,
    run_id: str,
    core: str,
    source_path: str,
    source_job_ids: dict[str, str] | None = None,
    model: str = "",
    architecture: str = "",
    portal_name: str = "",
) -> None:
    """Append API-aligned aggregate rows, preserving legacy CSV aliases.

    One row per experiment that ran. An existing log with other columns is
    first rewritten with the current ones, its legacy column names carried over.

    Args:
        report: The report of ``run_retriever_benchmark``.
        path: The CSV log.
        run_id: The job id of experiments without one of their own.
        core: The Solr core benchmarked.
        source_path: The benchmark file.
        source_job_ids: The job id of each experiment label, when it has its own.
        model: The model column.
        architecture: The architecture column.
        portal_name: The portal column.
    """
    source_job_ids = source_job_ids or {}
    rows: list[dict[str, Any]] = []
    timestamp = str(report.get("created_at") or datetime.now(UTC).isoformat())
    for label, experiment in report.get("experiments", {}).items():
        if experiment.get("status") == "skipped":
            continue  # recorded in the report with its reason; it has no metrics
        config = experiment["config"]
        metrics = experiment["mean_metrics"]
        code_metrics = experiment.get("code_metrics", {})
        question_count = len(experiment["cases"])
        rows.append(
            {
                "TIMESTAMP": timestamp,
                "JOB_ID": source_job_ids.get(label, "") or run_id,
                "SOURCE_PATH": source_path,
                "MODEL": model,
                "ARCHITECTURE": architecture,
                "CORE": core,
                "PORTAL_NAME": portal_name,
                "STATUS": ("completed" if experiment["failed_case_count"] == 0 else "partial"),
                "BENCHMARK_TYPE": report.get("benchmark_type", "retriever-only"),
                "EXPERIMENT_ID": label,
                "QUESTION_COUNT": question_count,
                "SUCCESSFUL_QUERIES": experiment["successful_case_count"],
                "FAILED_QUERIES": experiment["failed_case_count"],
                "RETRIEVAL_MODE": config["mode"],
                "TOP_K": config["top_k"],
                "HYBRID_ALPHA": config["alpha"],
                "CANDIDATE_MULTIPLIER": config["candidate_multiplier"],
                "REPRESENTATION_VERSION": config["representation_version"],
                "EMBEDDING_MODEL": config["embedding_model"],
                "EMBEDDING_BASE_URL": config.get("embedding_base_url", ""),
                "VECTOR_FIELD": config.get("vector_field", ""),
                "LEXICAL_QUERY_FIELDS": config.get("lexical_query_fields") or "",
                "MISSING_SIGNAL_POLICY": config.get("missing_signal_policy", ""),
                "FUSION_METHOD": config["fusion_method"],
                "RRF_K": config["rrf_k"],
                "HIT_AT_1": metrics.get("Hit@1", ""),
                "HIT_AT_5": metrics.get("Hit@5", ""),
                "HIT_AT_10": metrics.get("Hit@10", ""),
                "HIT_AT_15": metrics.get("Hit@15", ""),
                "HIT_AT_20": metrics.get("Hit@20", ""),
                "RECALL_AT_1": metrics.get("Recall@1", ""),
                "RECALL_AT_5": metrics.get("Recall@5", ""),
                "RECALL_AT_10": metrics.get("Recall@10", ""),
                "RECALL_AT_15": metrics.get("Recall@15", ""),
                "RECALL_AT_20": metrics.get("Recall@20", ""),
                "MRR": metrics.get("MRR", ""),
                "NDCG_AT_1": metrics.get("nDCG@1", ""),
                "NDCG_AT_5": metrics.get("nDCG@5", ""),
                "NDCG_AT_10": metrics.get("nDCG@10", ""),
                "NDCG_AT_15": metrics.get("nDCG@15", ""),
                "NDCG_AT_20": metrics.get("nDCG@20", ""),
                "CODER_CONTEXT_LEVEL": config.get("coder_context_level", ""),
                "SEMANTIC_CODE_JUDGE_MODEL": config.get("semantic_code_judge_model", ""),
                "CODE_APPLICABLE_CASES": code_metrics.get("applicable_case_count", ""),
                "CODE_EXECUTION_SUCCESS_RATE": code_metrics.get("execution_success_rate", ""),
                "EXACT_RESULT_MATCH_RATE": code_metrics.get("exact_result_match_rate", ""),
                "REPRESENTATION_EQUIVALENT_MATCH_RATE": code_metrics.get(
                    "representation_equivalent_match_rate", ""
                ),
                "SUPPORTED_RESULT_RATE": code_metrics.get("supported_result_rate", ""),
                "AMBIGUOUS_RESULT_RATE": code_metrics.get("ambiguous_result_rate", ""),
                "PASS_AT_1": code_metrics.get("pass_at_1", ""),
                "SUCCESS_WITHIN_3": code_metrics.get("success_within_3", ""),
                "MEAN_CODE_ATTEMPTS": code_metrics.get("mean_attempts", ""),
                "CODE_CASE_COUNT_BY_RESULT_TYPE_JSON": (
                    json.dumps(
                        code_metrics.get("case_count_by_result_type", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if code_metrics
                    else ""
                ),
                "EXACT_RESULT_MATCH_RATE_BY_TYPE_JSON": (
                    json.dumps(
                        code_metrics.get("exact_result_match_rate_by_type", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if code_metrics
                    else ""
                ),
                "CODE_ERROR_CATEGORIES_JSON": (
                    json.dumps(
                        code_metrics.get("error_categories", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if code_metrics
                    else ""
                ),
                "EVALUATION_DISPOSITIONS_JSON": (
                    json.dumps(
                        code_metrics.get("evaluation_dispositions", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if code_metrics
                    else ""
                ),
                "SEMANTIC_JUDGE_CASE_COUNT": code_metrics.get("semantic_judge_case_count", ""),
                "SEMANTIC_JUDGE_TOTAL_TOKENS": code_metrics.get("semantic_judge_total_tokens", ""),
                "SEMANTIC_JUDGE_PARSE_SUCCESS_RATE": code_metrics.get(
                    "semantic_judge_parse_success_rate", ""
                ),
                "SEMANTIC_JUDGE_EVIDENCE_COVERAGE": code_metrics.get(
                    "semantic_judge_mean_evidence_coverage", ""
                ),
                "SEMANTIC_JUDGE_DOWNGRADE_RATE": code_metrics.get(
                    "semantic_judge_downgrade_rate", ""
                ),
                "SEMANTIC_JUDGE_BLOCKING_OBJECTIONS": code_metrics.get(
                    "semantic_judge_blocking_objection_count", ""
                ),
                "SEMANTIC_JUDGE_OUTCOMES_JSON": (
                    json.dumps(
                        code_metrics.get("semantic_judge_outcomes", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if code_metrics
                    else ""
                ),
                "SEMANTIC_JUDGE_HARDCODING_RISKS_JSON": (
                    json.dumps(
                        code_metrics.get("semantic_judge_hardcoding_risks", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if code_metrics
                    else ""
                ),
            }
        )

    path = Path(path)
    with _BENCHMARK_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows: list[dict[str, Any]] = []
        existing_columns: list[str] = []
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                existing_columns = list(reader.fieldnames or [])
                existing_rows = list(reader)
        numeric_ids = [int(row["ID"]) for row in existing_rows if str(row.get("ID", "")).isdigit()]
        next_id = max(numeric_ids, default=0) + 1
        fieldnames = list(BENCHMARK_LOG_COLUMNS)
        if path.is_file():
            migrated_rows = []
            for existing in existing_rows:
                migrated = dict(existing)
                migrated["JOB_ID"] = migrated.get("JOB_ID") or migrated.get("RUN_ID", "")
                migrated["EXPERIMENT_ID"] = migrated.get("EXPERIMENT_ID") or migrated.get(
                    "EXPERIMENT", ""
                )
                migrated["RETRIEVAL_MODE"] = migrated.get("RETRIEVAL_MODE") or migrated.get(
                    "MODE", ""
                )
                migrated["HYBRID_ALPHA"] = migrated.get("HYBRID_ALPHA") or migrated.get("ALPHA", "")
                migrated_rows.append(migrated)
            existing_rows = migrated_rows
            if fieldnames != existing_columns:
                temporary_path = path.with_suffix(path.suffix + ".tmp")
                with temporary_path.open("w", newline="", encoding="utf-8") as migrated_file:
                    migrated_writer = csv.DictWriter(
                        migrated_file,
                        fieldnames=fieldnames,
                        extrasaction="ignore",
                    )
                    migrated_writer.writeheader()
                    migrated_writer.writerows(existing_rows)
                os.replace(temporary_path, path)
        is_new = not path.is_file()
        with path.open("a", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            for row in rows:
                row["ID"] = next_id
                next_id += 1
                writer.writerow(row)
