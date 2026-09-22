"""Retriever-only benchmark runner (no agents, selection, or code execution)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from src.client_solr import LocalSolrClient, resolve_solr_core
from lakegen.experiment_config import load_experiment_config
from lakegen.retrieval.config import FusionMethod, RetrievalConfig, RetrievalMode
from lakegen.retrieval.embeddings import EmbeddingModel
from lakegen.retrieval.evaluation import evaluate_ranking, mean_metrics
from lakegen.retrieval.retrievers import TableRetrievalService


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


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    question: str
    keywords: tuple[str, ...]
    relevant_table_ids: tuple[str, ...]


def _generated_case_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a generated-queries file into benchmark case rows.

    The file nests records as ``ENGINE -> table scope -> group -> record``, and
    each group's ``_meta.tables`` maps the aliases its records use (``Table_0``)
    to table ids. Only successful records are cases. The generator often asks
    the same question over the same tables once per engine; it is kept once, so
    no question weighs twice in the means.
    """

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for engine, scopes in payload.items():
        if not isinstance(scopes, dict):
            continue
        for scope, groups in scopes.items():
            if not isinstance(groups, dict):
                continue
            for group_name, group in groups.items():
                if not isinstance(group, dict):
                    continue
                table_ids = (group.get("_meta") or {}).get("tables") or {}
                for key, record in group.items():
                    if key == "_meta" or not isinstance(record, dict):
                        continue
                    if record.get("status") != "success":
                        continue
                    location = f"{engine}/{scope}/{group_name}/{key}"
                    aliases = [
                        table.get("name") if isinstance(table, dict) else table
                        for table in record.get("tables") or []
                    ]
                    if not aliases or any(alias not in table_ids for alias in aliases):
                        raise ValueError(
                            f"Generated query {location} uses tables {aliases} that "
                            "its group's _meta.tables does not resolve"
                        )
                    relevant = list(
                        dict.fromkeys(str(table_ids[alias]) for alias in aliases)
                    )
                    question = str(record.get("question", "")).strip()
                    key = (question, tuple(sorted(relevant)))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "id": record.get("client_id") or location,
                            "question": question,
                            # The keyword choice of build_benchmark.py, so a case
                            # searches identically in a sample drawn from this file.
                            "keywords": record.get("question_keywords")
                            or record.get("plan_keywords"),
                            "relevant_table_ids": relevant,
                        }
                    )
    if not rows:
        raise ValueError(
            "Benchmark input must be a list, an object with a cases list, or a "
            "generated-queries file with successful queries"
        )
    return rows


def load_benchmark_cases(path: Path) -> list[BenchmarkCase]:
    """Load a curated benchmark or a generated-queries file.

    A curated benchmark (``benchmark/100q_nyc.json``) is a list of cases or an
    object with a ``cases`` list. Any other object is read as a generated-queries
    file (``generated_queries_semantic.json``), see ``_generated_case_rows``.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "cases" not in payload:
        rows = _generated_case_rows(payload)
    else:
        rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Benchmark input must be a list or an object with a cases list")
    cases: list[BenchmarkCase] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Case {index} is not an object")
        case_id = row.get("id", index)
        relevant = row.get("relevant_table_ids") or row.get("gold_table_ids")
        if not relevant:
            raise ValueError(f"Case {case_id} has no relevant_table_ids")
        keywords = row.get("keywords")
        if not keywords:
            raise ValueError(f"Case {case_id} has no fixed retrieval keywords")
        cases.append(
            BenchmarkCase(
                case_id=str(case_id),
                question=str(row["question"]).strip(),
                keywords=tuple(map(str, keywords)),
                relevant_table_ids=tuple(dict.fromkeys(map(str, relevant))),
            )
        )
    return cases


def validate_gold_tables(
    cases: Sequence[BenchmarkCase], table_dir: Path
) -> dict[str, dict[str, Any]]:
    """Verify that every declared current-data gold table exists and is readable."""

    import pandas as pd
    import pyarrow.parquet as pq

    validation: dict[str, dict[str, Any]] = {}
    for table_id in sorted(
        {table_id for case in cases for table_id in case.relevant_table_ids}
    ):
        matches = [
            path
            for suffix in (".parquet", ".pq", ".csv")
            if (path := Path(table_dir) / f"{table_id}{suffix}").is_file()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Gold table {table_id!r} must resolve to exactly one current file; "
                f"found {len(matches)} in {table_dir}"
            )
        path = matches[0]
        if path.suffix.casefold() == ".csv":
            frame = pd.read_csv(path, nrows=0)
            row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
            columns = list(map(str, frame.columns))
        else:
            parquet = pq.ParquetFile(path)
            row_count = parquet.metadata.num_rows
            columns = list(map(str, parquet.schema_arrow.names))
        stat = path.stat()
        validation[table_id] = {
            "path": str(path),
            "row_count": row_count,
            "columns": columns,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "schema_sha256": hashlib.sha256(
                json.dumps(columns, ensure_ascii=False).encode()
            ).hexdigest(),
        }
    return validation


def _resource_id(document: dict[str, Any]) -> str:
    for field in ("resource_id", "dataset_id", "id"):
        value = document.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _experiment_configs(
    base: RetrievalConfig,
    modes: Iterable[RetrievalMode | str],
    alphas: Sequence[float],
    include_rrf: bool,
) -> list[tuple[str, RetrievalConfig]]:
    experiments: list[tuple[str, RetrievalConfig]] = []
    for raw_mode in modes:
        mode = RetrievalMode(raw_mode)
        if mode != RetrievalMode.HYBRID:
            experiments.append((mode.value, replace(base, mode=mode)))
            continue
        for alpha in alphas:
            experiments.append(
                (
                    f"hybrid-weighted-a{alpha:g}",
                    replace(
                        base,
                        mode=mode,
                        alpha=float(alpha),
                        fusion_method=FusionMethod.WEIGHTED,
                    ),
                )
            )
        if include_rrf:
            experiments.append(
                (
                    f"hybrid-rrf-k{base.rrf_k}",
                    replace(base, mode=mode, fusion_method=FusionMethod.RRF),
                )
            )
    return experiments


def run_retriever_benchmark(
    solr: LocalSolrClient,
    cases: Sequence[BenchmarkCase],
    *,
    base_config: RetrievalConfig | None = None,
    modes: Sequence[RetrievalMode | str] = tuple(RetrievalMode),
    alphas: Sequence[float] = (0.25, 0.5, 0.75),
    include_rrf: bool = True,
    k_values: Sequence[int] = (1, 5, 10),
    embedding_model: EmbeddingModel | None = None,
    table_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run exactly one retrieval per case for each explicit experiment config."""

    base = base_config or RetrievalConfig(top_k=max(k_values))
    if base.top_k < max(k_values):
        base = replace(base, top_k=max(k_values))
    experiments: dict[str, Any] = {}
    for label, config in _experiment_configs(base, modes, alphas, include_rrf):
        if config.mode.requires_table_dir and table_dir is None:
            # Recorded rather than skipped silently: a missing arm must not read
            # as an arm that scored nothing.
            experiments[label] = {
                "status": "skipped",
                "reason": f"mode {config.mode.value} requires --table-dir",
            }
            continue
        service = TableRetrievalService(
            solr,
            config,
            embedding_model=embedding_model,
            table_dir=str(table_dir) if table_dir is not None else None,
        )
        rows: list[dict[str, Any]] = []
        metric_rows: list[dict[str, float]] = []
        successful_metric_rows: list[dict[str, float]] = []
        for case in cases:
            error = ""
            try:
                hits = service.retrieve(
                    question=case.question,
                    keywords=case.keywords,
                    top_k=config.top_k,
                )
            except Exception as exc:
                hits = []
                error = f"{type(exc).__name__}: {exc}"
            ranking = [_resource_id(hit.document) for hit in hits]
            metrics = evaluate_ranking(
                ranking, case.relevant_table_ids, k_values=k_values
            )
            metric_rows.append(metrics)
            if not error:
                successful_metric_rows.append(metrics)
            rows.append(
                {
                    "case_id": case.case_id,
                    "question": case.question,
                    "keywords": list(case.keywords),
                    "relevant_table_ids": list(case.relevant_table_ids),
                    "ranking": ranking,
                    "hits": [hit.to_log_dict() for hit in hits],
                    "metrics": metrics,
                    "error": error,
                }
            )
        experiments[label] = {
            "config": {
                **asdict(config),
                "mode": config.mode.value,
                "missing_signal_policy": config.missing_signal_policy.value,
                "fusion_method": config.fusion_method.value,
            },
            "mean_metrics": mean_metrics(metric_rows),
            "mean_metrics_successful_queries": mean_metrics(successful_metric_rows),
            "successful_case_count": len(successful_metric_rows),
            "failed_case_count": len(cases) - len(successful_metric_rows),
            "cases": rows,
        }
    return {
        "benchmark_type": "retriever-only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "experiments": experiments,
    }


def _parse_job_id_map(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        label, separator, job_id = value.partition("=")
        if not separator or not label.strip() or not job_id.strip():
            raise ValueError(
                f"Invalid --source-job-id {value!r}; expected EXPERIMENT=JOB_ID"
            )
        result[label.strip()] = job_id.strip()
    return result


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
    """Append API-aligned aggregate rows, preserving legacy CSV aliases."""

    source_job_ids = source_job_ids or {}
    rows: list[dict[str, Any]] = []
    timestamp = str(report.get("created_at") or datetime.now(timezone.utc).isoformat())
    for label, experiment in report.get("experiments", {}).items():
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
                "STATUS": (
                    "completed"
                    if experiment["failed_case_count"] == 0
                    else "partial"
                ),
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
                "SEMANTIC_CODE_JUDGE_MODEL": config.get(
                    "semantic_code_judge_model", ""
                ),
                "CODE_APPLICABLE_CASES": code_metrics.get(
                    "applicable_case_count", ""
                ),
                "CODE_EXECUTION_SUCCESS_RATE": code_metrics.get(
                    "execution_success_rate", ""
                ),
                "EXACT_RESULT_MATCH_RATE": code_metrics.get(
                    "exact_result_match_rate", ""
                ),
                "REPRESENTATION_EQUIVALENT_MATCH_RATE": code_metrics.get(
                    "representation_equivalent_match_rate", ""
                ),
                "SUPPORTED_RESULT_RATE": code_metrics.get(
                    "supported_result_rate", ""
                ),
                "AMBIGUOUS_RESULT_RATE": code_metrics.get(
                    "ambiguous_result_rate", ""
                ),
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
                "SEMANTIC_JUDGE_CASE_COUNT": code_metrics.get(
                    "semantic_judge_case_count", ""
                ),
                "SEMANTIC_JUDGE_TOTAL_TOKENS": code_metrics.get(
                    "semantic_judge_total_tokens", ""
                ),
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
                    json.dumps(code_metrics.get("semantic_judge_outcomes", {}),
                               ensure_ascii=False, sort_keys=True)
                    if code_metrics else ""
                ),
                "SEMANTIC_JUDGE_HARDCODING_RISKS_JSON": (
                    json.dumps(code_metrics.get("semantic_judge_hardcoding_risks", {}),
                               ensure_ascii=False, sort_keys=True)
                    if code_metrics else ""
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
        numeric_ids = [
            int(row["ID"])
            for row in existing_rows
            if str(row.get("ID", "")).isdigit()
        ]
        next_id = max(numeric_ids, default=0) + 1
        fieldnames = list(BENCHMARK_LOG_COLUMNS)
        if path.is_file():
            migrated_rows = []
            for existing in existing_rows:
                migrated = dict(existing)
                migrated["JOB_ID"] = migrated.get("JOB_ID") or migrated.get("RUN_ID", "")
                migrated["EXPERIMENT_ID"] = (
                    migrated.get("EXPERIMENT_ID") or migrated.get("EXPERIMENT", "")
                )
                migrated["RETRIEVAL_MODE"] = (
                    migrated.get("RETRIEVAL_MODE") or migrated.get("MODE", "")
                )
                migrated["HYBRID_ALPHA"] = (
                    migrated.get("HYBRID_ALPHA") or migrated.get("ALPHA", "")
                )
                migrated_rows.append(migrated)
            existing_rows = migrated_rows
            if fieldnames != existing_columns:
                temporary_path = path.with_suffix(path.suffix + ".tmp")
                with temporary_path.open(
                    "w", newline="", encoding="utf-8"
                ) as migrated_file:
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
            writer = csv.DictWriter(
                output_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            if is_new:
                writer.writeheader()
            for row in rows:
                row["ID"] = next_id
                next_id += 1
                writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Curated benchmark or generated-queries JSON "
        "(default: benchmark.path from --config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Experiment YAML/JSON supplying benchmark.path, the core, and the "
        "retrieval settings every experiment starts from",
    )
    parser.add_argument(
        "--core", default=None, help="Solr core (default: the config's, else nyc)"
    )
    parser.add_argument(
        "--solr-base-url",
        default=os.environ.get("SOLR_BASE_URL", "http://localhost:8983/solr"),
        help="Solr base URL (default: SOLR_BASE_URL or localhost)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-multiplier", type=int, default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    parser.add_argument(
        "--metrics-log",
        type=Path,
        default=Path("logs/retrieval_benchmarks_log.csv"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--source-job-id",
        action="append",
        default=[],
        metavar="EXPERIMENT=JOB_ID",
    )
    args = parser.parse_args(argv)

    experiment = load_experiment_config(args.config) if args.config else None
    input_path = args.input
    if input_path is None and experiment is not None and experiment.benchmark.path:
        input_path = Path(experiment.benchmark.path)
    if input_path is None:
        parser.error("give a benchmark file, or --config with benchmark.path set")
    core = args.core or (experiment.core if experiment is not None else "nyc")
    base_config = replace(
        experiment.retrieval.to_runtime() if experiment is not None else RetrievalConfig(),
        **{
            name: value
            for name, value in {
                "top_k": args.top_k,
                "candidate_multiplier": args.candidate_multiplier,
            }.items()
            if value is not None
        },
    )

    cases = load_benchmark_cases(input_path)
    report = run_retriever_benchmark(
        LocalSolrClient(resolve_solr_core(core), base_url=args.solr_base_url),
        cases,
        base_config=base_config,
        alphas=args.alphas,
        table_dir=args.table_dir,
    )
    if args.table_dir:
        report["gold_validation"] = validate_gold_tables(cases, args.table_dir)
    run_id = args.run_id or uuid.uuid4().hex
    report["run_id"] = run_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    append_benchmark_metrics_log(
        report,
        args.metrics_log,
        run_id=run_id,
        core=core,
        source_path=str(input_path),
        source_job_ids=_parse_job_id_map(args.source_job_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
