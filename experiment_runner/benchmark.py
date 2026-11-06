"""Retriever-only benchmark runner (no agents, selection, or code execution)."""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lakegen.config.experiment import load_experiment_config
from lakegen.config.retrieval import FusionMethod, RetrievalConfig, RetrievalMode
from lakegen.config.settings import get_settings
from lakegen.llm.embeddings import EmbeddingModel
from lakegen.retrieval.benchmark import (
    append_benchmark_metrics_log,
    evaluate_ranking,
    mean_metrics,
)
from lakegen.retrieval.service import RetrievalService
from lakegen.retrieval.solr.catalogue import match_local_csv
from lakegen.retrieval.solr.client import SolrClient, resolve_solr_core
from lakegen.utils.table_io import list_table_files


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark question with its fixed keywords and gold tables.

    Attributes:
        case_id: The case's id in the benchmark file.
        question: The question.
        keywords: The retrieval keywords, fixed so every mode searches alike.
        relevant_table_ids: The gold tables, by local file stem.
    """

    case_id: str
    question: str
    keywords: tuple[str, ...]
    relevant_table_ids: tuple[str, ...]


def _successful_generated_records(
    payload: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Walk a generated-queries file down to its successful records.

    Args:
        payload: The parsed file, nested as ``ENGINE -> table scope -> group -> record``.

    Yields:
        Each successful record's location (``engine/scope/group/key``), the
        record, and its group's ``_meta.tables`` alias map, in file order.
    """
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
                    yield f"{engine}/{scope}/{group_name}/{key}", record, table_ids


def _generated_case_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a generated-queries file into benchmark case rows.

    The file nests records as ``ENGINE -> table scope -> group -> record``, and
    each group's ``_meta.tables`` maps the aliases its records use (``Table_0``)
    to table ids. Only successful records are cases. The generator often asks
    the same question over the same tables once per engine; it is kept once, so
    no question weighs twice in the means.

    Args:
        payload: The parsed generated-queries file.

    Returns:
        The case rows (``id``, ``question``, ``keywords``, ``relevant_table_ids``).

    Raises:
        ValueError: When a record uses a table alias its group does not resolve,
            or the file has no successful record.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for location, record, table_ids in _successful_generated_records(payload):
        aliases = [
            table.get("name") if isinstance(table, dict) else table
            for table in record.get("tables") or []
        ]
        if not aliases or any(alias not in table_ids for alias in aliases):
            raise ValueError(
                f"Generated query {location} uses tables {aliases} that "
                "its group's _meta.tables does not resolve"
            )
        relevant = list(dict.fromkeys(str(table_ids[alias]) for alias in aliases))
        question = str(record.get("question", "")).strip()
        identity = (question, tuple(sorted(relevant)))
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(
            {
                "id": record.get("client_id") or location,
                "question": question,
                # The keyword choice of build_benchmark.py, so a case
                # searches identically in a sample drawn from this file.
                "keywords": record.get("question_keywords") or record.get("plan_keywords"),
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

    Args:
        path: The JSON file.

    Returns:
        The cases, in file order; repeated gold ids count once.

    Raises:
        ValueError: When the file has no list of cases, or a case is not an
            object or lacks gold tables or keywords.
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
    """Verify that every declared current-data gold table exists and is readable.

    Args:
        cases: The benchmark cases.
        table_dir: The lake's table directory.

    Returns:
        For each gold table id: its path, row count, columns, size, mtime and the
        SHA-256 of its column list.

    Raises:
        FileNotFoundError: When a gold id resolves to no file, or to more than
            one (``.parquet``, ``.pq``, ``.csv``).
    """
    import pandas as pd
    import pyarrow.parquet as pq

    validation: dict[str, dict[str, Any]] = {}
    for table_id in sorted({table_id for case in cases for table_id in case.relevant_table_ids}):
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
    """The first non-blank of the document's resource, dataset and plain id; empty when none."""
    for field in ("resource_id", "dataset_id", "id"):
        value = document.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _ranked_id(document: dict[str, Any], table_files: list[str]) -> str:
    """The id a hit is scored under: the stem of the local file it maps to.

    Gold ids name local files (``validate_gold_tables`` resolves
    ``<table_dir>/<id>.parquet``), but a Solr document's ``resource_id`` is only
    part of a UK file name (``<dataset_id>___<resource_id>``) and DuckDB names a
    hit by its file name, suffix included. So a hit is mapped the way the
    workflow maps it, with ``match_local_csv``; without a table directory, or
    when nothing maps, the document's own id is used.

    Args:
        document: The hit's document.
        table_files: The lake's local table files; empty without a table directory.

    Returns:
        The id the hit is scored under.
    """
    matched = match_local_csv(document, table_files) if table_files else None
    return Path(matched).stem if matched is not None else _resource_id(document)


def _experiment_configs(
    base: RetrievalConfig,
    modes: Iterable[RetrievalMode | str],
    alphas: Sequence[float],
    include_rrf: bool,
) -> list[tuple[str, RetrievalConfig]]:
    """The labelled configurations to run: one per mode, one per alpha and RRF for the hybrids.

    Args:
        base: The configuration every experiment starts from.
        modes: The modes to run.
        alphas: The weights of the weighted hybrid arms.
        include_rrf: Whether the hybrids also get an RRF arm.

    Returns:
        The experiment labels with their configurations, in mode order.
    """
    experiments: list[tuple[str, RetrievalConfig]] = []
    for raw_mode in modes:
        mode = RetrievalMode(raw_mode)
        if mode not in (RetrievalMode.HYBRID, RetrievalMode.HYBRID_DUCKDB_SEMANTIC):
            experiments.append((mode.value, replace(base, mode=mode)))
            continue
        # "hybrid-weighted-a0.5", "hybrid (duckdb + semantic)-weighted-a0.5", ...
        name = mode.label
        for alpha in alphas:
            experiments.append(
                (
                    f"{name}-weighted-a{alpha:g}",
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
                    f"{name}-rrf-k{base.rrf_k}",
                    replace(base, mode=mode, fusion_method=FusionMethod.RRF),
                )
            )
    return experiments


def run_retriever_benchmark(
    solr: SolrClient,
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
    """Run exactly one retrieval per case for each explicit experiment config.

    A retrieval that fails scores as an empty ranking, with its error kept on
    the case row; a mode that needs a table directory, without one, is
    recorded as skipped.

    Args:
        solr: The lake's Solr client.
        cases: The benchmark cases.
        base_config: The configuration every experiment starts from; ``top_k``
            is raised to the largest k evaluated.
        modes: The modes to run.
        alphas: The weights of the weighted hybrid arms.
        include_rrf: Whether the hybrids also get an RRF arm.
        k_values: The cut-offs to evaluate.
        embedding_model: The embedding model of the semantic branches.
        table_dir: The lake's local tables, for the modes that read them and to
            map hits to file stems.

    Returns:
        The report: benchmark type, creation time, case count and, per
        experiment label, its configuration, mean metrics and case rows.
    """
    base = base_config or RetrievalConfig(top_k=max(k_values))
    if base.top_k < max(k_values):
        base = replace(base, top_k=max(k_values))
    table_files = list_table_files(table_dir) if table_dir is not None else []
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
        service = RetrievalService(
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
            ranking = [_ranked_id(hit.document, table_files) for hit in hits]
            metrics = evaluate_ranking(ranking, case.relevant_table_ids, k_values=k_values)
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
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(cases),
        "experiments": experiments,
    }


def _parse_job_id_map(values: Sequence[str]) -> dict[str, str]:
    """Parse ``EXPERIMENT=JOB_ID`` pairs into a map.

    Args:
        values: The ``--source-job-id`` values.

    Returns:
        The job id of each experiment label.

    Raises:
        ValueError: When a value is not ``EXPERIMENT=JOB_ID`` with both sides non-blank.
    """
    result: dict[str, str] = {}
    for value in values:
        label, separator, job_id = value.partition("=")
        if not separator or not label.strip() or not job_id.strip():
            raise ValueError(f"Invalid --source-job-id {value!r}; expected EXPERIMENT=JOB_ID")
        result[label.strip()] = job_id.strip()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Run the retriever-only benchmark from the command line.

    Args:
        argv: The arguments; the process's when None.

    Returns:
        The exit status, 0.
    """
    # The same .env the API, the CLI and the UI load, so the benchmark queries
    # the Solr cores and Pneuma services the other entry points use.
    load_dotenv()
    parser = argparse.ArgumentParser(prog="python -m experiment_runner bench", description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Curated benchmark or generated-queries JSON (default: benchmark.path from --config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Experiment YAML/JSON supplying benchmark.path, the core, and the "
        "retrieval settings every experiment starts from",
    )
    parser.add_argument("--core", default=None, help="Solr core (default: the config's, else nyc)")
    parser.add_argument(
        "--solr-base-url",
        default=get_settings().solr_base_url,
        help="Solr base URL (default: SOLR_BASE_URL or localhost)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--candidate-multiplier", type=int, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in RetrievalMode],
        default=None,
        help="Retrieval modes to run (default: every mode)",
    )
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
        SolrClient(resolve_solr_core(core), base_url=args.solr_base_url),
        cases,
        base_config=base_config,
        modes=args.modes or tuple(RetrievalMode),
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
