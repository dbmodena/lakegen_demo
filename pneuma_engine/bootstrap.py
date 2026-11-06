"""Build a resumable Pneuma index from a lake's tables and catalogue metadata.

Run from the repository root inside the engine's environment::

    .venv-pneuma/bin/python -m pneuma_engine.bootstrap --portal nyc --out-path pneuma-out/nyc

Every step resumes: tables already registered, summarized or given metadata
are skipped, so an interrupted build continues where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
from pneuma import Pneuma

from pneuma_engine.indexing import (
    DEFAULT_EXTERNAL_VIEW_COLUMN_LIMIT,
    DEFAULT_EXTERNAL_VIEW_ROW_LIMIT,
    configure_bounded_external_views,
    configure_pneuma_indexing,
    tune_duckdb_connections,
)
from pneuma_engine.metadata import metadata_entries, normalize_metadata_for_pneuma
from pneuma_engine.narration import cap_long_cells, concurrent_prompt_openai_llm
from pneuma_engine.oci_client import add_oci_arguments, client_from_arguments

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_SUFFIXES = {".csv", ".parquet"}
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_CHROMA_INSERT_BATCH_SIZE = 1_000
# Progress is logged for the first table, every this many, and the last.
_PROGRESS_EVERY = 25


def response_payload(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Decode a Pneuma API response.

    Args:
        raw: The response, as a JSON string or already decoded.

    Returns:
        The response object.

    Raises:
        RuntimeError: If the response is not a JSON object.
    """
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        raise RuntimeError("Pneuma returned a non-object response")
    return payload


def require_success(raw: str | dict[str, Any], operation: str) -> dict[str, Any]:
    """Decode a Pneuma API response and fail unless it reports success.

    Args:
        raw: The response, as a JSON string or already decoded.
        operation: What was attempted, for the error message.

    Returns:
        The response object.

    Raises:
        RuntimeError: If the response does not report ``SUCCESS``.
    """
    payload = response_payload(raw)
    if payload.get("status") != "SUCCESS":
        raise RuntimeError(f"{operation} failed: {payload.get('message', payload)}")
    return payload


def table_paths(table_dir: Path) -> list[Path]:
    """List the CSV and Parquet files of a table folder, resolved and sorted."""
    return sorted(
        path.resolve()
        for path in table_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    )


def existing_table_ids(db_path: Path) -> set[str]:
    """Return the IDs of the tables already registered in Pneuma's ``storage.db``."""
    if not db_path.exists():
        return set()
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return {row[0] for row in connection.sql("SELECT id FROM table_status").fetchall()}


def pending_summary_ids(db_path: Path) -> list[str]:
    """Return the IDs of the registered tables not summarized yet, sorted."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return [
            row[0]
            for row in connection.sql(
                "SELECT id FROM table_status WHERE status = 'TableStatus.REGISTERED' ORDER BY id"
            ).fetchall()
        ]


def context_table_ids(db_path: Path) -> set[str]:
    """Return the IDs of the tables that already have a metadata context."""
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return {
            row[0]
            for row in connection.sql("SELECT DISTINCT table_id FROM table_contexts").fetchall()
        }


def register_external_view(
    db_path: Path,
    path: Path,
    *,
    creator: str,
    connection: Any | None = None,
) -> None:
    """Register a Parquet file as a view, for a table too large to materialize.

    Args:
        db_path: Pneuma's ``storage.db``, opened when ``connection`` is not given.
        path: The Parquet file.
        creator: Creator recorded in ``table_status``.
        connection: An open connection to ``storage.db`` to reuse, if any.
    """
    identifier = str(path).replace('"', '""')
    literal = str(path).replace("'", "''")
    stat = path.stat()
    fingerprint = hashlib.sha256(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()

    def register(target: Any) -> None:
        # The view is named by its quoted file path and reads it as an escaped literal.
        view_sql = (
            f"CREATE OR REPLACE VIEW \"{identifier}\" AS SELECT * FROM read_parquet('{literal}')"  # nosec B608
        )
        target.execute(view_sql)
        target.execute(
            "INSERT INTO table_status (id, table_name, status, creator, hash) "
            "VALUES (?, ?, 'TableStatus.REGISTERED', ?, ?)",
            [str(path), path.stem, creator, f"external:{fingerprint}"],
        )

    if connection is not None:
        register(connection)
        return
    with duckdb.connect(str(db_path)) as owned_connection:
        register(owned_connection)


def add_metadata(
    pneuma: Pneuma,
    metadata_path: Path,
    paths_by_stem: dict[str, Path],
    db_path: Path,
    work_path: Path,
) -> None:
    """Register the catalogue text of every table that has no context yet.

    Args:
        pneuma: The Pneuma instance building the index.
        metadata_path: The lake's metadata JSON (a list of entries).
        paths_by_stem: Registered table files by file stem.
        db_path: Pneuma's ``storage.db``.
        work_path: Folder for the staging CSV Pneuma reads the contexts from.

    Raises:
        RuntimeError: If the metadata file does not hold a JSON list.
    """
    items = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise RuntimeError(f"Expected a JSON list in {metadata_path}")
    existing = context_table_ids(db_path)
    resolved_entries = metadata_entries(items, paths_by_stem)
    logger.info("[metadata] resolved=%d source_entries=%d", len(resolved_entries), len(items))
    rows: list[tuple[str, str]] = []
    normalized_apostrophes = 0
    for path, value in resolved_entries:
        if str(path) in existing:
            continue
        table_id, table_id_count = normalize_metadata_for_pneuma(str(path))
        text, value_count = normalize_metadata_for_pneuma(value)
        normalized_apostrophes += table_id_count + value_count
        rows.append((table_id, text))
    if not rows:
        logger.info("[metadata] no missing metadata entries")
        return
    csv_path = work_path / "lakegen-pneuma-metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("table_id", "value"))
        writer.writerows(rows)
    logger.info("[metadata] normalized_apostrophes=%d staging=%s", normalized_apostrophes, csv_path)
    logger.info("[metadata] adding %d entries", len(rows))
    require_success(pneuma.add_metadata(str(csv_path)), "metadata registration")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser of the bootstrap.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--portal", default="nyc")
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--out-path", type=Path)
    parser.add_argument("--index-name", default="lakegen")
    parser.add_argument("--creator", default="lakegen")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument(
        "--provider",
        choices=("openai", "oci"),
        default="openai",
        help="Remote inference provider (openai also covers Ollama-compatible APIs)",
    )
    parser.add_argument(
        "--openai-base-url",
        help="Use an OpenAI-compatible endpoint instead of loading HF models",
    )
    add_oci_arguments(parser)
    parser.add_argument("--skip-summaries", action="store_true")
    parser.add_argument(
        "--summary-limit",
        type=int,
        help="Process at most this many currently pending table summaries",
    )
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--registration-mode",
        choices=("auto", "external"),
        help="Table registration policy. Defaults to external for UK and auto for other portals",
    )
    parser.add_argument(
        "--external-view-row-limit",
        type=int,
        default=DEFAULT_EXTERNAL_VIEW_ROW_LIMIT,
        help="Maximum rows exposed by Parquet-backed external views during summaries",
    )
    parser.add_argument(
        "--external-view-column-limit",
        type=int,
        default=DEFAULT_EXTERNAL_VIEW_COLUMN_LIMIT,
        help="Maximum representative columns exposed during external-view summaries",
    )
    parser.add_argument(
        "--external-view-sampling",
        choices=("prefix", "reservoir"),
        default="prefix",
        help="Rows an external view exposes during summaries: its first ones, or a "
        "repeatable reservoir sample of the whole table",
    )
    parser.add_argument(
        "--row-sample-value-limit",
        type=int,
        default=0,
        help="Cut text cells longer than this in row samples (0 keeps Pneuma's rows as-is)",
    )
    parser.add_argument(
        "--narration-workers",
        type=int,
        default=1,
        help="Column narrations of one table requested concurrently",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Documents sent per OpenAI-compatible embedding request",
    )
    parser.add_argument(
        "--chroma-insert-batch-size",
        type=int,
        default=DEFAULT_CHROMA_INSERT_BATCH_SIZE,
        help="Documents inserted per Chroma operation",
    )
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject inconsistent or out-of-range options through ``parser.error``."""
    positive = {
        "--limit": args.limit,
        "--summary-limit": args.summary_limit,
        "--external-view-row-limit": args.external_view_row_limit,
        "--embedding-batch-size": args.embedding_batch_size,
        "--chroma-insert-batch-size": args.chroma_insert_batch_size,
        "--narration-workers": args.narration_workers,
    }
    for option, value in positive.items():
        if value is not None and value < 1:
            parser.error(f"{option} must be at least 1")
    if args.external_view_column_limit < 2:
        parser.error("--external-view-column-limit must be at least 2")
    if args.row_sample_value_limit < 0:
        parser.error("--row-sample-value-limit must not be negative")
    if args.chat_base_url and args.provider != "oci":
        parser.error("--chat-base-url needs --provider oci (embeddings stay on OCI)")


def _lake_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve the table folder, metadata file and index folder of the lake.

    Args:
        args: Parsed options; ``--table-dir``, ``--metadata`` and ``--out-path``
            override the defaults under ``data/<portal>`` and ``pneuma-out/<portal>``.

    Returns:
        The table folder, the metadata file and the index folder.
    """
    lake_dir = REPO_ROOT / "data" / args.portal
    is_uk = args.portal.casefold() == "uk"
    default_table_dir = lake_dir / ("clean_datasets" if is_uk else "datasets") / "parquet"
    default_metadata = (
        lake_dir
        / "metadata"
        / ("metadata_retrieved_cleaned.json" if is_uk else "metadata_retrieved_only.json")
    )
    table_dir = (args.table_dir or default_table_dir).resolve()
    metadata_path = (args.metadata or default_metadata).resolve()
    out_path = (args.out_path or REPO_ROOT / "pneuma-out" / args.portal).resolve()
    return table_dir, metadata_path, out_path


def _connect_pneuma(args: argparse.Namespace, out_path: Path) -> Pneuma:
    """Create Pneuma on the chosen provider and record that provider next to the index.

    Args:
        args: Parsed options naming the provider and its models.
        out_path: The index folder.

    Returns:
        Pneuma, with the OCI client in place of its own when ``--provider oci``.
    """
    use_local_model = args.provider == "openai" and args.openai_base_url is None
    if args.provider == "openai" and args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
    pneuma = Pneuma(
        out_path=str(out_path),
        use_local_model=use_local_model,
        openai_api_key=os.environ.get("OPENAI_API_KEY") if not use_local_model else None,
        llm_path=args.llm_model,
        embed_path=args.embedding_model,
    )
    if args.provider == "oci":
        oci_client = client_from_arguments(args)
        pneuma.llm = oci_client
        pneuma.embed_model = oci_client
        provider_record = {
            "provider": "oci",
            "llm_model": args.chat_model if args.chat_base_url else args.oci_llm_model,
            "chat_base_url": args.chat_base_url,
            "external_view_sampling": args.external_view_sampling,
            "external_view_row_limit": args.external_view_row_limit,
            "external_view_column_limit": args.external_view_column_limit,
            "row_sample_value_limit": args.row_sample_value_limit,
            "narration_workers": args.narration_workers,
            "embedding_model": args.oci_embedding_model,
            "embedding_dimensions": args.oci_embedding_dimensions,
        }
    else:
        provider_record = {
            "provider": "openai",
            "llm_model": "gpt-4o-mini" if args.openai_base_url else args.llm_model,
            "embedding_model": (
                "text-embedding-3-small" if args.openai_base_url else args.embedding_model
            ),
            "base_url": args.openai_base_url,
        }
    # Model names and endpoints only: credentials are never written here.
    (out_path / "inference-provider.json").write_text(
        json.dumps(provider_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pneuma


def _log_progress(number: int, total: int, message: str) -> None:
    if number == 1 or number % _PROGRESS_EVERY == 0 or number == total:
        logger.info(message)


def _register_tables(
    pneuma: Pneuma, paths: list[Path], db_path: Path, *, creator: str, registration_mode: str
) -> list[Path]:
    """Register the tables not registered yet.

    In ``external`` mode, Parquet files become views over the file. Otherwise
    Pneuma loads each table, and a Parquet file that exhausts memory falls back
    to a view; a table Pneuma rejects for another reason is skipped.

    Args:
        pneuma: The Pneuma instance building the index.
        paths: Every table file of the lake.
        db_path: Pneuma's ``storage.db``.
        creator: Creator recorded for each table.
        registration_mode: ``external`` or ``auto``.

    Returns:
        The table files that are registered, in the order of ``paths``.
    """
    registered = existing_table_ids(db_path)
    pending = [path for path in paths if str(path) not in registered]
    logger.info("[register] existing=%d pending=%d", len(registered), len(pending))
    external_connection = duckdb.connect(str(db_path)) if registration_mode == "external" else None
    try:
        for number, path in enumerate(pending, 1):
            if external_connection is not None and path.suffix.casefold() == ".parquet":
                register_external_view(
                    db_path, path, creator=creator, connection=external_connection
                )
                _log_progress(
                    number,
                    len(pending),
                    f"[register] external-view {number}/{len(pending)} {path.name}",
                )
                continue
            payload = response_payload(pneuma.add_tables(str(path), creator=creator))
            if payload.get("status") != "SUCCESS":
                message = str(payload.get("message") or "")
                if "Out of Memory Error" in message and path.suffix.casefold() == ".parquet":
                    register_external_view(db_path, path, creator=creator)
                    logger.warning("[register] external-view fallback %s", path.name)
                else:
                    logger.warning("[register] skipped %s: %s", path.name, message)
            _log_progress(number, len(pending), f"[register] {number}/{len(pending)}")
    finally:
        if external_connection is not None:
            external_connection.close()
    registered = existing_table_ids(db_path)
    active_paths = [path for path in paths if str(path) in registered]
    logger.info(
        "[register] usable=%d excluded=%d", len(active_paths), len(paths) - len(active_paths)
    )
    return active_paths


def _patch_summarizer(args: argparse.Namespace) -> None:
    """Apply the narration concurrency and row-sample cap the options ask for."""
    import pneuma.summarizer.summarizer as pneuma_summarizer

    if args.narration_workers > 1:
        pneuma_summarizer.prompt_openai_llm = concurrent_prompt_openai_llm(args.narration_workers)
    if args.row_sample_value_limit > 0:
        summarizer = pneuma_summarizer.Summarizer
        generate_row_samples = summarizer._Summarizer__generate_row_samples
        limit = args.row_sample_value_limit
        summarizer._Summarizer__generate_row_samples = lambda self, frame: generate_row_samples(
            self, cap_long_cells(frame, limit)
        )


def _summarize_tables(
    pneuma: Pneuma, args: argparse.Namespace, db_path: Path, active_paths: list[Path]
) -> None:
    """Summarize the registered tables still pending, up to ``--summary-limit``.

    Args:
        pneuma: The Pneuma instance building the index.
        args: Parsed options: view bounds, narration workers, row-sample cap.
        db_path: Pneuma's ``storage.db``.
        active_paths: The registered table files.
    """
    bounded_views = configure_bounded_external_views(
        db_path,
        row_limit=args.external_view_row_limit,
        column_limit=args.external_view_column_limit,
        sampling=args.external_view_sampling,
    )
    _patch_summarizer(args)
    logger.info(
        "[summarize] bounded_external_views=%d row_limit=%d column_limit=%d sampling=%s "
        "narration_workers=%d row_sample_value_limit=%d",
        bounded_views,
        args.external_view_row_limit,
        args.external_view_column_limit,
        args.external_view_sampling,
        args.narration_workers,
        args.row_sample_value_limit,
    )
    allowed = {str(path) for path in active_paths}
    pending = [table_id for table_id in pending_summary_ids(db_path) if table_id in allowed]
    if args.summary_limit is not None:
        pending = pending[: args.summary_limit]
    logger.info("[summarize] pending=%d", len(pending))
    for number, table_id in enumerate(pending, 1):
        name = Path(table_id).name
        logger.info("[summarize] starting %d/%d %s", number, len(pending), name)
        require_success(pneuma.summarize(table_id), f"summary for {table_id}")
        logger.info("[summarize] %d/%d %s", number, len(pending), name)


def main() -> None:
    """Build or resume the index of one lake.

    Raises:
        RuntimeError: If the table folder holds no CSV or Parquet file, or a Pneuma step fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
    parser = build_parser()
    args = parser.parse_args()
    _validate_arguments(parser, args)
    tune_duckdb_connections()
    table_dir, metadata_path, out_path = _lake_paths(args)
    out_path.mkdir(parents=True, exist_ok=True)
    paths = table_paths(table_dir)[: args.limit]
    if not paths:
        raise RuntimeError(f"No CSV or Parquet tables found in {table_dir}")
    registration_mode = args.registration_mode or (
        "external" if args.portal.casefold() == "uk" else "auto"
    )
    configure_pneuma_indexing(
        embedding_batch_size=args.embedding_batch_size,
        chroma_insert_batch_size=args.chroma_insert_batch_size,
    )
    logger.info(
        "[setup] portal=%s tables=%d out=%s index=%s registration_mode=%s",
        args.portal,
        len(paths),
        out_path,
        args.index_name,
        registration_mode,
    )
    pneuma = _connect_pneuma(args, out_path)
    require_success(pneuma.setup(), "Pneuma setup")
    db_path = out_path / "storage.db"
    active_paths = _register_tables(
        pneuma, paths, db_path, creator=args.creator, registration_mode=registration_mode
    )
    if not args.skip_summaries:
        _summarize_tables(pneuma, args, db_path, active_paths)
    if not args.skip_metadata and metadata_path.is_file():
        add_metadata(
            pneuma, metadata_path, {path.stem: path for path in active_paths}, db_path, out_path
        )
    if not args.skip_index:
        logger.info("[index] generating %s", args.index_name)
        require_success(
            pneuma.generate_index(args.index_name, tuple(map(str, active_paths))),
            "index generation",
        )
    logger.info("[done] Pneuma bootstrap completed")


if __name__ == "__main__":
    main()
