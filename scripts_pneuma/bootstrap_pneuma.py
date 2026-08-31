"""Build a resumable Pneuma index from LakeGen's existing portal tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import duckdb
from pneuma import Pneuma


SUPPORTED_SUFFIXES = {".csv", ".parquet"}
DEFAULT_EXTERNAL_VIEW_ROW_LIMIT = 10_000
BOUNDED_TABLE_IDS: set[str] = set()
BOUNDED_TABLE_ROW_LIMIT = DEFAULT_EXTERNAL_VIEW_ROW_LIMIT


def tune_duckdb_connections() -> None:
    """Apply low-memory settings to every connection opened inside Pneuma."""
    original_connect = duckdb.connect

    class BoundedConnection:
        def __init__(self, connection: Any):
            self._connection = connection

        def __enter__(self):
            self._connection.__enter__()
            return self

        def __exit__(self, *args: Any):
            return self._connection.__exit__(*args)

        def __getattr__(self, name: str):
            return getattr(self._connection, name)

        def sql(self, query: str, *args: Any, **kwargs: Any):
            match = re.fullmatch(
                r"\s*SELECT\s+\*\s+FROM\s+'((?:''|[^'])+)'\s*",
                query,
                flags=re.IGNORECASE,
            )
            if match:
                table_id = match.group(1).replace("''", "'")
                if table_id in BOUNDED_TABLE_IDS:
                    query = f"{query.rstrip()} LIMIT {BOUNDED_TABLE_ROW_LIMIT}"
            return self._connection.sql(query, *args, **kwargs)

    def connect(*args: Any, **kwargs: Any):
        config = dict(kwargs.pop("config", {}) or {})
        config.setdefault("threads", "4")
        config.setdefault("preserve_insertion_order", "false")
        return BoundedConnection(
            original_connect(*args, config=config, **kwargs)
        )

    duckdb.connect = connect  # type: ignore[assignment]


def response_payload(raw: str | dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict):
        raise RuntimeError("Pneuma returned a non-object response")
    return payload


def require_success(raw: str | dict[str, Any], operation: str) -> dict[str, Any]:
    payload = response_payload(raw)
    if payload.get("status") != "SUCCESS":
        raise RuntimeError(f"{operation} failed: {payload.get('message', payload)}")
    return payload


def table_paths(table_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in table_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in SUPPORTED_SUFFIXES
    )


def existing_table_ids(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return {row[0] for row in connection.sql("SELECT id FROM table_status").fetchall()}


def register_external_view(
    db_path: Path, path: Path, *, creator: str
) -> None:
    """Register a Parquet-backed view when a full materialization exhausts RAM."""
    identifier = str(path).replace('"', '""')
    literal = str(path).replace("'", "''")
    stat = path.stat()
    fingerprint = hashlib.sha256(
        f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    with duckdb.connect(str(db_path)) as connection:
        connection.execute(
            f'CREATE OR REPLACE VIEW "{identifier}" AS '
            f"SELECT * FROM read_parquet('{literal}')"
        )
        connection.execute(
            "INSERT INTO table_status (id, table_name, status, creator, hash) "
            "VALUES (?, ?, 'TableStatus.REGISTERED', ?, ?)",
            [str(path), path.stem, creator, f"external:{fingerprint}"],
        )


def configure_bounded_external_views(
    db_path: Path, *, row_limit: int
) -> int:
    """Limit direct Parquet reads for external views during summarization."""
    global BOUNDED_TABLE_IDS, BOUNDED_TABLE_ROW_LIMIT
    with duckdb.connect(str(db_path), read_only=True) as connection:
        table_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM table_status WHERE hash LIKE 'external:%'"
            ).fetchall()
        ]
    BOUNDED_TABLE_IDS = {
        table_id
        for table_id in table_ids
        if Path(table_id).is_file()
        and Path(table_id).suffix.casefold() == ".parquet"
    }
    BOUNDED_TABLE_ROW_LIMIT = row_limit
    return len(BOUNDED_TABLE_IDS)


def pending_summary_ids(db_path: Path) -> list[str]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return [
            row[0]
            for row in connection.sql(
                "SELECT id FROM table_status "
                "WHERE status = 'TableStatus.REGISTERED' ORDER BY id"
            ).fetchall()
        ]


def context_table_ids(db_path: Path) -> set[str]:
    with duckdb.connect(str(db_path), read_only=True) as connection:
        return {
            row[0]
            for row in connection.sql(
                "SELECT DISTINCT table_id FROM table_contexts"
            ).fetchall()
        }


def metadata_text(item: dict[str, Any]) -> str:
    resource = item.get("resource") if isinstance(item.get("resource"), dict) else {}
    classification = (
        item.get("classification")
        if isinstance(item.get("classification"), dict)
        else {}
    )
    parts = [
        str(resource.get("name") or "").strip(),
        str(resource.get("description") or "").strip(),
        "Tags: " + ", ".join(map(str, classification.get("tags") or [])),
        "Columns: " + ", ".join(map(str, resource.get("columns_name") or [])),
    ]
    return "\n".join(part for part in parts if part and part not in {"Tags: ", "Columns: "})


def add_metadata(
    pneuma: Pneuma,
    metadata_path: Path,
    paths_by_stem: dict[str, Path],
    db_path: Path,
    work_path: Path,
) -> None:
    items = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise RuntimeError(f"Expected a JSON list in {metadata_path}")
    existing = context_table_ids(db_path)
    rows: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        resource = item.get("resource")
        resource_id = resource.get("id") if isinstance(resource, dict) else None
        path = paths_by_stem.get(str(resource_id))
        if path is None or str(path) in existing:
            continue
        value = metadata_text(item)
        if value:
            rows.append((str(path), value))
    if not rows:
        print("[metadata] no missing metadata entries", flush=True)
        return
    csv_path = work_path / "lakegen-pneuma-metadata.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("table_id", "value"))
        writer.writerows(rows)
    print(f"[metadata] adding {len(rows)} entries", flush=True)
    require_success(pneuma.add_metadata(str(csv_path)), "metadata registration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portal", default="nyc")
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--out-path", type=Path)
    parser.add_argument("--index-name", default="lakegen")
    parser.add_argument("--creator", default="lakegen")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument(
        "--openai-base-url",
        help="Use an OpenAI-compatible endpoint instead of loading HF models",
    )
    parser.add_argument("--skip-summaries", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--external-view-row-limit",
        type=int,
        default=DEFAULT_EXTERNAL_VIEW_ROW_LIMIT,
        help="Maximum rows exposed by Parquet-backed external views during summaries",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tune_duckdb_connections()
    root = Path(__file__).resolve().parents[1]
    default_dataset_dir = (
        root / "data" / args.portal
        / ("clean_datasets" if args.portal.casefold() == "uk" else "datasets")
        / "parquet"
    )
    table_dir = (args.table_dir or default_dataset_dir).resolve()
    metadata_path = (
        args.metadata
        or root / "data" / args.portal / "metadata" / (
            "metadata_retrieved_cleaned.json"
            if args.portal.casefold() == "uk"
            else "metadata_retrieved_only.json"
        )
    ).resolve()
    out_path = (args.out_path or root / "pneuma-out" / args.portal).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    paths = table_paths(table_dir)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No CSV or Parquet tables found in {table_dir}")
    if args.external_view_row_limit <= 0:
        raise ValueError("--external-view-row-limit must be greater than zero")

    print(
        f"[setup] portal={args.portal} tables={len(paths)} out={out_path} "
        f"index={args.index_name}",
        flush=True,
    )
    use_local_model = args.openai_base_url is None
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url.rstrip("/")
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
    pneuma = Pneuma(
        out_path=str(out_path),
        use_local_model=use_local_model,
        openai_api_key=os.environ.get("OPENAI_API_KEY") if not use_local_model else None,
        llm_path=args.llm_model,
        embed_path=args.embedding_model,
    )
    require_success(pneuma.setup(), "Pneuma setup")
    db_path = out_path / "storage.db"

    registered = existing_table_ids(db_path)
    pending = [path for path in paths if str(path) not in registered]
    print(f"[register] existing={len(registered)} pending={len(pending)}", flush=True)
    for number, path in enumerate(pending, 1):
        payload = response_payload(pneuma.add_tables(str(path), creator=args.creator))
        if payload.get("status") != "SUCCESS":
            message = str(payload.get("message") or "")
            if "Out of Memory Error" in message and path.suffix.casefold() == ".parquet":
                register_external_view(
                    db_path,
                    path,
                    creator=args.creator,
                )
                print(
                    f"[register] external-view fallback {path.name}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[register] skipped {path.name}: {message}",
                    file=sys.stderr,
                    flush=True,
                )
        if number == 1 or number % 25 == 0 or number == len(pending):
            print(f"[register] {number}/{len(pending)}", flush=True)

    registered = existing_table_ids(db_path)
    active_paths = [path for path in paths if str(path) in registered]
    print(
        f"[register] usable={len(active_paths)} excluded={len(paths) - len(active_paths)}",
        flush=True,
    )

    if not args.skip_summaries:
        bounded_views = configure_bounded_external_views(
            db_path, row_limit=args.external_view_row_limit
        )
        print(
            f"[summarize] bounded_external_views={bounded_views} "
            f"row_limit={args.external_view_row_limit}",
            flush=True,
        )
        pending_summaries = pending_summary_ids(db_path)
        allowed = {str(path) for path in active_paths}
        pending_summaries = [item for item in pending_summaries if item in allowed]
        print(f"[summarize] pending={len(pending_summaries)}", flush=True)
        for number, table_id in enumerate(pending_summaries, 1):
            print(
                f"[summarize] starting {number}/{len(pending_summaries)} "
                f"{Path(table_id).name}",
                flush=True,
            )
            require_success(pneuma.summarize(table_id), f"summary for {table_id}")
            print(
                f"[summarize] {number}/{len(pending_summaries)} {Path(table_id).name}",
                flush=True,
            )

    if not args.skip_metadata and metadata_path.is_file():
        add_metadata(
            pneuma,
            metadata_path,
            {path.stem: path for path in active_paths},
            db_path,
            out_path,
        )

    if not args.skip_index:
        print(f"[index] generating {args.index_name}", flush=True)
        require_success(
            pneuma.generate_index(args.index_name, tuple(map(str, active_paths))),
            "index generation",
        )
    print("[done] Pneuma bootstrap completed", flush=True)


if __name__ == "__main__":
    main()
