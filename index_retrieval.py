"""Create versioned table embeddings in a Solr core.

Example:
    # Safe default: validates and embeds without writing to Solr.
    uv run python index_retrieval.py --core bologna --ensure-schema \
        --metadata-source solr-ready-metadata.json

    # Apply only after reviewing the dry-run, with a local backup first.
    uv run python index_retrieval.py --core bologna --ensure-schema \
        --metadata-source solr-ready-metadata.json --backup backup.json --apply
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client_solr import LocalSolrClient
from lakegen.retrieval import RetrievalConfig, SolrEmbeddingIndexer
from lakegen.retrieval.embeddings import get_embedding_model


def load_source_documents(path: Path) -> list[dict[str, Any]]:
    """Load complete, Solr-ready documents from the canonical metadata export."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("documents"), list):
        rows = payload["documents"]
    elif isinstance(payload, dict):
        rows = payload.get("response", {}).get("docs")
    else:
        rows = None
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            "Metadata source must be a JSON list of complete Solr documents, "
            "an object with a documents list, or a Solr response export"
        )
    return rows


def validate_source_coverage(
    source_documents: list[dict[str, Any]],
    current_documents: list[dict[str, Any]],
    *,
    unique_key: str,
    vector_field: str,
) -> None:
    """Ensure a source-based replacement cannot drop current stored fields."""
    ignored = {"_version_", vector_field, "representation_version", "embedding_model"}

    def keyed(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(rows):
            value = row.get(unique_key)
            if value is None or not str(value).strip():
                raise RuntimeError(f"{label} document {index} has no {unique_key!r}")
            key = str(value)
            if key in result:
                raise RuntimeError(f"Duplicate {unique_key!r} in {label}: {key!r}")
            result[key] = row
        return result

    source_by_key = keyed(source_documents, "source")
    current_by_key = keyed(current_documents, "Solr")
    if source_by_key.keys() != current_by_key.keys():
        missing = sorted(current_by_key.keys() - source_by_key.keys())[:10]
        extra = sorted(source_by_key.keys() - current_by_key.keys())[:10]
        raise RuntimeError(
            "Source/Solr document IDs differ; refusing full replacement. "
            f"Missing from source: {missing}; extra in source: {extra}"
        )
    for key, current in current_by_key.items():
        required_fields = set(current) - ignored
        absent = sorted(required_fields - set(source_by_key[key]))
        if absent:
            raise RuntimeError(
                f"Source document {key!r} would lose stored fields: {absent}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely index metadata-v1 embeddings for Solr KNN retrieval."
    )
    parser.add_argument("--core", required=True)
    parser.add_argument(
        "--solr-base-url",
        "--solr-url",
        dest="solr_base_url",
        default=os.environ.get("SOLR_BASE_URL", "http://localhost:8983/solr"),
        help="Solr base URL (default: SOLR_BASE_URL or localhost)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ensure-schema", action="store_true")
    parser.add_argument(
        "--metadata-source",
        type=Path,
        help="Canonical JSON export containing complete Solr-ready documents",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        help="Required with --apply; receives schema and current stored documents",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write after validation; without this flag the command is a dry-run",
    )
    args = parser.parse_args()

    if args.apply and (args.metadata_source is None or args.backup is None):
        parser.error("--apply requires both --metadata-source and --backup")

    config = RetrievalConfig.from_env()
    embedding_model = get_embedding_model(
        config.embedding_model, config.embedding_base_url
    )
    solr = LocalSolrClient(core=args.core, base_url=args.solr_base_url)
    schema = solr.schema().get("schema", {})
    unique_key = schema.get("uniqueKey")
    if not unique_key:
        raise RuntimeError("Solr schema does not define a uniqueKey")

    source_documents = (
        load_source_documents(args.metadata_source)
        if args.metadata_source is not None
        else None
    )
    current_documents = None
    if source_documents is not None:
        current_documents = list(
            solr.iter_documents(
                fields=("*",),
                batch_size=max(args.batch_size, 100),
                sort_field=unique_key,
                restore_columns=False,
            )
        )
        validate_source_coverage(
            source_documents,
            current_documents,
            unique_key=unique_key,
            vector_field=config.vector_field,
        )
    if args.apply:
        assert current_documents is not None and args.backup is not None
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(
            json.dumps(
                {"schema": schema, "documents": current_documents},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    indexer = SolrEmbeddingIndexer(
        solr,
        config,
        embedding_model,
    )
    summary = indexer.run(
        batch_size=args.batch_size,
        create_schema=args.ensure_schema,
        dry_run=not args.apply,
        source_documents=source_documents,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
