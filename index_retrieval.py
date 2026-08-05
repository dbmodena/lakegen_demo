"""Create versioned table embeddings in a Solr core.

Example:
    uv run python index_retrieval.py --core bologna --ensure-schema
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.client_solr import LocalSolrClient
from lakegen.retrieval import RetrievalConfig, SolrEmbeddingIndexer
from lakegen.retrieval.embeddings import get_embedding_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index metadata-v1 embeddings for Solr KNN retrieval."
    )
    parser.add_argument("--core", required=True)
    parser.add_argument("--solr-url", default="http://localhost:8983/solr")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ensure-schema", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = RetrievalConfig.from_env()
    embedding_model = get_embedding_model(
        config.embedding_model, config.embedding_base_url
    )
    indexer = SolrEmbeddingIndexer(
        LocalSolrClient(core=args.core, base_url=args.solr_url),
        config,
        embedding_model,
    )
    summary = indexer.run(
        batch_size=args.batch_size,
        create_schema=args.ensure_schema,
        dry_run=args.dry_run,
    )
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
