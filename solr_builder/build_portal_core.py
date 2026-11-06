r"""Build a LakeGen portal from a fresh CKAN/ODS crawl into a NEW Solr core.

For one portal crawl (``<crawl>/metadata/metadata_retrieved_only.json`` plus
``<crawl>/datasets/csv``) this:

1. normalizes the raw portal metadata with orqa's ``normalize_file`` -- the
   normalizer the current portal cores were built from (title, description
   and HTML cleanup included);
2. builds a LakeGen data folder ``<out>/datasets/csv`` that LakeGen can read in
   full: readable files are symlinked (the crawl is never modified),
   non-UTF-8 files are re-encoded to UTF-8 copies (cp1252 or Latin-1 decoded
   byte for byte, nothing else changed), and empty or still unreadable files
   are left out -- a table LakeGen cannot open must not be retrievable;
3. writes Solr-ready documents for exactly the tables served
   (``<out>/metadata/solr_documents.json``): columns flattened into aligned
   ``columns.*`` arrays, dates in Solr form, ``indexed_ts`` set;
4. with ``--apply``: creates the core on the ``nyc`` configset (the fields of
   the portal configsets plus the vector field) and posts the documents.

It refuses to write to an existing core that already holds documents, so a
production core can never be overwritten. Embeddings are a separate, validated
step with ``python -m solr_builder.index``, on this core's own export. Nothing
points LakeGen at the result: see the printed next steps.

  .venv/bin/python -m solr_builder.build_portal_core --portal valencia --source ckan \
      --crawl <crawls>/ckan/valencia --core valencia_v2 --out <lakes>/valencia \
      --orqa-src <orqa checkout>/src [--apply]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from lakegen.config.settings import get_settings
from lakegen.retrieval.solr.catalogue import match_local_csv
from lakegen.utils.table_io import read_table
from solr_builder.portal import solr_timestamp

logger = logging.getLogger(__name__)

CONFIGSET = "nyc"
DATE_FIELDS = ("created_at", "modified_at")
# Seconds a Solr admin or select request may take; posting the documents gets longer.
REQUEST_TIMEOUT = 60
POST_TIMEOUT = 300


def readable(path: Path) -> bool:
    """Tell whether LakeGen's table reader opens the file."""
    try:
        read_table(path, nrows=20)
    except Exception:
        logger.debug("unreadable table %s", path, exc_info=True)
        return False
    return True


def utf8_copy(source: Path, target: Path) -> str | None:
    """Re-encode a non-UTF-8 CSV to a UTF-8 copy LakeGen can read.

    Args:
        source: The crawled file.
        target: Where the copy goes; removed again when no encoding works.

    Returns:
        The encoding the file was decoded with, or None when none gave a readable copy.
    """
    raw = source.read_bytes()
    for encoding in ("cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        target.write_text(text, encoding="utf-8")
        if readable(target):
            return encoding
        target.unlink()
    return None


def build_folder(crawl_csv: Path, out_csv: Path) -> tuple[list[str], dict[str, Any]]:
    """Serve every table LakeGen can read; report what was converted or left out.

    Args:
        crawl_csv: The crawl's CSV folder, never modified.
        out_csv: The served folder: symlinks to readable files, UTF-8 copies of the others.

    Returns:
        The served file names, and a report of the symlinked, re-encoded and left-out files.
    """
    out_csv.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"symlinked": 0, "re_encoded": {}, "left_out": {}}
    served = []
    for source in sorted(crawl_csv.glob("*.csv")):
        target = out_csv / source.name
        if target.is_symlink() or target.exists():
            target.unlink()
        if source.stat().st_size == 0:
            report["left_out"][source.name] = "empty file"
            continue
        if readable(source):
            target.symlink_to(source)
            report["symlinked"] += 1
        else:
            encoding = utf8_copy(source, target)
            if encoding is None:
                report["left_out"][source.name] = "unreadable"
                continue
            report["re_encoded"][source.name] = encoding
        served.append(source.name)
    return served, report


def solr_document(record: dict[str, Any], indexed_ts: str) -> dict[str, Any]:
    """Turn one normalized record into a flat Solr document, its columns kept aligned.

    Args:
        record: A record of orqa's normalized metadata.
        indexed_ts: The build's timestamp, set on every document.

    Returns:
        The Solr document.
    """
    document = {
        key: value
        for key, value in record.items()
        if key != "columns" and value not in (None, "", [])
    }
    for field in DATE_FIELDS:
        if field in document:
            stamp = solr_timestamp(document[field])
            if stamp:
                document[field] = stamp
            else:
                document.pop(field)
    columns = [column for column in record.get("columns") or [] if isinstance(column, dict)]
    if columns:
        # One entry per column in every array: Solr drops empty strings, which
        # would shift the arrays LakeGen zips back together out of line. So a
        # missing name, label or type falls back, and descriptions are kept
        # only when every column has one.
        names = [
            str(column.get("name") or f"column_{index}") for index, column in enumerate(columns)
        ]
        document["columns.name"] = names
        document["columns.label"] = [
            str(column.get("label") or name) for column, name in zip(columns, names, strict=True)
        ]
        document["columns.type"] = [str(column.get("type") or "text") for column in columns]
        if all(str(column.get("description") or "").strip() for column in columns):
            document["columns.description"] = [str(column["description"]) for column in columns]
    document["indexed_ts"] = indexed_ts
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--portal", required=True)
    parser.add_argument("--source", required=True, choices=("ckan", "ods"))
    parser.add_argument("--crawl", required=True, type=Path)
    parser.add_argument("--core", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--solr", default=get_settings().solr_base_url)
    parser.add_argument(
        "--orqa-src",
        required=True,
        type=Path,
        help="the src folder of an orqa checkout, whose normalize_metadata is used",
    )
    parser.add_argument("--apply", action="store_true")
    return parser


def _link_metadata(crawl: Path, metadata_dir: Path) -> None:
    """Link the crawl's raw metadata files into the new lake's metadata folder."""
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "metadata_retrieved_only.json"):
        link = metadata_dir / name
        if not link.exists() and (crawl / "metadata" / name).exists():
            link.symlink_to(crawl / "metadata" / name)


def _documents(
    records: list[dict[str, Any]], served: list[str], indexed_ts: str
) -> tuple[list[dict[str, Any]], int, set[str]]:
    """Build the documents of the records that have a served file.

    Args:
        records: orqa's normalized records.
        served: The served file names.
        indexed_ts: The build's timestamp.

    Returns:
        The documents, the number of records without a served file, and the
        served files that a record claimed.

    Raises:
        SystemExit: When two records map to the same file.
    """
    documents, unmatched, claimed = [], 0, set()
    for record in records:
        table = match_local_csv(record, served)
        if table is None:
            unmatched += 1
            continue
        if table in claimed:
            raise SystemExit(f"two records map to {table}")
        claimed.add(table)
        documents.append(solr_document(record, indexed_ts))
    return documents, unmatched, claimed


def _post(args: argparse.Namespace, documents: list[dict[str, Any]]) -> None:
    """Create the core if needed and post the documents, or say what ``--apply`` would do.

    Args:
        args: Parsed options: the Solr URL, the core and ``--apply``.
        documents: The documents to post.

    Raises:
        SystemExit: When the core already holds documents, posting fails, or
            the core does not hold exactly the posted documents afterwards.
    """
    status = requests.get(
        f"{args.solr}/admin/cores",
        params={"action": "STATUS", "core": args.core, "wt": "json"},
        timeout=REQUEST_TIMEOUT,
    ).json()
    existing = status.get("status", {}).get(args.core) or {}
    if existing and existing.get("index", {}).get("numDocs"):
        raise SystemExit(f"core {args.core!r} already holds documents; refusing to write to it")
    if not args.apply:
        print(
            f"dry run: would {'reuse the empty' if existing else 'create'} core {args.core!r} "
            f"(configSet={CONFIGSET}) and post {len(documents)} documents"
        )
        return
    if not existing:
        response = requests.get(
            f"{args.solr}/admin/cores",
            params={"action": "CREATE", "name": args.core, "configSet": CONFIGSET, "wt": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    response = requests.post(
        f"{args.solr}/{args.core}/update",
        params={"commit": "true", "wt": "json"},
        json=documents,
        timeout=POST_TIMEOUT,
    )
    if response.status_code != 200:
        raise SystemExit(f"posting failed: {response.status_code} {response.text[:500]}")
    count = requests.get(
        f"{args.solr}/{args.core}/select",
        params={"q": "*:*", "rows": 0, "wt": "json"},
        timeout=REQUEST_TIMEOUT,
    ).json()["response"]["numFound"]
    print(f"core {args.core}: {count} documents indexed (expected {len(documents)})")
    if count != len(documents):
        raise SystemExit("document count mismatch")


def main() -> None:
    """Build the lake folder and the documents, then post them with ``--apply``."""
    parser = _parser()
    args = parser.parse_args()
    if args.core == args.portal:
        parser.error("the new core must not reuse the portal's current core name")
    # orqa is a separate checkout, not an installed package.
    sys.path.insert(0, str(args.orqa_src))
    from orqa.normalize_metadata import normalize_file

    metadata_dir = args.out / "metadata"
    _link_metadata(args.crawl, metadata_dir)
    records = normalize_file(
        args.crawl / "metadata" / "metadata_retrieved_only.json",
        args.source,
        metadata_dir / "normalized_metadata.json",
    )
    served, report = build_folder(args.crawl / "datasets" / "csv", args.out / "datasets" / "csv")
    indexed_ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    documents, unmatched, claimed = _documents(records, served, indexed_ts)
    (metadata_dir / "solr_documents.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=1)
    )
    unclaimed = len(set(served) - claimed)
    (args.out / "BUILD.json").write_text(
        json.dumps(
            {
                "portal": args.portal,
                "source": args.source,
                "crawl": str(args.crawl),
                "core": args.core,
                "configset": CONFIGSET,
                "built_at": indexed_ts,
                "records": len(records),
                "documents": len(documents),
                "records_without_a_served_file": unmatched,
                "tables_served": len(served),
                "tables_without_a_record": unclaimed,
                **report,
            },
            indent=2,
        )
    )
    print(
        f"{args.portal}: {len(records)} normalized records -> {len(documents)} documents for "
        f"{len(served)} served tables (symlinked {report['symlinked']}, "
        f"re-encoded {len(report['re_encoded'])}, left out {len(report['left_out'])}); "
        f"records without a served file {unmatched}; served tables without a record {unclaimed}"
    )
    _post(args, documents)


if __name__ == "__main__":
    main()
