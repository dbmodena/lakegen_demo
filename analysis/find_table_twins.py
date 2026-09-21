#!/usr/bin/env python3
"""Find twin tables across a whole parquet lake.

Two tiers, cheapest first:

1. byte_identical: files with the same sha256 (only files that share a size with
   another file are hashed).
2. content_identical: tables that are not byte-identical but hold the same rows and
   the same columns, ignoring row order and column order. Candidates are grouped by
   (row count, column names) from the parquet footers, then confirmed with a content
   digest over the normalised values (see check_table_twins._normalise).

Identical content does not make two tables interchangeable: the same numbers can
describe two different things (ticket sales of a football match and a rugby match at
one stadium). Twins are backups or re-uploads and nothing else, so every cluster is then
split by description (Solr description fields, CKAN package/resource fields incl. the period
the file covers; see check_table_twins.descriptive_signature): members are twins only if
their descriptions are identical and nothing shows they are different files. One file listed
under different labels is not a twin; it goes to `review`. The result is `twin_groups`; a
cluster with several groups is marked `metadata_split`.

Tables with renamed columns are out of scope here; use check_table_twins.py on a
suspected pair for that. Trivial tables (few cells) are flagged, not dropped,
because two tiny tables being equal says little.

Usage:
    python analysis/find_table_twins.py --out twins.json [--parquet-dir DIR]
    python analysis/find_table_twins.py --refine scan.json --out twins.json   # re-split only
    (add --no-metadata to skip the metadata split)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_table_twins import (  # noqa: E402  (same normalisation and metadata rules as the pairwise checker)
    _normalise, descriptive_signature, fetch_solr_docs, file_relation, load_ckan_index, member_file_url,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _footer(path: Path) -> tuple[int, tuple[str, ...]]:
    metadata = pq.read_metadata(path)
    return metadata.num_rows, tuple(sorted(metadata.schema.to_arrow_schema().names))


def _content_digest(path: Path) -> str:
    """Digest of the table ignoring row order and column order."""
    df = pq.read_table(path).to_pandas()
    order = sorted(range(df.shape[1]), key=lambda i: str(df.columns[i]))
    columns = [_normalise(df.iloc[:, i]).reset_index(drop=True) for i in order]
    frame = pd.concat(columns, axis=1)
    frame.columns = range(frame.shape[1])
    rows = np.sort(pd.util.hash_pandas_object(frame, index=False).to_numpy())
    digest = hashlib.sha256(rows.tobytes())
    digest.update("\x1f".join(str(df.columns[i]) for i in order).encode())
    return digest.hexdigest()


def scan(parquet_dir: Path, *, workers: int, max_content_mb: float, min_cells: int) -> dict[str, Any]:
    started = time.time()
    sizes = {e.name: e.stat().st_size for e in os.scandir(parquet_dir) if e.name.endswith(".parquet")}
    by_size: dict[int, list[str]] = defaultdict(list)
    for name, size in sizes.items():
        by_size[size].append(name)
    to_hash = [n for names in by_size.values() if len(names) > 1 for n in names]

    with ThreadPoolExecutor(workers) as pool:
        digests = dict(zip(to_hash, pool.map(lambda n: _sha256(parquet_dir / n), to_hash)))
    by_hash: dict[str, list[str]] = defaultdict(list)
    for name, digest in digests.items():
        by_hash[digest].append(name)
    byte_clusters = [sorted(m) for m in by_hash.values() if len(m) > 1]
    in_byte_cluster = {n for members in byte_clusters for n in members}
    t_bytes = time.time()

    # Tier 2: one representative per byte cluster, plus every unclustered file.
    representatives = {members[0]: members for members in byte_clusters}
    candidates = sorted(set(representatives) | (set(sizes) - in_byte_cluster))
    with ThreadPoolExecutor(workers) as pool:
        footers = dict(zip(candidates, pool.map(lambda n: _footer(parquet_dir / n), candidates)))
    by_shape: dict[tuple, list[str]] = defaultdict(list)
    for name, (rows, names) in footers.items():
        if rows > 0 and names:
            by_shape[(rows, names)].append(name)
    shape_groups = [m for m in by_shape.values() if len(m) > 1]
    to_digest = [n for m in shape_groups for n in m if sizes[n] <= max_content_mb * 1e6]
    skipped_large = [n for m in shape_groups for n in m if sizes[n] > max_content_mb * 1e6]
    with ThreadPoolExecutor(max(1, workers // 2)) as pool:
        content = dict(zip(to_digest, pool.map(lambda n: _content_digest(parquet_dir / n), to_digest)))
    by_content: dict[str, list[str]] = defaultdict(list)
    for name, digest in content.items():
        by_content[digest].append(name)

    # Final clusters partition the twin tables: a content cluster absorbs the byte
    # clusters (and single files) it joins; the remaining byte clusters stay as they are.
    clusters: list[dict[str, Any]] = []
    absorbed: set[str] = set()
    for group in by_content.values():
        if len(group) > 1:
            absorbed.update(group)
            clusters.append({
                "tier": "content_identical",
                "members": sorted({m for n in group for m in representatives.get(n, [n])}),
            })
    for members in byte_clusters:
        if members[0] not in absorbed:
            clusters.append({"tier": "byte_identical", "members": members})
    for i, cluster in enumerate(clusters):
        rows, names = footers.get(cluster["members"][0]) or _footer(parquet_dir / cluster["members"][0])
        cluster.update({
            "id": i, "size": len(cluster["members"]), "rows": rows, "cols": len(names),
            "trivial": rows * len(names) <= min_cells,
            "dataset_count": len({m.split("___")[0] for m in cluster["members"]}),
        })
    return {
        "parquet_dir": str(parquet_dir),
        "summary": {
            "files": len(sizes),
            "hashed_files": len(to_hash),
            "byte_identical_clusters": len(byte_clusters),
            "content_identical_clusters": sum(c["tier"] == "content_identical" for c in clusters),
            "shape_candidate_groups": len(shape_groups),
            "content_digested_files": len(to_digest),
            "skipped_large_files": skipped_large,
            "seconds_bytes": round(t_bytes - started, 1),
            "seconds_total": round(time.time() - started, 1),
        },
        "clusters": clusters,
    }


def refine_by_metadata(clusters: list[dict[str, Any]], *, solr: tuple[str, str] | None,
                       ckan: dict[str, tuple[dict, dict]]) -> None:
    """Split each content cluster into twin groups (in place).

    Twins are backups or re-uploads: identical content, identical description (publisher,
    title, period, columns, notes) and no sign of being different files. Members joined by
    that relation form a twin group. Two weaker relations are kept apart for review and never
    make twins: `same_file_different_labels` (one published file listed with different
    descriptions) and `same_description_different_file_names`.
    """
    resource = lambda member: member.removesuffix(".parquet").split("___")[-1]  # noqa: E731
    docs = fetch_solr_docs(*solr, [resource(m) for c in clusters for m in c["members"]]) if solr else {}
    for cluster in clusters:
        members = cluster["members"]
        signature = {m: descriptive_signature(docs.get(resource(m)), ckan.get(resource(m))) for m in members}
        url = {m: member_file_url(docs.get(resource(m)), ckan.get(resource(m))) for m in members}
        parent = {m: m for m in members}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        evidence: dict[frozenset[str], str] = {}
        relabelled: list[list[str]] = []
        renamed: list[list[str]] = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                relation = file_relation(url[a], url[b])
                same_description = bool(signature[a]) and signature[a] == signature[b]
                if same_description and relation != "different_file_name":
                    evidence[frozenset((a, b))] = relation
                    parent[find(a)] = find(b)
                elif same_description:
                    renamed.append([a, b])
                elif relation == "same_source":
                    relabelled.append([a, b])
        components: dict[str, list[str]] = defaultdict(list)
        for m in members:
            components[find(m)].append(m)
        groups = sorted((sorted(g) for g in components.values()), key=lambda g: g[0])
        cluster["twin_groups"] = groups
        cluster["twin_evidence"] = [
            sorted({rel for pair, rel in evidence.items() if pair <= set(g)}) for g in groups]
        cluster["review"] = {"same_file_different_labels": relabelled,
                             "same_description_different_file_names": renamed}
        cluster["metadata_split"] = len(groups) > 1
        cluster["metadata_known"] = all(signature[m] for m in members)
        parsed = [json.loads(signature[g[0]]) if signature[g[0]] else {} for g in groups]
        differing: set[str] = set()
        for section in {k for d in parsed for k in d}:
            for field in {f for d in parsed for f in d.get(section, {})}:
                if len({json.dumps(d.get(section, {}).get(field), sort_keys=True) for d in parsed}) > 1:
                    differing.add(f"{section}.{field}")
        cluster["split_on"] = sorted(differing) if len(groups) > 1 else []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet-dir", type=Path, default=ROOT_DIR / "data" / "uk" / "clean_datasets" / "parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-content-mb", type=float, default=300.0,
                        help="skip the content digest for files larger than this")
    parser.add_argument("--min-cells", type=int, default=10, help="clusters with rows*cols at or below this are flagged trivial")
    parser.add_argument("--refine", type=Path, help="reuse the clusters of an earlier scan instead of scanning")
    parser.add_argument("--no-metadata", action="store_true", help="skip the split by descriptive metadata")
    parser.add_argument("--core", default="uk")
    parser.add_argument("--solr-url", default=os.environ.get("SOLR_BASE_URL", "http://localhost:8983/solr"))
    parser.add_argument("--ckan-metadata", type=Path, help="metadata_retrieved_only.json (default: data/<core>/metadata)")
    args = parser.parse_args(argv)
    if args.refine:
        result = json.loads(args.refine.read_text(encoding="utf-8"))
    else:
        result = scan(args.parquet_dir, workers=args.workers, max_content_mb=args.max_content_mb, min_cells=args.min_cells)
    if not args.no_metadata:
        ckan_path = args.ckan_metadata or ROOT_DIR / "data" / args.core / "metadata" / "metadata_retrieved_only.json"
        refine_by_metadata(result["clusters"], solr=(args.solr_url, args.core), ckan=load_ckan_index(ckan_path))
        nontrivial = [c for c in result["clusters"] if not c["trivial"]]
        result["summary"].update({
            "clusters_metadata_split": sum(c["metadata_split"] for c in nontrivial),
            "clusters_all_members_one_twin_group": sum(not c["metadata_split"] and c["metadata_known"] for c in nontrivial),
            "clusters_without_metadata": sum(not c["metadata_known"] for c in nontrivial),
        })
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
