"""Rebuild the ``uk_fixed`` Solr core from the cleaned UK catalogue.

    # Dry run (default): build every document, diff against the core, write a
    # report. Nothing is written to Solr.
    .venv/bin/python -m solr_builder.build_uk_fixed

    # Write the rebuilt documents (backs up the core first, then verifies).
    .venv/bin/python -m solr_builder.build_uk_fixed --apply

    # Also embed with OCI cohere.embed-v4.0 (resumable; spends OCI credits).
    .venv/bin/python -m solr_builder.build_uk_fixed --apply --embed

    # Re-run only the checks, or put the pre-build documents back.
    .venv/bin/python -m solr_builder.build_uk_fixed --verify
    .venv/bin/python -m solr_builder.build_uk_fixed --restore backups/uk_fixed_build/before.jsonl.gz

Documents are rebuilt from ``metadata_retrieved_cleaned.json`` and the parquet
files (``solr_builder.portal``), not copied from another core. The
script writes only to a core named on the command line, never creates one, and
refuses the production cores.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import gzip
import hashlib
import json
import random
import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq
import requests

from lakegen.config.paths import resolve_portal_metadata_path, resolve_portal_tables_dir
from lakegen.config.settings import get_settings
from lakegen.retrieval.solr.client import SolrClient
from lakegen.utils.naming import is_uninformative
from solr_builder.indexing import validate_vector_schema
from solr_builder.portal import (
    build_uk_document,
    has_markup,
    needs_engine_names,
    resolve_column_names,
    solr_timestamp,
)
from solr_builder.representation import METADATA_V1, represent_table

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_CORES = {"uk", "nyc", "nyc_old", "nyc_new", "bologna", "paris", "valencia"}
VECTOR_FIELD = "table_embedding"
EMBEDDING_MODEL = "cohere.embed-v4.0"
EMBEDDING_DIMENSION = 1024
# The fields build_uk_document writes, compared against what the core holds.
COMPARE_FIELDS = (
    "dataset_id",
    "source",
    "title",
    "description",
    "publisher",
    "tags",
    "created_at",
    "modified_at",
    "metadata_created",
    "metadata_modified",
    "dataset_url",
    "download_url",
    "format",
    "columns.name",
    "columns.label",
    "columns.type",
)
DATE_FIELDS = {"created_at", "modified_at", "metadata_created", "metadata_modified"}
PROVENANCE_FIELDS = ("representation_version", "embedding_model")
CSV_TOKEN = re.compile(r"(?<![0-9A-Za-z])csv(?![0-9A-Za-z])", re.IGNORECASE)
FILE_EXTENSION = re.compile(r"\.(?:csv|tsv|xlsx?|json|zip|txt|xml)\s*$", re.IGNORECASE)
ENTITY = re.compile(r"&(?:[A-Za-z]+|#\d+);")


def say(message: str = "") -> None:
    """Print a progress line at once."""
    print(message, flush=True)


# --- reading the source ------------------------------------------------------


def load_catalogue(path: Path) -> dict[tuple[str, str], tuple[dict, dict]]:
    """Index the cleaned catalogue by package and resource ID.

    Args:
        path: The cleaned metadata file, a JSON list of CKAN packages.

    Returns:
        The package and the resource of every ``(package id, resource id)`` pair.
    """
    index = {}
    for package in json.loads(path.read_text(encoding="utf-8")):
        package_id = str(package.get("id") or "").strip()
        for resource in package.get("resources", []):
            resource_id = str(resource.get("id") or "").strip()
            if package_id and resource_id:
                index[(package_id, resource_id)] = (package, resource)
    return index


def duckdb_names(path: Path) -> list[str]:
    """Return a parquet file's column names as DuckDB exposes them."""
    connection = duckdb.connect()
    try:
        cursor = connection.execute("SELECT * FROM read_parquet(?) LIMIT 0", [str(path)])
        return [column[0] for column in cursor.description]
    finally:
        connection.close()


def read_columns(path: Path) -> tuple[list[str], list[str]]:
    """Return the column names and types of a parquet file, names as the SQL engine exposes them."""
    schema = pq.ParquetFile(path).schema_arrow
    names = [field.name for field in schema]
    types = [str(field.type) for field in schema]
    engine = duckdb_names(path) if needs_engine_names(names) else None
    return resolve_column_names(names, engine), types


def parquet_widths(files: Sequence[Path], workers: int) -> dict[str, int]:
    """Count the columns of each parquet file.

    Args:
        files: The parquet files.
        workers: Files read in parallel.

    Returns:
        The column count by file stem.
    """
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        return dict(
            zip(
                (path.stem for path in files),
                pool.map(lambda path: len(pq.ParquetFile(path).schema_arrow), files),
                strict=True,
            )
        )


def build_documents(
    files: Sequence[Path],
    catalogue: dict[tuple[str, str], tuple[dict, dict]],
    generation: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Rebuild the document of every parquet file from the catalogue and the file's schema.

    Args:
        files: The UK parquet files, named ``<package id>___<resource id>``.
        catalogue: The index of :func:`load_catalogue`.
        generation: This run's ``indexed_ts``.
        workers: Files read in parallel.

    Returns:
        The documents, in the order of ``files``.

    Raises:
        RuntimeError: When a file name does not split into the two IDs, the
            catalogue has no entry for it, or two documents share a resource ID.
    """

    def build(path: Path) -> dict[str, Any]:
        try:
            package_id, resource_id = path.stem.split("___", 1)
        except ValueError as exc:
            raise RuntimeError(f"Unexpected UK filename: {path.name}") from exc
        try:
            package, resource = catalogue[(package_id, resource_id)]
        except KeyError as exc:
            raise RuntimeError(f"No cleaned metadata for {path.name}") from exc
        names, types = read_columns(path)
        return build_uk_document(
            package,
            resource,
            column_names=names,
            column_types=types,
            generation=generation,
        )

    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        documents = list(pool.map(build, files))
    ids = [document["resource_id"] for document in documents]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate resource_id in the rebuilt documents")
    return documents


# --- reading and writing the core --------------------------------------------


def as_list(value: Any) -> list[Any]:
    """Return a Solr field value as a list: a list as it is, None empty, a scalar wrapped."""
    return value if isinstance(value, list) else ([] if value is None else [value])


def read_core(client: SolrClient) -> dict[str, dict[str, Any]]:
    """Return the core's documents by resource ID, with the compared and provenance fields."""
    fields = ("resource_id", *COMPARE_FIELDS, *PROVENANCE_FIELDS)
    return {
        doc["resource_id"]: doc
        for doc in client.iter_documents(fields=fields, batch_size=1000, restore_columns=False)
    }


def comparable(field: str, value: Any) -> list[Any]:
    """Return a field's values as two builds can be compared.

    Dates are normalized: Solr prints 12:41:01Z where the builder writes
    12:41:01.000Z, the same instant.

    Args:
        field: The field's name.
        value: Its value, as stored or as built.

    Returns:
        The values as a list.
    """
    values = as_list(value)
    return [solr_timestamp(item) for item in values] if field in DATE_FIELDS else values


def description_change_kind(old: str, new: str) -> str:
    """Name what kind of change turned the old description into the new one."""
    if has_markup(old) or ENTITY.search(old):
        return "markup or entities removed"
    if " ".join(old.split()) == new:
        return "whitespace only"
    return "whitespace, rejoined or de-duplicated"


def diff_report(
    current: dict[str, dict[str, Any]], documents: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Compare the rebuilt documents with the core's, field by field.

    Args:
        current: The core's documents by resource ID (:func:`read_core`).
        documents: The rebuilt documents.

    Returns:
        Counts of new and stale IDs, of changed fields and of kinds of
        description change, a few samples per changed field, and how many
        stored documents carry a representation label but no vector.
    """
    new = {doc["resource_id"]: doc for doc in documents}
    changed: Counter[str] = Counter()
    description_kinds: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    for resource_id, doc in new.items():
        old = current.get(resource_id)
        if old is None:
            continue
        for field in COMPARE_FIELDS:
            if comparable(field, old.get(field)) != comparable(field, doc.get(field)):
                changed[field] += 1
                if field == "description":
                    description_kinds[
                        description_change_kind(
                            str(old.get(field) or ""), str(doc.get(field) or "")
                        )
                    ] += 1
                bucket = samples.setdefault(field, [])
                if len(bucket) < 3:
                    bucket.append(
                        {
                            "resource_id": resource_id,
                            "before": str(old.get(field))[:220],
                            "after": str(doc.get(field))[:220],
                        }
                    )
    return {
        "documents_built": len(new),
        "documents_in_core": len(current),
        "new_ids": len(new.keys() - current.keys()),
        "stale_ids": sorted(current.keys() - new.keys()),
        "fields_changed": dict(changed),
        "description_changes": dict(description_kinds),
        "samples": samples,
        "without_vector_but_labelled": sum(
            1
            for old in current.values()
            if old.get("representation_version") and not old.get("embedding_model")
        ),
    }


def assert_replacement_loses_nothing(client: SolrClient, schema: dict[str, Any]) -> None:
    """Refuse to replace documents when that would drop data that cannot be read back.

    Writing a full document replaces the old one. That is only safe if no field
    that cannot be read back (indexed but not stored, and not rebuilt by a
    copyField) is populated in the core.

    Args:
        client: The core.
        schema: The core's schema.

    Raises:
        RuntimeError: When such a field holds values.
    """
    rebuilt = {str(item.get("dest")) for item in schema.get("copyFields", [])}
    rebuilt |= {"_root_", VECTOR_FIELD}
    lost = {}
    for field in schema.get("fields", []):
        name = str(field.get("name") or "")
        if name in rebuilt or field.get("indexed") is not True or field.get("stored") is True:
            continue
        populated = count(client, f"{name}:[* TO *]")
        if populated:
            lost[name] = populated
    if lost:
        raise RuntimeError(f"Replacing documents would drop data that cannot be re-created: {lost}")


def write_documents(
    client: SolrClient, documents: Sequence[dict[str, Any]], batch_size: int
) -> None:
    """Post the documents in batches, then commit.

    Args:
        client: The core.
        documents: The documents, each replacing the stored one with its ID.
        batch_size: Documents per update request.
    """
    for start in range(0, len(documents), batch_size):
        client.update_documents(list(documents[start : start + batch_size]))
        say(f"  posted {min(start + batch_size, len(documents))}/{len(documents)}")
    client.commit()


def delete_ids(client: SolrClient, ids: Sequence[str]) -> None:
    """Delete documents by ID, 100 per request, then commit.

    Args:
        client: The core.
        ids: The IDs to delete.
    """
    for start in range(0, len(ids), 100):
        response = requests.post(
            f"{client.core_url}/update",
            params={"wt": "json"},
            json={"delete": [{"id": value} for value in ids[start : start + 100]]},
            timeout=client.timeout,
        )
        response.raise_for_status()
    if ids:
        client.commit()


def backup_core(client: SolrClient, path: Path) -> None:
    """Save every stored field of every document, unless a backup already exists.

    An existing backup is kept: the first one is the state before this tool
    ever touched the core.

    Args:
        client: The core.
        path: The gzipped JSON-lines backup file.
    """
    if path.exists():
        say(f"backup already exists, keeping it: {path}")
        return
    temporary = path.with_suffix(".tmp")
    written = 0
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        for doc in client.iter_documents(fields=("*",), batch_size=1000, restore_columns=False):
            stream.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1
    temporary.replace(path)
    say(f"backed up {written} documents to {path}")


def restore_core(client: SolrClient, path: Path, batch_size: int) -> None:
    """Put a backup back: write its documents and delete every other one.

    Args:
        client: The core.
        path: A backup written by :func:`backup_core`.
        batch_size: Documents per update request.
    """
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        documents = [
            {key: value for key, value in json.loads(line).items() if key != "_version_"}
            for line in stream
        ]
    write_documents(client, documents, batch_size)
    keep = {doc["resource_id"] for doc in documents}
    stale = sorted(set(read_core(client)) - keep)
    delete_ids(client, stale)
    say(f"restored {len(documents)} documents, removed {len(stale)} others")


# --- embeddings ---------------------------------------------------------------


def _staged_vectors(staged: Path, digests: dict[str, str]) -> dict[str, np.ndarray]:
    """Read back the staged vectors whose text is unchanged.

    Args:
        staged: The staging file, one JSON record per vector.
        digests: The SHA-1 of each document's current embedding text.

    Returns:
        The reusable vectors by resource ID.
    """
    vectors: dict[str, np.ndarray] = {}
    if staged.exists():
        with staged.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if digests.get(record["id"]) == record["sha"]:
                    vectors[record["id"]] = np.frombuffer(
                        base64.b64decode(record["v"]), dtype="<f4"
                    )
    return vectors


def embed_documents(
    documents: Sequence[dict[str, Any]], staged: Path, workers: int
) -> dict[str, np.ndarray]:
    """Embed every document on OCI, staged to disk so an interrupted run resumes.

    A staged vector is reused only if the text it was made from is unchanged.

    Args:
        documents: The rebuilt documents.
        staged: The staging file, appended to as batches complete.
        workers: Concurrent embedding requests.

    Returns:
        A vector for every document, by resource ID.

    Raises:
        RuntimeError: When a document has no embedding text, or a batch failed;
            what completed stays staged for the next run.
    """
    from lakegen.llm.embeddings import OCIEmbedding, _validate

    texts = {doc["resource_id"]: represent_table(doc, METADATA_V1) for doc in documents}
    if not all(text.strip() for text in texts.values()):
        raise RuntimeError("A document has an empty embedding text")
    digests = {
        key: hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        for key, text in texts.items()
    }
    vectors = _staged_vectors(staged, digests)
    say(f"embedding: {len(vectors)} staged, {len(documents) - len(vectors)} to do")

    pending = [doc["resource_id"] for doc in documents if doc["resource_id"] not in vectors]
    batches = [pending[i : i + 16] for i in range(0, len(pending), 16)]
    local = threading.local()

    def embed(batch: list[str]) -> list[tuple[str, np.ndarray]]:
        if not hasattr(local, "model"):
            local.model = OCIEmbedding(EMBEDDING_MODEL)
        result = local.model.encode_documents([texts[key] for key in batch])
        return [
            (
                key,
                np.asarray(_validate(vector, expected_dimension=EMBEDDING_DIMENSION), dtype="<f4"),
            )
            for key, vector in zip(batch, result, strict=True)
        ]

    failure: Exception | None = None
    with (
        staged.open("a", encoding="utf-8") as out,
        concurrent.futures.ThreadPoolExecutor(workers) as pool,
    ):
        for start in range(0, len(batches), workers * 8):
            futures = [pool.submit(embed, batch) for batch in batches[start : start + workers * 8]]
            for future in futures:
                try:
                    produced = future.result()
                except Exception as exc:  # keep what completed, resume on the next run
                    failure = failure or exc
                    continue
                for key, vector in produced:
                    record = {
                        "id": key,
                        "sha": digests[key],
                        "v": base64.b64encode(vector.tobytes()).decode("ascii"),
                    }
                    out.write(json.dumps(record) + "\n")
                    vectors[key] = vector
            out.flush()
            say(f"  embedded {len(vectors)}/{len(documents)}")
            if failure:
                break
    if failure:
        raise RuntimeError(f"Embedding stopped; re-run to resume: {failure}") from failure
    return vectors


def attach_vectors(
    documents: Sequence[dict[str, Any]], vectors: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Add each document's vector and its provenance fields.

    Args:
        documents: The rebuilt documents.
        vectors: A vector for every document, by resource ID.

    Returns:
        The documents with the vector field, the embedding model and the representation version.
    """
    result = []
    for doc in documents:
        vector = vectors[doc["resource_id"]]
        result.append(
            {
                **doc,
                VECTOR_FIELD: [float(f"{value:.9g}") for value in vector],
                "embedding_model": EMBEDDING_MODEL,
                "representation_version": METADATA_V1,
            }
        )
    return result


# --- verification -------------------------------------------------------------

Check = Callable[..., None]


def count(client: SolrClient, query: str) -> int:
    """Count the documents of the core that match a Solr query."""
    response = requests.get(
        client.select_url,
        params={"q": query, "rows": 0, "wt": "json"},
        timeout=client.timeout,
    )
    response.raise_for_status()
    return int(response.json()["response"]["numFound"])


def search(client: SolrClient, words: str, rows: int = 20) -> list[dict[str, Any]]:
    """Return the IDs of the documents that match every word, best first."""
    response = requests.get(
        client.select_url,
        params={
            "q": words,
            "q.op": "AND",
            "rows": rows,
            "fl": "resource_id,dataset_id",
            "wt": "json",
        },
        timeout=client.timeout,
    )
    response.raise_for_status()
    return response.json()["response"]["docs"]


def _verify_documents(docs: list[dict[str, Any]], widths: dict[str, int], check: Check) -> None:
    """Check titles, descriptions and column lists against the parquet files.

    Args:
        docs: The core's documents.
        widths: Column count by parquet file stem.
        check: Records one named check and whether it passed.
    """
    stems = {f"{d['dataset_id']}___{d['resource_id']}" for d in docs}
    check(
        "document count matches parquet files",
        len(docs) == len(widths),
        f"{len(docs)} docs / {len(widths)} files",
    )
    check(
        "every document has a parquet file and vice versa",
        stems == set(widths),
        f"{len(stems - set(widths))} without file, {len(set(widths) - stems)} files without doc",
    )
    titles = [str(d.get("title") or "") for d in docs]
    check("no empty title", all(t.strip() for t in titles))
    check(
        "no title with a csv token",
        not any(CSV_TOKEN.search(t) for t in titles),
        str(sum(1 for t in titles if CSV_TOKEN.search(t))),
    )
    check("no title ending in a file extension", not any(FILE_EXTENSION.search(t) for t in titles))
    check("no title that says nothing but its format", not any(is_uninformative(t) for t in titles))
    descriptions = [str(d.get("description") or "") for d in docs]
    check(
        "no HTML markup in descriptions",
        not any(has_markup(t) for t in descriptions),
        str(sum(1 for t in descriptions if has_markup(t))),
    )
    entities = sum(1 for t in descriptions if ENTITY.search(t))
    say(
        f"  info: {entities} descriptions contain an entity-like token; "
        f"{sum(1 for t in descriptions if not t.strip())} are empty"
    )
    misaligned = blank = wrong_width = 0
    for d in docs:
        names, labels, types = (
            as_list(d.get(k)) for k in ("columns.name", "columns.label", "columns.type")
        )
        if not (len(names) == len(labels) == len(types)):
            misaligned += 1
        if any(not str(n).strip() for n in names):
            blank += 1
        if len(names) != widths.get(f"{d['dataset_id']}___{d['resource_id']}", -1):
            wrong_width += 1
    check("columns.name/label/type lists are aligned", misaligned == 0, f"{misaligned} docs")
    check("no blank column name", blank == 0, f"{blank} docs")
    check("column count equals the parquet file's", wrong_width == 0, f"{wrong_width} docs")


def _verify_provenance(
    client: SolrClient,
    total: int,
    check: Check,
    *,
    expect_vectors: bool,
    generation: str | None,
) -> None:
    """Check the vector provenance labels and, after a write, this run's timestamp.

    Args:
        client: The core.
        total: How many documents the core holds.
        check: Records one named check and whether it passed.
        expect_vectors: Whether every document should carry an OCI vector.
        generation: This run's ``indexed_ts``, after a write.
    """
    labelled = count(client, "representation_version:*")
    with_model = count(client, "embedding_model:*")
    if expect_vectors:
        good = count(
            client,
            f'embedding_model:"{EMBEDDING_MODEL}" AND representation_version:"{METADATA_V1}"',
        )
        check("every document has an OCI vector and provenance", good == total, f"{good}/{total}")
    else:
        check(
            "no provenance label without a vector",
            labelled == 0 and with_model == 0,
            f"{labelled} labelled, {with_model} with a model",
        )
    if generation:
        check(
            "every document carries this run's indexed_ts",
            count(client, f'indexed_ts:"{generation}"') == total,
        )


def _verify_keyword_search(
    client: SolrClient, docs: list[dict[str, Any]], rng: random.Random, check: Check
) -> None:
    """Check that keyword search finds a sample of documents' datasets from their titles.

    Args:
        client: The core.
        docs: The core's documents.
        rng: The seeded sampler shared by the checks.
        check: Records one named check and whether it passed.
    """
    sample = rng.sample(docs, min(30, len(docs)))
    dataset_hits = resource_hits = 0
    for d in sample:
        words = " ".join(re.findall(r"[A-Za-z0-9]+", str(d.get("title") or "")))
        found = search(client, words)
        dataset_hits += any(hit["dataset_id"] == d["dataset_id"] for hit in found)
        resource_hits += any(hit["resource_id"] == d["resource_id"] for hit in found)
    check(
        "keyword search finds a document's dataset from its own title (top 20)",
        dataset_hits >= 0.9 * len(sample),
        f"{dataset_hits}/{len(sample)}",
    )
    say(
        f"  info: exact resource in top 20 for {resource_hits}/{len(sample)} "
        "(siblings can outrank it)"
    )


def _verify_vectors(
    client: SolrClient, docs: list[dict[str, Any]], rng: random.Random, check: Check
) -> None:
    """Check the stored vectors through LakeGen's own KNN path, and semantic search by title.

    Twins (same title, description and columns) share one vector, so "the
    document itself is rank 1" cannot hold; and HNSW is approximate, so a tiny
    topK occasionally misses. What must hold: a stored vector is retrievable
    as an identical vector (cosine ~ 1). The KNN path is the /query handler,
    not the edismax /select.

    Args:
        client: The core.
        docs: The core's documents.
        rng: The seeded sampler shared by the checks.
        check: Records one named check and whether it passed.
    """
    knn_ok = self_top3 = 0
    picks = rng.sample(docs, min(20, len(docs)))
    for d in picks:
        response = requests.get(
            client.select_url,
            params={
                "q": f'resource_id:"{d["resource_id"]}"',
                "fl": VECTOR_FIELD,
                "rows": 1,
                "wt": "json",
            },
            timeout=client.timeout,
        )
        vector = response.json()["response"]["docs"][0][VECTOR_FIELD]
        top = client.knn_select(vector, vector_field=VECTOR_FIELD, top_k=10)["response"]["docs"]
        knn_ok += bool(top) and float(top[0].get("score", 0)) >= 0.999
        self_top3 += d["resource_id"] in [hit["resource_id"] for hit in top[:3]]
    check(
        "a stored vector retrieves an identical vector (cosine >= 0.999)",
        knn_ok >= 0.9 * len(picks),
        f"{knn_ok}/{len(picks)}",
    )
    say(
        f"  info: the document itself is in the top 3 for {self_top3}/{len(picks)} "
        "(ties between twins)"
    )

    from lakegen.llm.embeddings import OCIEmbedding

    model = OCIEmbedding(EMBEDDING_MODEL)
    titled = rng.sample(docs, min(10, len(docs)))
    semantic_hits = 0
    for d in titled:
        query_vector = model.encode_query(str(d.get("title") or ""))
        response = client.knn_select(query_vector, vector_field=VECTOR_FIELD, top_k=20)
        found = response["response"]["docs"]
        semantic_hits += any(hit.get("dataset_id") == d["dataset_id"] for hit in found)
    check(
        "semantic search finds a document's dataset from its own title (top 20)",
        semantic_hits >= 0.8 * len(titled),
        f"{semantic_hits}/{len(titled)}",
    )


def verify(
    client: SolrClient,
    widths: dict[str, int],
    *,
    expect_vectors: bool,
    generation: str | None = None,
    seed: int = 7,
) -> bool:
    """Read the core back and check every property this tool exists to guarantee.

    Args:
        client: The core.
        widths: Column count by parquet file stem.
        expect_vectors: Whether every document should carry an OCI vector.
        generation: This run's ``indexed_ts``, after a write.
        seed: Seed of the sampled spot checks.

    Returns:
        Whether every check passed; each check is printed with its outcome.
    """
    fields = (
        "resource_id",
        "dataset_id",
        "title",
        "description",
        "columns.name",
        "columns.label",
        "columns.type",
    )
    docs = list(client.iter_documents(fields=fields, batch_size=1000, restore_columns=False))
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    _verify_documents(docs, widths, check)
    _verify_provenance(
        client, len(docs), check, expect_vectors=expect_vectors, generation=generation
    )
    # Seeded sampling of documents to spot-check, not a security use.
    rng = random.Random(seed)  # nosec B311
    _verify_keyword_search(client, docs, rng, check)
    if expect_vectors:
        _verify_vectors(client, docs, rng, check)

    width = max(len(name) for name, _, _ in results)
    say()
    for name, ok, detail in results:
        say(f"  {'PASS' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    passed = all(ok for _, ok, _ in results)
    say(
        f"\n{'ALL CHECKS PASSED' if passed else 'CHECKS FAILED'} "
        f"({sum(ok for _, ok, _ in results)}/{len(results)})"
    )
    return passed


# --- entry point ---------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--core", default="uk_fixed")
    parser.add_argument("--solr-base-url", default=get_settings().solr_base_url)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--parquet-dir", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "backups" / "uk_fixed_build")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8, help="parallel parquet readers")
    parser.add_argument("--embed-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="dry run on the first N files only")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--apply", action="store_true", help="write to the core")
    modes.add_argument("--verify", action="store_true", help="only check the core")
    modes.add_argument("--restore", type=Path, metavar="BACKUP", help="put a backup back")
    parser.add_argument("--embed", action="store_true", help="with --apply: add OCI vectors")
    return parser


def _report_diff(
    client: SolrClient, documents: list[dict[str, Any]], work_dir: Path, core: str
) -> dict[str, Any]:
    """Diff the rebuilt documents against the core, save the report and print its summary.

    Args:
        client: The core.
        documents: The rebuilt documents.
        work_dir: Where ``report.json`` is written.
        core: The core's name, for the summary.

    Returns:
        The report of :func:`diff_report`.
    """
    current = read_core(client)
    report = diff_report(current, documents)
    (work_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    say(f"\nagainst the {len(current)} documents now in {core}:")
    say(f"  new ids: {report['new_ids']}   stale ids: {len(report['stale_ids'])}")
    say(
        "  docs labelled with a representation but holding no vector: "
        f"{report['without_vector_but_labelled']}"
    )
    for field, n in sorted(report["fields_changed"].items(), key=lambda item: -item[1]):
        say(f"  {field}: {n} documents change")
        if field == "description":
            for kind, kind_count in sorted(
                report["description_changes"].items(), key=lambda item: -item[1]
            ):
                say(f"      {kind}: {kind_count}")
    if not report["fields_changed"]:
        say("  no field changes")
    say(f"report: {work_dir / 'report.json'}")
    return report


def _apply(
    client: SolrClient,
    schema: dict[str, Any],
    documents: list[dict[str, Any]],
    report: dict[str, Any],
    args: argparse.Namespace,
    generation: str,
) -> int:
    """Write the rebuilt documents: backup, vectors if asked, write, stale deletion, checks.

    Stale documents are deleted only once every rebuilt document landed.

    Args:
        client: The core.
        schema: The core's schema.
        documents: The rebuilt documents.
        report: The diff of :func:`diff_report`, which names the stale IDs.
        args: Parsed options: ``--embed``, the batch sizes and the work folder.
        generation: This run's ``indexed_ts``.

    Returns:
        The exit code: 0 when every check passed.
    """
    assert_replacement_loses_nothing(client, schema)
    if args.embed:
        validate_vector_schema(schema, vector_field=VECTOR_FIELD, dimension=EMBEDDING_DIMENSION)
    backup_core(client, args.work_dir / "before.jsonl.gz")

    batch_size = args.batch_size
    if args.embed:
        vectors = embed_documents(documents, args.work_dir / "vectors.jsonl", args.embed_workers)
        documents = attach_vectors(documents, vectors)
        batch_size = min(batch_size, 200)
    say(f"\nwriting {len(documents)} documents to {args.core}")
    write_documents(client, documents, batch_size)

    landed = count(client, f'indexed_ts:"{generation}"')
    if landed != len(documents):
        say(
            f"only {landed}/{len(documents)} documents carry this run's indexed_ts; "
            "stale ids NOT deleted"
        )
        return 1
    if report["stale_ids"]:
        delete_ids(client, report["stale_ids"])
        say(f"deleted {len(report['stale_ids'])} stale documents")

    widths = {f"{d['dataset_id']}___{d['resource_id']}": len(d["columns.name"]) for d in documents}
    ok = verify(client, widths, expect_vectors=args.embed, generation=generation)
    return 0 if ok else 1


def main() -> int:
    """Run a dry run, a write, a verification or a restore of the core.

    Returns:
        The exit code: 0 on success.
    """
    parser = _parser()
    args = parser.parse_args()
    if args.embed and not args.apply:
        parser.error("--embed needs --apply")
    if args.limit and args.apply:
        parser.error("--limit is for dry runs only")
    if args.core in PRODUCTION_CORES:
        parser.error(f"refusing to touch production core {args.core!r}")

    client = SolrClient(args.core, base_url=args.solr_base_url, timeout=300)
    try:
        schema = client.schema()["schema"]
    except requests.HTTPError as exc:
        parser.error(f"core {args.core!r} does not exist or is unreachable: {exc}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.restore:
        restore_core(client, args.restore, args.batch_size)
        return 0

    tables = args.parquet_dir or resolve_portal_tables_dir("uk")
    files = sorted(Path(tables).glob("*.parquet"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        parser.error(f"no parquet files in {tables}")

    if args.verify:
        widths = parquet_widths(files, args.workers)
        expect_vectors = count(client, "embedding_model:*") > 0
        return 0 if verify(client, widths, expect_vectors=expect_vectors) else 1

    metadata = args.metadata or resolve_portal_metadata_path("uk")
    generation = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    say(f"target core: {args.core} ({client.core_url})")
    say(f"metadata: {metadata}\nparquet:  {tables} ({len(files)} files)")
    documents = build_documents(files, load_catalogue(metadata), generation, args.workers)
    say(f"built {len(documents)} documents")

    report = _report_diff(client, documents, args.work_dir, args.core)
    if not args.apply:
        say("\nDry run: nothing written. Re-run with --apply to write.")
        return 0
    return _apply(client, schema, documents, report, args, generation)


if __name__ == "__main__":
    raise SystemExit(main())
