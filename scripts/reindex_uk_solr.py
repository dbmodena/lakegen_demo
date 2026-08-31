"""Rebuild the generic UK Solr core from the cleaned canonical dataset.

The operation is deliberately recoverable: it exports every current document,
upserts all retained resources with fresh embeddings, verifies the generation,
and only then deletes stale resource IDs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
from typing import Any, Iterable

import pyarrow.parquet as pq
import requests


def utc_solr_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def normalized_datetime(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def named_values(values: Any) -> list[str]:
    result: list[str] = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, dict):
            value = value.get("display_name") or value.get("title") or value.get("name")
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def representations(documents: Iterable[dict[str, Any]]) -> list[str]:
    rendered = []
    for document in documents:
        sections = []
        for label, values in (
            ("Title", [document.get("title")]),
            ("Description", [document.get("description")]),
            ("Tags", document.get("tags", [])),
            ("Column names", document.get("columns.name", [])),
            ("Column descriptions", document.get("columns.description", [])),
        ):
            clean = [str(value).strip() for value in values if str(value or "").strip()]
            if clean:
                sections.append(f"{label}: {' | '.join(clean)}")
        rendered.append(
            "\n".join(
                ("Represent this table metadata for information retrieval.", *sections)
            )
        )
    return rendered


def request_json(
    method: str,
    url: str,
    *,
    timeout: float = 120,
    attempts: int = 4,
    **kwargs: Any,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # network/provider errors are retryable here
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def solr_documents(core_url: str, batch_size: int = 100) -> Iterable[dict[str, Any]]:
    cursor = "*"
    while True:
        payload = request_json(
            "GET",
            f"{core_url}/select",
            params={
                "q": "*:*",
                "fl": "*",
                "rows": batch_size,
                "sort": "resource_id asc",
                "cursorMark": cursor,
                "wt": "json",
            },
        )
        documents = payload.get("response", {}).get("docs", [])
        yield from documents
        next_cursor = payload.get("nextCursorMark")
        if not documents or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor


def build_documents(metadata_path: Path, parquet_dir: Path, generation: str):
    packages = json.loads(metadata_path.read_text(encoding="utf-8"))
    resources: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for package in packages:
        package_id = str(package.get("id") or "").strip()
        for resource in package.get("resources", []):
            resource_id = str(resource.get("id") or "").strip()
            if package_id and resource_id:
                resources[(package_id, resource_id)] = (package, resource)

    documents = []
    for parquet_path in sorted(parquet_dir.glob("*.parquet")):
        try:
            package_id, resource_id = parquet_path.stem.split("___", 1)
        except ValueError as exc:
            raise RuntimeError(f"Unexpected UK filename: {parquet_path.name}") from exc
        try:
            package, resource = resources[(package_id, resource_id)]
        except KeyError as exc:
            raise RuntimeError(
                f"No cleaned metadata for {parquet_path.name}"
            ) from exc
        schema = pq.ParquetFile(parquet_path).schema_arrow
        column_names = [field.name for field in schema]
        organization = package.get("organization") or {}
        document = {
            "dataset_id": package_id,
            "resource_id": resource_id,
            "source": "UK Open Data",
            "title": str(
                resource.get("name") or package.get("title") or package.get("name") or ""
            ).strip(),
            "description": " ".join(
                value
                for value in (
                    str(package.get("notes") or "").strip(),
                    str(resource.get("description") or "").strip(),
                )
                if value
            ),
            "publisher": str(
                organization.get("title") or organization.get("name") or ""
            ).strip(),
            "tags": named_values(package.get("tags")),
            "created_at": normalized_datetime(resource.get("created")),
            "modified_at": normalized_datetime(resource.get("metadata_modified")),
            "metadata_created": normalized_datetime(package.get("metadata_created")),
            "metadata_modified": normalized_datetime(package.get("metadata_modified")),
            "dataset_url": f"https://www.data.gov.uk/dataset/{package_id}",
            "download_url": str(resource.get("url") or "").strip(),
            "format": "parquet",
            "columns.name": column_names,
            "columns.label": column_names,
            "columns.type": [str(field.type) for field in schema],
            "indexed_ts": generation,
            "representation_version": "metadata-v1",
            "embedding_model": "bge-m3",
        }
        documents.append(
            {key: value for key, value in document.items() if value not in (None, "", [])}
        )
    return documents


def save_checkpoint(path: Path, **values: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solr-url", default="http://127.0.0.1:8983/solr")
    parser.add_argument("--core", default="uk")
    parser.add_argument(
        "--metadata", default="data/uk/metadata/metadata_retrieved_cleaned.json"
    )
    parser.add_argument("--parquet-dir", default="data/uk/clean_datasets/parquet")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    core_url = f"{args.solr_url.rstrip('/')}/{args.core}"
    checkpoint_path = args.work_dir / "checkpoint.json"
    generation_path = args.work_dir / "generation.txt"
    if generation_path.exists():
        generation = generation_path.read_text(encoding="utf-8").strip()
    else:
        generation = utc_solr_timestamp()
        generation_path.write_text(generation, encoding="utf-8")

    print(f"generation={generation}", flush=True)
    documents = build_documents(Path(args.metadata), Path(args.parquet_dir), generation)
    source_ids = {str(document["resource_id"]) for document in documents}
    if len(documents) != 15205 or len(source_ids) != len(documents):
        raise RuntimeError(
            f"Expected 15205 unique cleaned documents, got {len(documents)} / "
            f"{len(source_ids)} unique IDs"
        )
    print(f"prepared={len(documents)}", flush=True)

    schema_path = args.work_dir / "schema.json"
    backup_path = args.work_dir / "documents.jsonl.gz"
    if not backup_path.exists():
        schema = request_json("GET", f"{core_url}/schema", params={"wt": "json"})
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        current_ids = set()
        backup_count = 0
        with gzip.open(backup_path, "wt", encoding="utf-8") as stream:
            for document in solr_documents(core_url):
                current_ids.add(str(document.get("resource_id")))
                stream.write(json.dumps(document, ensure_ascii=False) + "\n")
                backup_count += 1
                if backup_count % 500 == 0:
                    print(f"backup={backup_count}", flush=True)
        (args.work_dir / "previous_ids.json").write_text(
            json.dumps(sorted(current_ids)), encoding="utf-8"
        )
        print(f"backup_complete={backup_count}", flush=True)
    previous_ids = set(
        json.loads((args.work_dir / "previous_ids.json").read_text(encoding="utf-8"))
    )

    start = 0
    if checkpoint_path.exists():
        start = int(json.loads(checkpoint_path.read_text(encoding="utf-8")).get("indexed", 0))
    for offset in range(start, len(documents), args.batch_size):
        batch = documents[offset : offset + args.batch_size]
        embedding_payload = request_json(
            "POST",
            f"{args.ollama_url.rstrip('/')}/api/embed",
            json={"model": "bge-m3", "input": representations(batch)},
            timeout=300,
        )
        vectors = embedding_payload.get("embeddings", [])
        if len(vectors) != len(batch) or any(len(vector) != 1024 for vector in vectors):
            raise RuntimeError("Ollama returned an invalid embedding batch")
        for document, vector in zip(batch, vectors, strict=True):
            document["table_embedding"] = vector
        request_json(
            "POST",
            f"{core_url}/update",
            params={"wt": "json"},
            json=batch,
            timeout=300,
        )
        indexed = offset + len(batch)
        save_checkpoint(checkpoint_path, indexed=indexed, generation=generation)
        print(f"indexed={indexed}/{len(documents)}", flush=True)

    request_json("POST", f"{core_url}/update", json={"commit": {}}, timeout=300)
    verification = request_json(
        "GET",
        f"{core_url}/select",
        params={
            "q": f'indexed_ts:"{generation}"',
            "rows": 0,
            "wt": "json",
        },
    )
    generation_count = int(verification.get("response", {}).get("numFound", 0))
    if generation_count != len(documents):
        raise RuntimeError(
            f"Refusing stale deletion: generation has {generation_count} documents, "
            f"expected {len(documents)}"
        )
    print(f"verified_generation={generation_count}", flush=True)

    stale_ids = sorted(previous_ids - source_ids)
    for offset in range(0, len(stale_ids), 100):
        commands = [{"id": resource_id} for resource_id in stale_ids[offset : offset + 100]]
        request_json("POST", f"{core_url}/update", json={"delete": commands})
    request_json("POST", f"{core_url}/update", json={"commit": {}}, timeout=300)

    final = request_json(
        "GET", f"{core_url}/select", params={"q": "*:*", "rows": 0, "wt": "json"}
    )
    final_count = int(final.get("response", {}).get("numFound", 0))
    if final_count != len(documents):
        raise RuntimeError(f"Final Solr count is {final_count}, expected {len(documents)}")
    save_checkpoint(
        checkpoint_path,
        indexed=len(documents),
        generation=generation,
        stale_deleted=len(stale_ids),
        final_count=final_count,
        completed=True,
    )
    print(
        f"completed=true final_count={final_count} stale_deleted={len(stale_ids)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
