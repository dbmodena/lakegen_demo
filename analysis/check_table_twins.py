#!/usr/bin/env python3
"""Decide whether two lake tables are the same table, from content and metadata.

For each pair the checker compares

* content: file bytes, parquet schema, column names/order, per-column and
  per-row hashes, numeric sums, and the same on the raw (pre-cleaning) files;
* metadata: the Solr documents (title, description, publisher, columns, tags,
  URLs, dates) and, when available, the CKAN package/resource records.

and returns one content verdict, one metadata verdict and a conclusion. Twins are
backups or re-uploads and nothing else: the same content is not enough, because the same
numbers can describe two things (ticket sales of a football match and a rugby match at one
stadium; two weekly releases that happen to agree). A pair is SAME_TABLE only when

    * the content is identical, and
    * the description is identical: publisher, title, period, columns, package notes, and
    * nothing shows they are different files (a re-upload keeps its file name).

    SAME_TABLE                        re-upload or backup
    SAME_FILE_DIFFERENT_LABELS        one published file listed with different labels
                                      (not a re-upload; review)
    SAME_DATA_DIFFERENT_DESCRIPTION   same content, description differs: NOT the same table
    SAME_DATA_CKAN_ONLY_DIFFERENCES   indexed description equal, CKAN records differ (review)
    SAME_DATA_DIFFERENT_FILE_NAMES    description equal but files named differently (review)
    VARIANT_SAME_SHAPE                same rows/columns, some values differ
    DIFFERENT_TABLES                  anything else

Usage:
    python analysis/check_table_twins.py A B [--md report.md] [--json out.json]
    python analysis/check_table_twins.py --pairs pairs.json --md report.md

A and B are `<dataset_id>___<resource_id>[.parquet]`. `pairs.json` is a list of
`{"label": "...", "a": "...", "b": "..."}`.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT_DIR = Path(__file__).resolve().parents[1]

CONTENT_SAME = (
    "IDENTICAL_FILE", "IDENTICAL_CONTENT",
    "SAME_CONTENT_REORDERED", "SAME_CONTENT_RENAMED_COLUMNS",
)

SOLR_FIELDS = (
    "dataset_id", "resource_id", "title", "description", "publisher", "format",
    "tags", "columns.name", "columns.label", "columns.type", "dataset_url",
    "download_url", "source", "created_at", "modified_at", "metadata_created",
    "metadata_modified",
)
# Fields that say what the table is. A difference here lets a retriever tell the
# two tables apart. Everything else is bookkeeping about where/when it was published.
SOLR_DESCRIPTIVE = ("title", "description", "publisher", "format",
                    "columns.name", "columns.label", "columns.type")
SOLR_BOOKKEEPING = ("tags", "dataset_id", "dataset_url", "download_url",
                    "created_at", "modified_at", "metadata_created", "metadata_modified")
CKAN_PACKAGE_DESCRIPTIVE = ("title", "notes", "organization", "license_id")
CKAN_PACKAGE_BOOKKEEPING = ("name", "metadata_created", "metadata_modified", "tags")
CKAN_RESOURCE_DESCRIPTIVE = ("name", "description", "format", "datafile-date")
CKAN_RESOURCE_BOOKKEEPING = ("url", "hash", "size", "created", "last_modified", "metadata_modified")
# Package extras that say what the data covers. They are harvest-derived and differ between
# copies of one record (bounding boxes at another precision, fields on one copy only), so
# they are reported but never decide whether two tables are the same.
CKAN_EXTRA_DESCRIPTIVE = (
    "spatial_text", "spatial", "bbox-east-long", "bbox-north-lat", "bbox-south-lat", "bbox-west-long",
    "theme", "topic_category", "lineage", "frequency", "frequency-of-update", "language",
    "dataset-reference-date",
)

_NULL_FLOAT = -9.87654321e-300  # sentinel so every NaN hashes the same way


# --------------------------------------------------------------------------- ids


def split_table_id(table_id: str) -> tuple[str, str]:
    stem = Path(table_id).name.removesuffix(".parquet")
    if "___" not in stem:
        raise ValueError(f"Expected <dataset_id>___<resource_id>, got {table_id!r}")
    dataset_id, resource_id = stem.split("___", 1)
    return dataset_id, resource_id


def table_file(directory: Path, table_id: str) -> Path:
    dataset_id, resource_id = split_table_id(table_id)
    return directory / f"{dataset_id}___{resource_id}.parquet"


# ----------------------------------------------------------------------- content


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(series: pd.Series) -> pd.Series:
    """Compare-friendly view: numerics as float64 (1 == 1.0), text stripped."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").astype("float64")
        return values.mask(values.isna(), _NULL_FLOAT)
    return series.astype("string").str.strip()


def _hashes(series: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(series, index=False).to_numpy()


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _name_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).casefold()).strip("_")


def _match_columns(a: list[pd.Series], b: list[pd.Series]) -> list[tuple[int, int, str]]:
    """Pair every column of A with one of B: same name, then same content."""
    pairs: list[tuple[int, int, str]] = []
    free_a, free_b = list(range(len(a))), list(range(len(b)))
    for i in list(free_a):
        j = next((j for j in free_b if str(a[i].name) == str(b[j].name)), None)
        if j is not None:
            pairs.append((i, j, "name"))
            free_a.remove(i)
            free_b.remove(j)
    for kind, key in (("content", lambda s: _digest(_hashes(s))),
                      ("content_reordered", lambda s: _digest(np.sort(_hashes(s))))):
        keys_b = {j: key(b[j]) for j in free_b}
        for i in list(free_a):
            ki = key(a[i])
            j = next((j for j in free_b if keys_b[j] == ki), None)
            if j is not None:
                pairs.append((i, j, kind))
                free_a.remove(i)
                free_b.remove(j)
    return pairs


def compare_content(path_a: Path, path_b: Path) -> dict[str, Any]:
    """Compare two parquet files. Returns facts plus a verdict from CONTENT_SAME
    or SAME_SHAPE_DIFFERENT_VALUES / DIFFERENT."""
    result: dict[str, Any] = {"path_a": str(path_a), "path_b": str(path_b)}
    missing = [str(p) for p in (path_a, path_b) if not p.exists()]
    if missing:
        return {**result, "verdict": "UNAVAILABLE", "missing": missing}

    sha_a, sha_b = _sha256(path_a), _sha256(path_b)
    file_a, file_b = pq.ParquetFile(path_a), pq.ParquetFile(path_b)
    df_a, df_b = file_a.read().to_pandas(), file_b.read().to_pandas()
    result.update({
        "bytes": [path_a.stat().st_size, path_b.stat().st_size],
        "file_sha256_equal": sha_a == sha_b,
        "parquet_created_by": [file_a.metadata.created_by, file_b.metadata.created_by],
        "arrow_schema_equal": file_a.schema_arrow.equals(file_b.schema_arrow),
        "rows": [len(df_a), len(df_b)],
        "cols": [df_a.shape[1], df_b.shape[1]],
        "columns_a": [str(c) for c in df_a.columns],
        "columns_b": [str(c) for c in df_b.columns],
    })
    if sha_a == sha_b:
        return {**result, "verdict": "IDENTICAL_FILE", "column_relation": "identical"}

    cols_a = [_normalise(df_a.iloc[:, i]).rename(df_a.columns[i]) for i in range(df_a.shape[1])]
    cols_b = [_normalise(df_b.iloc[:, i]).rename(df_b.columns[i]) for i in range(df_b.shape[1])]
    same_rows = len(df_a) == len(df_b)

    # Column matching needs equal-length columns to compare content.
    pairs = _match_columns(cols_a, cols_b) if same_rows else [
        (i, j, "name") for i, ca in enumerate(cols_a)
        for j, cb in enumerate(cols_b) if str(ca.name) == str(cb.name)
    ]
    matched_a = {i for i, _, _ in pairs}
    matched_b = {j for _, j, _ in pairs}
    result["unmatched_a"] = [str(cols_a[i].name) for i in range(len(cols_a)) if i not in matched_a]
    result["unmatched_b"] = [str(cols_b[j].name) for j in range(len(cols_b)) if j not in matched_b]
    renames = [(str(cols_a[i].name), str(cols_b[j].name), kind) for i, j, kind in pairs
               if str(cols_a[i].name) != str(cols_b[j].name)]
    result["renamed_columns"] = [
        {"a": a, "b": b, "matched_by": kind, "cosmetic": _name_key(a) == _name_key(b)}
        for a, b, kind in renames
    ]
    order_b = [j for _, j, _ in sorted(pairs)]
    if not renames and not result["unmatched_a"] and not result["unmatched_b"]:
        relation = "identical" if order_b == sorted(order_b) else "reordered"
    elif renames and not result["unmatched_a"] and not result["unmatched_b"]:
        relation = "renamed"
    else:
        relation = "partial"
    result["column_relation"] = relation

    if not same_rows or not pairs:
        return {**result, "verdict": "DIFFERENT"}

    aligned_a = pd.concat([cols_a[i].reset_index(drop=True) for i, _, _ in sorted(pairs)], axis=1)
    aligned_b = pd.concat([cols_b[j].reset_index(drop=True) for i, j, _ in sorted(pairs)], axis=1)
    aligned_a.columns = aligned_b.columns = range(len(pairs))
    rows_a = pd.util.hash_pandas_object(aligned_a, index=False).to_numpy()
    rows_b = pd.util.hash_pandas_object(aligned_b, index=False).to_numpy()
    ordered_equal = bool(np.array_equal(rows_a, rows_b))
    multiset_equal = bool(np.array_equal(np.sort(rows_a), np.sort(rows_b)))
    result["row_order_equal"] = ordered_equal
    result["row_multiset_equal"] = multiset_equal
    overlap = sum((Counter(rows_a.tolist()) & Counter(rows_b.tolist())).values())
    result["row_overlap_pct"] = round(100 * overlap / max(len(rows_a), 1), 3)

    numeric = []
    for i, j, _ in sorted(pairs):
        na = pd.to_numeric(df_a.iloc[:, i], errors="coerce") if pd.api.types.is_numeric_dtype(df_a.iloc[:, i]) else None
        nb = pd.to_numeric(df_b.iloc[:, j], errors="coerce") if pd.api.types.is_numeric_dtype(df_b.iloc[:, j]) else None
        if na is not None and nb is not None:
            numeric.append({"column": str(df_a.columns[i]), "sum_a": float(na.sum()), "sum_b": float(nb.sum()),
                            "nulls_a": int(na.isna().sum()), "nulls_b": int(nb.isna().sum())})
    result["numeric_columns"] = numeric

    fully_matched = relation != "partial"
    if fully_matched and multiset_equal:
        if relation == "renamed":
            verdict = "SAME_CONTENT_RENAMED_COLUMNS"
        elif relation == "reordered" or not ordered_equal:
            verdict = "SAME_CONTENT_REORDERED"
        else:
            verdict = "IDENTICAL_CONTENT"
        return {**result, "verdict": verdict}

    # Same shape but not the same rows: measure how far apart the values are.
    differing: dict[str, int] = {}
    max_rel = 0.0
    for k in range(len(pairs)):
        col_a, col_b = aligned_a[k], aligned_b[k]
        equal = (col_a == col_b).fillna(False).astype(bool)
        unequal = ~(equal | (col_a.isna() & col_b.isna()))
        if int(unequal.sum()):
            differing[str(df_a.columns[sorted(pairs)[k][0]])] = int(unequal.sum())
            if pd.api.types.is_float_dtype(col_a) and pd.api.types.is_float_dtype(col_b):
                both_valid = (col_a != _NULL_FLOAT) & (col_b != _NULL_FLOAT)
                denom = col_a.abs().where(col_a.abs() > 0)
                rel = ((col_a - col_b).abs() / denom)[unequal & both_valid].dropna()
                if len(rel):
                    max_rel = max(max_rel, float(rel.max()))
    total_cells = len(aligned_a) * len(pairs)
    result["differing_cells"] = sum(differing.values())
    result["differing_cells_pct"] = round(100 * sum(differing.values()) / max(total_cells, 1), 3)
    result["differing_columns"] = dict(sorted(differing.items(), key=lambda kv: -kv[1])[:10])
    result["max_relative_numeric_deviation"] = max_rel
    return {**result, "verdict": "SAME_SHAPE_DIFFERENT_VALUES" if fully_matched else "DIFFERENT"}


# ---------------------------------------------------------------------- metadata


def _plain(text: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", str(text or "")))).strip()


def fetch_solr_docs(solr_url: str, core: str, resource_ids: list[str], batch: int = 100) -> dict[str, dict[str, Any]]:
    """Solr documents by resource id. Ids Solr does not know are absent; a failed
    request maps every id of that batch to {"_error": ...}."""
    found: dict[str, dict[str, Any]] = {}
    for start in range(0, len(resource_ids), batch):
        chunk = resource_ids[start:start + batch]
        query = urllib.parse.urlencode({
            "q": "resource_id:(" + " OR ".join(f'"{r}"' for r in chunk) + ")",
            "rows": len(chunk), "wt": "json", "fl": ",".join(SOLR_FIELDS),
        })
        try:
            with urllib.request.urlopen(f"{solr_url.rstrip('/')}/{core}/select?{query}", timeout=30) as response:
                docs = json.load(response)["response"]["docs"]
            found.update({doc["resource_id"]: doc for doc in docs})
        except Exception as exc:  # noqa: BLE001 - Solr being down must not stop the content check
            found.update({r: {"_error": f"{type(exc).__name__}: {exc}"} for r in chunk})
    return found


def fetch_solr_doc(solr_url: str, core: str, resource_id: str) -> dict[str, Any] | None:
    return fetch_solr_docs(solr_url, core, [resource_id]).get(resource_id)


def load_ckan_index(path: Path | None) -> dict[str, tuple[dict, dict]]:
    if path is None or not path.exists():
        return {}
    packages = json.loads(path.read_text(encoding="utf-8"))
    return {res["id"]: (pkg, res) for pkg in packages for res in pkg.get("resources", [])}


_ARCGIS_ITEM = re.compile(r"/api/download/v1/items/(?P<item>[0-9a-f]{32})/\w+.*?[?&]layers=(?P<layer>\d+)")
_ARCGIS_DATASET = re.compile(r"/datasets/(?P<item>[0-9a-f]{32})_(?P<layer>\d+)\.\w+")
_ARCGIS_SLUG = re.compile(r"/datasets/(?P<slug>[^/?]+::[^/?]+?)\.\w+(?:\?|$)")


def source_key(url: str | None) -> tuple[str, str] | None:
    """Canonical key of the published source: the ArcGIS item+layer whatever the
    URL form, else the full URL. `slug:` keys cannot be resolved to an item offline."""
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    for pattern in (_ARCGIS_ITEM, _ARCGIS_DATASET):
        match = pattern.search(url)
        if match:
            return parsed.netloc, f"arcgis:{match['item']}_{match['layer']}"
    match = _ARCGIS_SLUG.search(url)
    if match:
        return parsed.netloc, f"slug:{match['slug']}"
    return parsed.netloc, parsed._replace(fragment="").geturl()


_UPLOAD_STAMP = re.compile(r"\d{4}-\d{2}-\d{2}t\d{2}-\d{2}-\d{2}z-?")


def file_name_key(url: str | None) -> tuple[str, str] | None:
    """What the file is called, ignoring where and when it was uploaded: ArcGIS item+layer,
    else the last path segment without upload timestamps. A re-upload keeps this key."""
    key = source_key(url)
    if key is None:
        return None
    if key[1].startswith(("arcgis:", "slug:")):
        return key
    name = _UPLOAD_STAMP.sub("", urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]).casefold())
    return ("name", name) if name else None


def file_relation(url_a: str | None, url_b: str | None) -> str:
    """same_source (one URL/item), same_file_name (re-upload: another URL, same file name),
    different_file_name, unresolved_slug (name slug vs item id) or unknown."""
    key_a, key_b = source_key(url_a), source_key(url_b)
    if key_a and key_b and key_a == key_b:
        return "same_source"
    name_a, name_b = file_name_key(url_a), file_name_key(url_b)
    if not (name_a and name_b):
        return "unknown"
    if name_a == name_b:
        return "same_file_name"
    kinds = {name_a[1].split(":", 1)[0] if name_a[1].startswith(("arcgis:", "slug:")) else "name",
             name_b[1].split(":", 1)[0] if name_b[1].startswith(("arcgis:", "slug:")) else "name"}
    return "unresolved_slug" if kinds == {"arcgis", "slug"} else "different_file_name"


def _field_comparison(a: dict, b: dict, fields: tuple[str, ...], get) -> dict[str, dict[str, Any]]:
    out = {}
    for field in fields:
        va, vb = get(a, field), get(b, field)
        out[field] = {"a": va, "b": vb, "equal": va == vb}
    return out


def _solr_value(doc: dict, field: str) -> Any:
    value = doc.get(field)
    if field == "description":
        return _plain(value)
    if field == "tags":
        return sorted(value or [])
    return value


def _ckan_package_value(record: tuple[dict, dict], field: str) -> Any:
    package = record[0]
    if field == "organization":
        return (package.get("organization") or {}).get("title")
    if field == "tags":
        return sorted(t.get("name") if isinstance(t, dict) else t for t in package.get("tags", []))
    value = package.get(field)
    return _plain(value) if field == "notes" else value


def _ckan_extras(record: tuple[dict, dict]) -> dict[str, Any]:
    return {e["key"]: _plain(e.get("value")) for e in record[0].get("extras", []) if e.get("key") in CKAN_EXTRA_DESCRIPTIVE}


def compare_metadata(res_a: str, res_b: str, solr: tuple[str, str] | None,
                     ckan: dict[str, tuple[dict, dict]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    docs = [fetch_solr_doc(*solr, r) for r in (res_a, res_b)] if solr else [None, None]
    if all(isinstance(d, dict) and "_error" not in d for d in docs):
        result["solr"] = _field_comparison(
            docs[0], docs[1], SOLR_DESCRIPTIVE + SOLR_BOOKKEEPING, _solr_value)
    else:
        result["solr_unavailable"] = [d if d is None or "_error" in d else "ok" for d in docs]

    records = [ckan.get(res_a), ckan.get(res_b)]
    if all(records):
        result["ckan_package"] = _field_comparison(
            records[0], records[1], CKAN_PACKAGE_DESCRIPTIVE + CKAN_PACKAGE_BOOKKEEPING, _ckan_package_value)
        result["ckan_resource"] = _field_comparison(
            records[0], records[1], CKAN_RESOURCE_DESCRIPTIVE + CKAN_RESOURCE_BOOKKEEPING,
            lambda rec, f: _plain(rec[1].get(f)) if f == "description" else rec[1].get(f))
        extras = [_ckan_extras(r) for r in records]
        result["ckan_extras"] = {
            key: {"a": extras[0].get(key), "b": extras[1].get(key), "equal": extras[0].get(key) == extras[1].get(key)}
            for key in CKAN_EXTRA_DESCRIPTIVE if key in extras[0] or key in extras[1]
        }
        result["same_package"] = records[0][0]["id"] == records[1][0]["id"]
    elif ckan:
        result["ckan_unavailable"] = [bool(r) for r in records]

    # Where the file came from: same ArcGIS item/layer under two URL forms is one source.
    urls = [
        (docs[i].get("download_url") if isinstance(docs[i], dict) and "_error" not in docs[i] else None)
        or (records[i][1].get("url") if records[i] else None)
        for i in range(2)
    ]
    keys = [source_key(u) for u in urls]
    relation = file_relation(urls[0], urls[1])
    result["source"] = {"urls": urls, "keys": [list(k) if k else None for k in keys], "relation": relation}

    def differing(group: str, fields: tuple[str, ...]) -> tuple[int, list[str]]:
        compared = [f for f in fields if f in result.get(group, {})]
        return len(compared), [f"{group}.{f}" for f in compared if not result[group][f]["equal"]]

    n_solr, diff_solr = differing("solr", SOLR_DESCRIPTIVE)
    n_pkg, diff_pkg = differing("ckan_package", CKAN_PACKAGE_DESCRIPTIVE)
    n_res, diff_res = differing("ckan_resource", CKAN_RESOURCE_DESCRIPTIVE)
    _, diff_ext = differing("ckan_extras", CKAN_EXTRA_DESCRIPTIVE)
    result["extras_differences"] = diff_ext
    result["index_descriptive_equal"] = n_solr > 0 and not diff_solr
    result["index_differences"] = diff_solr
    result["ckan_differences"] = diff_pkg + diff_res
    result["descriptive_fields_compared"] = n_solr + n_pkg + n_res
    result["descriptive_equal"] = result["descriptive_fields_compared"] > 0 and not (diff_solr + diff_pkg + diff_res)
    bookkeeping = [
        (result.get("solr", {}), SOLR_BOOKKEEPING),
        (result.get("ckan_package", {}), CKAN_PACKAGE_BOOKKEEPING),
        (result.get("ckan_resource", {}), CKAN_RESOURCE_BOOKKEEPING),
    ]
    result["bookkeeping_differences"] = sorted({
        f for g, fields in bookkeeping for f in fields if f in g and not g[f]["equal"]})
    return result


def member_file_url(solr_doc: dict[str, Any] | None, ckan_record: tuple[dict, dict] | None) -> str | None:
    """Where the table's file was published from (Solr download_url, else the CKAN resource url)."""
    url = solr_doc.get("download_url") if solr_doc and "_error" not in solr_doc else None
    return url or (ckan_record[1].get("url") if ckan_record else None)


def descriptive_signature(solr_doc: dict[str, Any] | None, ckan_record: tuple[dict, dict] | None) -> str | None:
    """Everything a reader could use to tell a table from a same-content twin:
    the Solr description fields plus the CKAN package and resource fields (name, period).
    Two tables with equal signatures cannot be told apart by their documentation.
    None when nothing is known about the table, so it never merges with another."""
    parts: dict[str, Any] = {}
    if solr_doc and "_error" not in solr_doc:
        parts["solr"] = {f: _solr_value(solr_doc, f) for f in SOLR_DESCRIPTIVE}
    if ckan_record:
        parts["package"] = {f: _ckan_package_value(ckan_record, f) for f in CKAN_PACKAGE_DESCRIPTIVE}
        parts["resource"] = {
            f: _plain(ckan_record[1].get(f)) if f == "description" else ckan_record[1].get(f)
            for f in CKAN_RESOURCE_DESCRIPTIVE
        }
    return json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str) if parts else None


# --------------------------------------------------------------------- top level


def conclude(content: dict, metadata: dict) -> tuple[str, str]:
    """Twins are backups or re-uploads: identical content, identical description (publisher,
    title, period, columns) and no sign of being different files."""
    verdict = content["verdict"]
    if verdict in CONTENT_SAME:
        extra = metadata.get("bookkeeping_differences") or []
        note = f"; only bookkeeping differs: {', '.join(extra)}" if extra else "; metadata identical"
        relation = metadata.get("source", {}).get("relation")
        if metadata.get("index_differences"):
            fields = ", ".join(metadata["index_differences"])
            if relation == "same_source":
                return "SAME_FILE_DIFFERENT_LABELS", f"one published file catalogued with different {fields}: not a re-upload, review"
            return "SAME_DATA_DIFFERENT_DESCRIPTION", f"same content but the index tells them apart: {fields}"
        if not metadata.get("index_descriptive_equal"):
            return "SAME_DATA_METADATA_UNVERIFIED", "same content; Solr metadata could not be compared"
        if metadata.get("ckan_differences"):
            fields = ", ".join(metadata["ckan_differences"])
            if relation == "same_source":
                return "SAME_FILE_DIFFERENT_LABELS", f"one published file catalogued with different {fields}: not a re-upload, review"
            return "SAME_DATA_CKAN_ONLY_DIFFERENCES", f"same content and indexed description; CKAN records differ in: {fields}{note}"
        if relation == "different_file_name":
            return "SAME_DATA_DIFFERENT_FILE_NAMES", "same content and description, but the files are named differently (the name may carry a period): review"
        return "SAME_TABLE", "re-upload or backup: same content, same description" + (
            ", same published file" if relation == "same_source" else "") + note
    if verdict == "SAME_SHAPE_DIFFERENT_VALUES":
        return "VARIANT_SAME_SHAPE", (
            f"same rows/columns but {content.get('differing_cells_pct')}% of cells differ "
            f"(max relative numeric deviation {content.get('max_relative_numeric_deviation', 0):.4%})")
    return "DIFFERENT_TABLES", f"content verdict {verdict}"


def check_pair(a: str, b: str, *, parquet_dir: Path, raw_dir: Path | None,
               solr: tuple[str, str] | None, ckan: dict, label: str = "") -> dict[str, Any]:
    (_, res_a), (_, res_b) = split_table_id(a), split_table_id(b)
    content = compare_content(table_file(parquet_dir, a), table_file(parquet_dir, b))
    raw = compare_content(table_file(raw_dir, a), table_file(raw_dir, b)) if raw_dir else None
    metadata = compare_metadata(res_a, res_b, solr, ckan)
    conclusion, reason = conclude(content, metadata)
    return {"label": label, "a": Path(a).name.removesuffix(".parquet"), "b": Path(b).name.removesuffix(".parquet"),
            "content": content, "raw_content": raw, "metadata": metadata,
            "conclusion": conclusion, "reason": reason}


def _short(value: Any, width: int = 90) -> str:
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def render_markdown(results: list[dict[str, Any]], title: str = "Table twin check") -> str:
    lines = [f"# {title}", "", "| pair | content | raw content | source | metadata | conclusion |", "|---|---|---|---|---|---|"]
    for r in results:
        raw = (r["raw_content"] or {}).get("verdict", "-")
        lines.append(
            f"| {r['label'] or '-'} | {r['content']['verdict']} | {raw} | "
            f"{r['metadata']['source']['relation']} | "
            f"{'index equal' if r['metadata'].get('index_descriptive_equal') else 'index differs'}, "
            f"{'ckan equal' if not r['metadata'].get('ckan_differences') else 'ckan differs'} | "
            f"**{r['conclusion']}** |")
    for r in results:
        c, m = r["content"], r["metadata"]
        lines += ["", f"## {r['label'] or r['a']}", "", f"- A: `{r['a']}`", f"- B: `{r['b']}`",
                  f"- conclusion: **{r['conclusion']}** — {r['reason']}", "", "### Content", ""]
        if c["verdict"] == "UNAVAILABLE":
            lines.append(f"- missing files: {c['missing']}")
        else:
            lines += [
                f"- verdict: `{c['verdict']}`; file bytes {c['bytes'][0]} / {c['bytes'][1]}; "
                f"sha256 equal: {c['file_sha256_equal']}; arrow schema equal: {c['arrow_schema_equal']}",
                f"- shape: {c['rows'][0]}x{c['cols'][0]} vs {c['rows'][1]}x{c['cols'][1]}; "
                f"column relation: {c.get('column_relation')}",
            ]
            if "row_multiset_equal" in c:
                lines.append(f"- rows in same order: {c['row_order_equal']}; same rows ignoring order: "
                             f"{c['row_multiset_equal']}; rows of A found in B: {c['row_overlap_pct']}%")
            if c.get("renamed_columns"):
                lines.append("- renamed columns: " + "; ".join(
                    f"`{x['a']}` → `{x['b']}` ({x['matched_by']}{', cosmetic' if x['cosmetic'] else ''})"
                    for x in c["renamed_columns"]))
            if c.get("unmatched_a") or c.get("unmatched_b"):
                lines.append(f"- unmatched columns: A {c['unmatched_a']}, B {c['unmatched_b']}")
            if "differing_cells" in c:
                lines.append(f"- differing cells: {c['differing_cells']} ({c['differing_cells_pct']}%), "
                             f"by column {c['differing_columns']}, max relative deviation "
                             f"{c['max_relative_numeric_deviation']:.4%}")
            if c.get("numeric_columns"):
                unequal = [n for n in c["numeric_columns"] if n["sum_a"] != n["sum_b"] or n["nulls_a"] != n["nulls_b"]]
                lines.append(f"- numeric columns: {len(c['numeric_columns'])}, with a different sum or null count: {len(unequal)}")
        raw = r["raw_content"]
        if raw:
            lines.append(f"- raw (pre-cleaning) files: `{raw['verdict']}`"
                         + (f", sha256 equal: {raw['file_sha256_equal']}" if "file_sha256_equal" in raw else ""))
        lines += ["", "### Metadata", ""]
        src = m["source"]
        lines.append(f"- source: **{src['relation']}** — A `{_short(src['urls'][0], 140)}`, B `{_short(src['urls'][1], 140)}`")
        if "same_package" in m:
            lines.append(f"- same CKAN package: {m['same_package']}")
        if m.get("solr_unavailable"):
            lines.append(f"- Solr comparison unavailable: {m['solr_unavailable']}")
        if m.get("ckan_unavailable"):
            lines.append(f"- CKAN record found for A/B: {m['ckan_unavailable']}")
        lines += ["", "| source | field | kind | equal | A | B |", "|---|---|---|---|---|---|"]
        for group, descriptive in (("solr", SOLR_DESCRIPTIVE), ("ckan_package", CKAN_PACKAGE_DESCRIPTIVE),
                                   ("ckan_resource", CKAN_RESOURCE_DESCRIPTIVE), ("ckan_extras", CKAN_EXTRA_DESCRIPTIVE)):
            for field, cmp in m.get(group, {}).items():
                kind = "descriptive" if field in descriptive else "bookkeeping"
                if cmp["equal"] and kind == "bookkeeping":
                    continue  # equal bookkeeping is noise; differences are what matter
                if cmp["equal"]:
                    lines.append(f"| {group} | {field} | {kind} | yes | {_short(cmp['a'], 50)} | = |")
                else:
                    lines.append(f"| {group} | {field} | {kind} | **no** | {_short(cmp['a'])} | {_short(cmp['b'])} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="*", help="two table ids: A B")
    parser.add_argument("--pairs", type=Path, help="JSON list of {label, a, b}")
    parser.add_argument("--core", default="uk", help="Solr core and data folder name (default: uk)")
    parser.add_argument("--parquet-dir", type=Path)
    parser.add_argument("--raw-dir", help="pre-cleaning parquet folder; '' to skip")
    parser.add_argument("--ckan-metadata", help="metadata_retrieved_only.json; '' to skip")
    parser.add_argument("--solr-url", default=os.environ.get("SOLR_BASE_URL", "http://localhost:8983/solr"))
    parser.add_argument("--no-solr", action="store_true")
    parser.add_argument("--md", type=Path, help="write a Markdown report here")
    parser.add_argument("--json", type=Path, help="write the full results here")
    args = parser.parse_args(argv)

    if args.pairs:
        pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    elif len(args.tables) == 2:
        pairs = [{"label": "", "a": args.tables[0], "b": args.tables[1]}]
    else:
        parser.error("give two table ids, or --pairs")

    data_dir = ROOT_DIR / "data" / args.core
    parquet_dir = args.parquet_dir or data_dir / "clean_datasets" / "parquet"
    raw_dir = data_dir / "datasets" / "parquet" if args.raw_dir is None else (Path(args.raw_dir) if args.raw_dir else None)
    ckan_path = data_dir / "metadata" / "metadata_retrieved_only.json" if args.ckan_metadata is None else (
        Path(args.ckan_metadata) if args.ckan_metadata else None)
    ckan = load_ckan_index(ckan_path)
    solr = None if args.no_solr else (args.solr_url, args.core)

    results = [
        check_pair(p["a"], p["b"], parquet_dir=parquet_dir, raw_dir=raw_dir, solr=solr, ckan=ckan,
                   label=p.get("label", ""))
        for p in pairs
    ]
    for r in results:
        print(f"{r['label'] or '-':<8} {r['conclusion']:<34} content={r['content']['verdict']:<30} "
              f"source={r['metadata']['source']['relation']:<16} {r['reason']}")
    if args.md:
        args.md.write_text(render_markdown(results), encoding="utf-8")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
