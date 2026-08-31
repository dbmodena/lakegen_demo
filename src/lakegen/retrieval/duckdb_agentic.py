"""Controlled, index-free keyword retrieval over local Parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

import duckdb
import pyarrow.parquet as pq

from lakegen.retrieval.config import RetrievalConfig
from lakegen.retrieval.models import RetrievalHit


_WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)
_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "which",
    "where", "when", "who", "how", "are", "was", "were", "has", "have",
    "dei", "del", "della", "delle", "degli", "con", "per", "che", "come",
    "quale", "quali", "dove", "sono", "nel", "nella", "nelle", "una", "uno",
}
_TERM_ALIASES = {
    "borough": ("boro",),
    "organization": ("organisation", "partner", "operator", "agency"),
    "community": ("neighborhood", "neighbourhood"),
    "identifier": (" id", "_id", "code"),
}


def _normalize_term(term: str) -> str:
    """Apply a deliberately small plural normalization for keyword matching."""
    if len(term) > 5 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _extract_terms(texts: Sequence[str], *, limit: int) -> list[str]:
    values: list[str] = []
    for text in texts:
        for term in _WORD.findall(str(text).casefold()):
            term = _normalize_term(term)
            if term not in _STOPWORDS and term not in values:
                values.append(term)
    return values[:limit]


def _query_terms(question: str, keywords: Sequence[str]) -> tuple[list[str], list[str]]:
    """Keep agent concepts primary and question-derived recall terms secondary."""
    primary = _extract_terms([str(value) for value in keywords], limit=8)
    if not primary:
        primary = _extract_terms([question], limit=8)
        return primary, []
    secondary = [
        term for term in _extract_terms([question], limit=12) if term not in primary
    ][:4]
    return primary, secondary


def _term_variants(term: str) -> tuple[str, ...]:
    return (term, *_TERM_ALIASES.get(term, ()))


def _contains_term(text: str, term: str) -> bool:
    return any(variant in text for variant in _term_variants(term))


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _search_projection(columns: Sequence[str]) -> str:
    """Build one bounded row text, avoiding terms × columns SQL expansion."""
    values = ", ".join(
        f"coalesce(cast({_identifier(column)} as varchar), '')"
        for column in columns
    )
    return f"lower(concat_ws(' ', {values}))"


def _named_values(values: Any) -> tuple[str, ...]:
    result: list[str] = []
    for value in values if isinstance(values, list) else []:
        if isinstance(value, dict):
            value = value.get("display_name") or value.get("title") or value.get("name")
        text = str(value or "").strip()
        if text:
            result.append(text)
    return tuple(result)


@lru_cache(maxsize=8)
def _load_normalized_metadata(path_text: str) -> dict[str, dict[str, Any]]:
    path = Path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        # Socrata/NYC: one top-level item maps directly to one table resource.
        resource = item.get("resource")
        if isinstance(resource, dict):
            resource_id = str(resource.get("id") or "").strip()
            if not resource_id:
                continue
            classification = item.get("classification")
            classification = classification if isinstance(classification, dict) else {}
            tags = (
                *_named_values(classification.get("tags")),
                *_named_values(classification.get("domain_tags")),
                str(classification.get("domain_category") or "").strip(),
            )
            catalog[resource_id.casefold()] = {
                "title": str(resource.get("name") or "").strip(),
                "description": str(resource.get("description") or "").strip(),
                "tags": tuple(tag for tag in tags if tag),
                "columns": _named_values(resource.get("columns_name")),
                "column_descriptions": _named_values(
                    resource.get("columns_description")
                ),
            }
            continue

        # CKAN/UK: one package contains multiple resources. Local filenames use
        # ``package_uuid___resource_uuid``, so preserve resource-level identity.
        package_id = str(item.get("id") or "").strip()
        if not package_id:
            continue
        organization = item.get("organization")
        organization = organization if isinstance(organization, dict) else {}
        package_tags = (
            *_named_values(item.get("tags")),
            *_named_values(item.get("groups")),
            str(item.get("theme-primary") or "").strip(),
            str(organization.get("title") or organization.get("name") or "").strip(),
        )
        package_title = str(item.get("title") or item.get("name") or "").strip()
        package_description = str(item.get("notes") or "").strip()
        resources = item.get("resources")
        for child in resources if isinstance(resources, list) else []:
            if not isinstance(child, dict):
                continue
            resource_id = str(child.get("id") or "").strip()
            if not resource_id:
                continue
            key = f"{package_id}___{resource_id}".casefold()
            resource_title = str(child.get("name") or "").strip()
            resource_description = str(child.get("description") or "").strip()
            catalog[key] = {
                "title": resource_title or package_title,
                "description": " ".join(
                    part for part in (package_description, resource_description) if part
                ),
                "tags": tuple(tag for tag in package_tags if tag),
                "columns": (),
                "column_descriptions": (),
            }
    return catalog


@dataclass(frozen=True)
class _CatalogEntry:
    path: Path
    rows: int
    columns: tuple[tuple[str, str], ...]
    title: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    metadata_term_scores: tuple[tuple[str, float], ...] = ()
    preliminary_score: float = 0.0
    preliminary_coverage: int = 0


@dataclass
class _SearchEvidence:
    entry: _CatalogEntry
    searched_columns: list[str]
    schema_columns: list[str]
    filename_terms: set[str]
    schema_terms: set[str]
    metadata_term_scores: dict[str, float]
    term_counts: dict[str, int]
    joint_count: int
    score: float = 0.0


class DuckDBAgenticRetriever:
    """Search Parquet schemas and values through bounded, generated DuckDB SQL.

    Callers cannot provide SQL, paths, or column expressions. The retriever first
    reads inexpensive Parquet footer metadata, then scans only bounded candidate
    files/columns and returns counts plus small samples as retrieval evidence.
    The surrounding LakeGen discovery agent supplies/refines the concepts and its
    existing inspection tools verify candidates before selection.
    """

    def __init__(self, config: RetrievalConfig, table_dir: str | Path) -> None:
        self.config = config
        self.table_dir = Path(table_dir).resolve()
        if not self.table_dir.is_dir():
            raise ValueError(f"DuckDB retrieval table directory does not exist: {table_dir}")

    def _metadata_path(self) -> Path | None:
        for ancestor in (self.table_dir, *self.table_dir.parents):
            candidate = ancestor / "metadata" / "metadata_retrieved_cleaned.json"
            if candidate.is_file():
                return candidate
            candidate = ancestor / "metadata" / "metadata_retrieved_only.json"
            if candidate.is_file():
                return candidate
            candidate = ancestor / "metadata" / "metadata.json"
            if candidate.is_file():
                return candidate
        return None

    def _metadata_catalog(self) -> dict[str, dict[str, Any]]:
        path = self._metadata_path()
        if path is None:
            return {}
        return _load_normalized_metadata(str(path.resolve()))

    @staticmethod
    def _metadata_scores(metadata: dict[str, Any], terms: Sequence[str]) -> dict[str, float]:
        fields = (
            (str(metadata.get("title", "")).casefold(), 20.0),
            (" ".join(metadata.get("tags", ())).casefold(), 12.0),
            (" ".join(metadata.get("columns", ())).casefold(), 10.0),
            (str(metadata.get("description", "")).casefold(), 8.0),
            (" ".join(metadata.get("column_descriptions", ())).casefold(), 5.0),
        )
        return {
            term: max(
                (weight for text, weight in fields if _contains_term(text, term)),
                default=0.0,
            )
            for term in terms
        }

    def _catalog(self, terms: Sequence[str]) -> list[_CatalogEntry]:
        paths = sorted((*self.table_dir.glob("*.parquet"), *self.table_dir.glob("*.pq")))
        metadata_catalog = self._metadata_catalog()
        entries: list[_CatalogEntry] = []
        for path in paths:
            metadata = metadata_catalog.get(path.stem.casefold(), {})
            scores = self._metadata_scores(metadata, terms)
            try:
                parquet = pq.ParquetFile(path)
                columns = tuple(
                    (field.name, str(field.type)) for field in parquet.schema_arrow
                )
                filename_terms = {
                    term for term in terms
                    if _contains_term(path.stem.casefold(), term)
                }
                schema_terms = {
                    term for term in terms
                    if any(_contains_term(name.casefold(), term) for name, _ in columns)
                }
                coverage = len(filename_terms | schema_terms | {
                    term for term, score in scores.items() if score > 0
                })
                preliminary_score = (
                    sum(scores.values())
                    + 8.0 * len(filename_terms)
                    + 5.0 * len(schema_terms)
                )
                entries.append(_CatalogEntry(
                    path,
                    parquet.metadata.num_rows,
                    columns,
                    title=str(metadata.get("title", "")),
                    description=str(metadata.get("description", "")),
                    tags=tuple(metadata.get("tags", ())),
                    metadata_term_scores=tuple(scores.items()),
                    preliminary_score=preliminary_score,
                    preliminary_coverage=coverage,
                ))
            except (OSError, ValueError):
                continue
        entries.sort(key=lambda entry: (
            -entry.preliminary_score,
            -entry.preliminary_coverage,
            entry.path.name,
        ))
        return entries

    def _probe_has_value_match(
        self,
        con: duckdb.DuckDBPyConnection,
        entry: _CatalogEntry,
        terms: Sequence[str],
    ) -> bool:
        """Probe a small prefix using only internally generated identifiers/SQL."""
        searchable = [
            name for name, dtype in entry.columns
            if any(token in dtype.casefold() for token in ("string", "varchar"))
        ][: self.config.duckdb_max_columns_per_file]
        if not searchable:
            return False
        projection = _search_projection(searchable)
        field = _identifier("__lakegen_search_text")
        variants = [variant for term in terms for variant in _term_variants(term)]
        predicates = " OR ".join(f"{field} like ?" for _ in variants)
        source = str(entry.path).replace("'", "''")
        columns = ", ".join(_identifier(name) for name in searchable)
        probe_rows = min(
            self.config.duckdb_probe_rows_per_file,
            self.config.duckdb_max_scan_rows_per_file,
        )
        try:
            return bool(con.execute(
                f"WITH source AS (SELECT {columns} FROM read_parquet('{source}') "
                f"LIMIT {probe_rows}), searchable AS (SELECT {projection} AS "
                f"{field} FROM source) SELECT count(*) FROM searchable WHERE "
                f"{predicates} LIMIT 1",
                [f"%{variant}%" for variant in variants],
            ).fetchone()[0])
        except duckdb.Error:
            return False

    def retrieve(
        self, question: str, keywords: Sequence[str], *, top_k: int
    ) -> list[RetrievalHit]:
        primary_terms, secondary_terms = _query_terms(question, keywords)
        terms = [*primary_terms, *secondary_terms]
        if not terms:
            return []
        catalog = self._catalog(terms)
        con = duckdb.connect(":memory:")
        evidence_rows: list[_SearchEvidence] = []
        try:
            primary_candidates = catalog[: self.config.duckdb_max_files]
            probe_candidates = catalog[
                self.config.duckdb_max_files:
                self.config.duckdb_max_files + self.config.duckdb_probe_files
            ]
            probed_matches = [
                entry for entry in probe_candidates
                if self._probe_has_value_match(con, entry, terms)
            ]
            # Value-only matches get priority over equally opaque candidates, but
            # the expensive second phase still scans at most duckdb_max_files.
            candidate_pool = [(entry, False) for entry in primary_candidates]
            candidate_pool.extend((entry, True) for entry in probed_matches)
            candidate_pool.sort(key=lambda item: (
                -int(item[1]),
                -item[0].preliminary_score,
                -item[0].preliminary_coverage,
                item[0].path.name,
            ))
            selected_entries = [
                entry for entry, _ in candidate_pool[: self.config.duckdb_max_files]
            ]
            for entry in selected_entries:
                # Always search the bounded textual projection. Restricting content
                # search to columns whose *names* match a term loses tables where a
                # subject appears only in values (the failure found by the canary).
                searchable = [
                    name for name, dtype in entry.columns
                    if any(token in dtype.casefold() for token in ("string", "varchar"))
                ]
                searchable = searchable[: self.config.duckdb_max_columns_per_file]
                filename_terms = {
                    term for term in terms
                    if _contains_term(entry.path.stem.casefold(), term)
                }
                schema_terms = {
                    term for term in terms
                    if any(_contains_term(name.casefold(), term) for name, _ in entry.columns)
                }
                schema_columns = [
                    name for name, _ in entry.columns
                    if any(_contains_term(name.casefold(), term) for term in terms)
                ]
                metadata_term_scores = dict(entry.metadata_term_scores)
                term_counts = {term: 0 for term in terms}
                joint_count = 0
                if searchable:
                    search_projection = _search_projection(searchable)
                    search_field = _identifier("__lakegen_search_text")
                    term_predicates = {
                        term: "(" + " OR ".join(
                            f"{search_field} like ?" for _ in _term_variants(term)
                        ) + ")"
                        for term in terms
                    }
                    select_counts = ", ".join(
                        f"count_if({term_predicates[term]})" for term in terms
                    )
                    joint = " AND ".join(term_predicates[term] for term in primary_terms)
                    params = [
                        f"%{variant}%" for term in terms
                        for variant in _term_variants(term)
                    ] + [
                        f"%{variant}%" for term in primary_terms
                        for variant in _term_variants(term)
                    ]
                    source = str(entry.path).replace("'", "''")
                    selected_columns = ", ".join(
                        _identifier(name) for name in searchable
                    )
                    scan_limit = self.config.duckdb_max_scan_rows_per_file
                    try:
                        counts = con.execute(
                            f"WITH source AS (SELECT {selected_columns} "
                            f"FROM read_parquet('{source}') LIMIT {scan_limit}), "
                            f"searchable AS (SELECT {search_projection} AS "
                            f"{search_field} FROM source) "
                            f"SELECT {select_counts}, count_if({joint}) FROM searchable",
                            params,
                        ).fetchone()
                        term_counts = {
                            term: int(counts[index] or 0)
                            for index, term in enumerate(terms)
                        }
                        joint_count = int(counts[-1] or 0)
                    except duckdb.Error:
                        pass
                present = filename_terms | schema_terms | {
                    term for term, count in term_counts.items() if count > 0
                } | {term for term, score in metadata_term_scores.items() if score > 0}
                if not present:
                    continue
                evidence_rows.append(_SearchEvidence(
                    entry=entry,
                    searched_columns=searchable,
                    schema_columns=schema_columns,
                    filename_terms=filename_terms,
                    schema_terms=schema_terms,
                    metadata_term_scores=metadata_term_scores,
                    term_counts=term_counts,
                    joint_count=joint_count,
                ))

            # IDF prevents ubiquitous terms from dominating. Counts contribute
            # only after row-count normalization; distinct term coverage is the
            # primary ranking signal and partial primary matches are penalized.
            document_frequency = {
                term: sum(
                    term in row.filename_terms
                    or term in row.schema_terms
                    or row.metadata_term_scores.get(term, 0) > 0
                    or row.term_counts.get(term, 0) > 0
                    for row in evidence_rows
                )
                for term in terms
            }
            population = max(1, len(evidence_rows))
            for row in evidence_rows:
                score = 0.0
                primary_covered = 0
                for term in terms:
                    count = row.term_counts.get(term, 0)
                    is_present = (
                        term in row.filename_terms or term in row.schema_terms or count > 0
                        or row.metadata_term_scores.get(term, 0) > 0
                    )
                    if not is_present:
                        continue
                    if term in primary_terms:
                        primary_covered += 1
                        weight = 1.0
                    else:
                        weight = 0.25
                    idf = math.log((population + 1) / (document_frequency[term] + 1)) + 1.0
                    field_score = (
                        (8.0 if term in row.filename_terms else 0.0)
                        + (5.0 if term in row.schema_terms else 0.0)
                        + row.metadata_term_scores.get(term, 0.0)
                    )
                    if count:
                        density = count / max(1, row.entry.rows)
                        field_score += 1.0 + 2.0 * math.sqrt(density)
                    score += weight * idf * field_score
                coverage = primary_covered / max(1, len(primary_terms))
                score += 30.0 * coverage**3
                score -= 6.0 * (len(primary_terms) - primary_covered)
                if row.joint_count:
                    score += 12.0 + 3.0 * math.sqrt(
                        row.joint_count / max(1, row.entry.rows)
                    )
                row.score = score

            evidence_rows.sort(key=lambda row: (-row.score, row.entry.path.name))
            hits: list[RetrievalHit] = []
            for row in evidence_rows[:top_k]:
                entry = row.entry
                match_count = sum(row.term_counts.values())
                samples: list[dict[str, object]] = []
                if row.searched_columns and match_count:
                    search_projection = _search_projection(row.searched_columns)
                    search_field = _identifier("__lakegen_search_text")
                    variants = [
                        variant for term in terms for variant in _term_variants(term)
                    ]
                    predicates = [f"{search_field} like ?" for _ in variants]
                    params = [f"%{variant}%" for variant in variants]
                    source = str(entry.path).replace("'", "''")
                    names = ", ".join(
                        _identifier(name) for name in row.searched_columns
                    )
                    scan_limit = self.config.duckdb_max_scan_rows_per_file
                    try:
                        values = con.execute(
                            f"WITH source AS (SELECT {names} FROM "
                            f"read_parquet('{source}') LIMIT {scan_limit}), "
                            f"searchable AS (SELECT *, {search_projection} AS "
                            f"{search_field} FROM source) "
                            f"SELECT {names} FROM searchable WHERE "
                            + " OR ".join(predicates)
                            + " LIMIT ?",
                            [*params, self.config.duckdb_sample_rows],
                        ).fetchall()
                        samples = [
                            {
                                name: value if value is None or isinstance(
                                    value, (str, int, float, bool)
                                ) else str(value)
                                for name, value in zip(row.searched_columns, values_row)
                            }
                            for values_row in values
                        ]
                    except duckdb.Error:
                        pass
                document = {
                    "resource_id": entry.path.name,
                    "dataset_id": entry.path.stem,
                    "title": entry.title or entry.path.stem.replace("_", " "),
                    "description": (
                        ((entry.description + " ") if entry.description else "")
                        + f"DuckDB keyword evidence: {len(primary_terms)} primary terms, "
                        f"{sum(count > 0 for count in row.term_counts.values())} found in values; "
                        f"{entry.rows} total rows."
                    ),
                    "columns": [
                        {"name": name, "type": dtype} for name, dtype in entry.columns
                    ],
                    "tags": list(entry.tags),
                    "duckdb_evidence": {
                        "terms": terms,
                        "primary_terms": primary_terms,
                        "secondary_terms": secondary_terms,
                        "matched_columns": row.schema_columns,
                        "searched_columns": row.searched_columns,
                        "term_counts": row.term_counts,
                        "match_count": match_count,
                        "joint_match_count": row.joint_count,
                        "primary_coverage": sum(
                            term in row.filename_terms
                            or term in row.schema_terms
                            or row.metadata_term_scores.get(term, 0) > 0
                            or row.term_counts.get(term, 0) > 0
                            for term in primary_terms
                        ) / max(1, len(primary_terms)),
                        "samples": samples,
                    },
                }
                hits.append(RetrievalHit(document=document, score=row.score))
        finally:
            con.close()
        for rank, hit in enumerate(hits[:top_k], 1):
            hit.rank = rank
            hit.lexical_rank = rank
            hit.lexical_score = hit.score
        return hits[:top_k]
