"""Regex ``grep`` retrieval over local Parquet files, scored with TF-IDF.

This is a content-first alternative to :mod:`lakegen.retrieval.duckdb_agentic`.
Both search local Parquet without an index, but they differ where it matters:

* every column is cast to text, so matches in numeric, temporal, and boolean
  columns are found too (a year, a ZIP code, a BBL) rather than string columns
  only;
* query terms become regular expressions, so a term and its aliases are one
  pass over the data instead of one ``LIKE`` per variant;
* each file is read one column at a time, with a row cap on every read. A lake
  of independently-schema'd tables therefore needs no common schema, and a
  multi-GB table cannot exhaust a worker: only one capped column is ever
  materialized.

Every polars query runs in a worker process (:mod:`lakegen.cell_scan`), never in
the process that calls :meth:`GrepRetriever.retrieve`. Polars has crashed
natively under the thousands of small collects a lake-wide scan makes, and a
segfault in the calling process would take the whole application down with it.
The row cap plus one-column-at-a-time reads are what bound each worker's memory,
so the queries deliberately use the default in-memory engine;
``engine="streaming"`` measures the same peak RSS on this workload but segfaults
more readily (polars 1.40).

Scoring is TF-IDF over the query terms: a term's inverse document frequency is
computed across the candidate files, so a term occurring in nearly every table
contributes little and a rare one dominates. Per-file term frequency is
normalized by row count, so a large sparse table cannot outrank a small focused
one.

Two modalities share this retriever, selected by :class:`RetrievalMode`:

``grep``
    Cells plus the free signals around them -- filename, column names, and the
    catalog's title, tags, columns, and description -- all contribute to both
    the prefilter and the score.
``grep_values``
    Cells only. Filenames, column names, and the catalog are never consulted to
    select or score a file, so a table is found only where the searched values
    appear in its data. Those values are exactly the ones the discovery model
    listed, each matched whole: nothing is split, aliased, or added from the
    question. The catalog is still read for the title, description,
    and tags a hit carries for display, which keeps the two modalities'
    *results* shaped identically and leaves the ranking as the single variable
    between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Callable, Sequence

import pyarrow.parquet as pq

from lakegen import cell_scan
from lakegen.retrieval.config import RetrievalConfig, RetrievalMode
from lakegen.retrieval.models import RetrievalHit

# Shared with the DuckDB modality on purpose: both turn the same agent concepts
# into the same terms and aliases, so the two are comparable on one benchmark.
from lakegen.retrieval.duckdb_agentic import (
    _contains_term,
    _load_normalized_metadata,
    _query_terms,
    _term_variants,
)


# Catalog fields are not equally informative. In a lake of Socrata/CKAN exports
# the filename is an opaque id (``43nn-pn8j``) and columns are cryptic (``CAMIS``,
# ``DBA``, ``BORO``), so the title carries most of the meaning. These are the same
# weights the DuckDB modality uses, which keeps the two comparable.
_METADATA_WEIGHTS = (
    ("title", 20.0),
    ("tags", 12.0),
    ("columns", 10.0),
    ("description", 8.0),
)


def _term_pattern(term: str) -> str:
    """One case-insensitive alternation over a term and its aliases."""
    return "(?i)" + "|".join(re.escape(variant) for variant in _term_variants(term))


# A cell-value search uses the listed values as given, capped at the number of
# primary concepts the other modes accept.
_MAX_SEARCH_VALUES = 8


def _search_values(values: Sequence[str]) -> list[str]:
    """The listed values, each kept whole: whitespace-normalized and casefolded.

    Nothing is split, stemmed, or taken from the question. A value shorter than
    two characters is dropped, because as a substring it would match nearly
    every cell.
    """
    terms: list[str] = []
    for value in values:
        term = " ".join(str(value).split()).casefold()
        if len(term) >= 2 and term not in terms:
            terms.append(term)
    return terms[:_MAX_SEARCH_VALUES]


def _patterns(
    terms: Sequence[str],
    *,
    literal: bool = False,
    pattern: Callable[[str], str] | None = None,
) -> list[tuple[str, str]]:
    """Each term with its escaped pattern, so a worker never sees a raw term.

    ``literal`` matches each term exactly as given, without the small alias
    table the concept-based modes use to widen a word like ``borough``.
    ``pattern`` replaces both with the caller's own builder, which must escape
    whatever part of the term it keeps.
    """
    if pattern is None:
        pattern = (lambda term: "(?i)" + re.escape(term)) if literal else _term_pattern
    return [(term, pattern(term)) for term in terms]


ScanResult = tuple[dict[str, dict[str, int]], tuple[str, ...], int]


def scan_many(
    paths: Sequence[Path],
    terms: Sequence[str],
    *,
    max_columns: int | None,
    max_rows: int | None,
    workers: int,
    literal: bool = False,
    pattern: Callable[[str], str] | None = None,
) -> list[ScanResult | None]:
    """Scan files in worker processes, one result per path, in the order given.

    ``None`` for ``max_columns`` or ``max_rows`` reads every column or row.

    A result is ``(term -> column -> matching cells, scanned columns,
    uncastable column count)``, or ``None`` for a file that crashed its worker
    twice on its own and was skipped. Scanning a file is independent of every
    other file, so the expensive phase is a parallel map and the ranking that
    follows is a pure reduce over its counts. Order is preserved, so every
    tie-break stays deterministic.
    """
    patterns = _patterns(terms, literal=literal, pattern=pattern)
    return cell_scan.run(
        "scan",
        [(str(path), patterns, max_columns, max_rows) for path in paths],
        workers=workers,
    )


def probe_many(
    paths: Sequence[Path],
    terms: Sequence[str],
    *,
    max_columns: int | None,
    max_rows: int,
    workers: int,
    literal: bool = False,
    pattern: Callable[[str], str] | None = None,
) -> list[int]:
    """Count matching rows in a short prefix of each file, in worker processes."""
    patterns = _patterns(terms, literal=literal, pattern=pattern)
    return [
        count or 0
        for count in cell_scan.run(
            "probe",
            [(str(path), patterns, max_columns, max_rows) for path in paths],
            workers=workers,
        )
    ]


def sample_many(
    jobs: Sequence[tuple[Path, Sequence[str]]],
    terms: Sequence[str],
    *,
    max_rows: int,
    limit: int,
    workers: int,
    literal: bool = False,
) -> list[list[dict[str, Any]]]:
    """A few matching rows for each ``(path, columns)`` job, read in workers."""
    patterns = _patterns(terms, literal=literal)
    tasks = [
        (str(path), list(columns), patterns, max_rows, limit)
        for path, columns in jobs
    ]
    return [rows or [] for rows in cell_scan.run("sample", tasks, workers=workers)]


@dataclass
class _FileEvidence:
    path: Path
    rows: int
    columns: tuple[tuple[str, str], ...]
    title: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    # term -> weight of the strongest catalog field the term was found in
    metadata_scores: dict[str, float] = field(default_factory=dict)
    filename_terms: frozenset[str] = frozenset()
    schema_terms: frozenset[str] = frozenset()
    scanned_columns: tuple[str, ...] = ()
    uncastable_columns: int = 0
    # The scanner crashed on this file twice on its own, so it was skipped.
    unscannable: bool = False
    # term -> column -> matching cell count
    term_columns: dict[str, dict[str, int]] = field(default_factory=dict)
    score: float = 0.0

    @property
    def metadata_terms(self) -> frozenset[str]:
        return frozenset(term for term, score in self.metadata_scores.items() if score)

    @property
    def prefilter_score(self) -> float:
        return (
            8.0 * len(self.filename_terms)
            + 5.0 * len(self.schema_terms)
            + sum(self.metadata_scores.values())
        )

    def term_count(self, term: str) -> int:
        return sum(self.term_columns.get(term, {}).values())

    def present(self, term: str) -> bool:
        return (
            term in self.filename_terms
            or term in self.schema_terms
            or term in self.metadata_terms
            or self.term_count(term) > 0
        )

    def matched_columns(self) -> list[str]:
        names: list[str] = []
        for columns in self.term_columns.values():
            for name in columns:
                if name not in names:
                    names.append(name)
        return names


class GrepRetriever:
    """Rank local Parquet tables by regex matches in their values and schema.

    Callers supply concepts, never patterns: every term is regex-escaped before
    it reaches the engine, so an agent cannot inject an expression that scans
    pathologically. Every file in the lake is scanned by default -- the map is
    run in parallel worker processes, which made that affordable and keeps a
    native polars crash out of this process -- and each read is bounded by
    ``grep_max_columns_per_file`` and ``grep_max_scan_rows_per_file``. Setting
    ``grep_max_files`` reintroduces a cut for deliberately cheap runs, ordered by
    the metadata prefilter here and by ``grep_probe_files`` prefix reads in the
    values-only mode; such a run is marked ``truncated`` in its evidence.

    ``RetrievalMode.GREP_VALUES`` selects the values-only modality described in
    the module docstring; the mode is the single source of truth, so a config
    and a retriever cannot disagree about which one is running.
    """

    def __init__(self, config: RetrievalConfig, table_dir: str | Path) -> None:
        self.config = config
        self.values_only = config.mode is RetrievalMode.GREP_VALUES
        # The mode is the stronger statement: values-only means metadata carries
        # no weight, whatever the config says. Otherwise the weights decide, and
        # a family weighted 0 has its work skipped entirely below.
        self.value_weight = config.grep_value_weight
        self.metadata_weight = 0.0 if self.values_only else config.grep_metadata_weight
        self.table_dir = Path(table_dir).resolve()
        if not self.table_dir.is_dir():
            raise ValueError(
                f"grep retrieval table directory does not exist: {table_dir}"
            )

    def _metadata_catalog(self) -> dict[str, dict[str, Any]]:
        for ancestor in (self.table_dir, *self.table_dir.parents):
            for name in (
                "metadata_retrieved_cleaned.json",
                "metadata_retrieved_only.json",
                "metadata.json",
            ):
                candidate = ancestor / "metadata" / name
                if candidate.is_file():
                    return _load_normalized_metadata(str(candidate.resolve()))
        return {}

    def _candidates(self, terms: Sequence[str]) -> list[_FileEvidence]:
        """Rank every file on free footer/metadata signals before reading values.

        The values-only modality records none of those signals, so every entry
        comes back with an empty filename, schema, and catalog term set. That is
        the whole of how the modality differs downstream: ``prefilter_score``,
        ``present``, and the per-field score terms all read from those sets, so
        each one falls away on its own rather than through a parallel branch.
        The catalog is still read, but only for the title, description, and tags
        the hit displays.
        """
        catalog = self._metadata_catalog()
        entries: list[_FileEvidence] = []
        paths = sorted(
            (*self.table_dir.glob("*.parquet"), *self.table_dir.glob("*.pq"))
        )
        for path in paths:
            try:
                parquet = pq.ParquetFile(path)
                columns = tuple(
                    (field_.name, str(field_.type))
                    for field_ in parquet.schema_arrow
                )
                rows = parquet.metadata.num_rows
            except (OSError, ValueError):
                continue
            metadata = catalog.get(path.stem.casefold(), {})
            metadata_scores: dict[str, float] = {}
            filename_terms: frozenset[str] = frozenset()
            schema_terms: frozenset[str] = frozenset()
            if self.metadata_weight:
                fields = []
                for name, weight in _METADATA_WEIGHTS:
                    value = metadata.get(name, "")
                    text = value if isinstance(value, str) else " ".join(value or ())
                    fields.append((text.casefold(), weight))
                scores = {
                    term: max(
                        (
                            weight
                            for text, weight in fields
                            if _contains_term(text, term)
                        ),
                        default=0.0,
                    )
                    for term in terms
                }
                metadata_scores = {
                    term: score for term, score in scores.items() if score
                }
                filename_terms = frozenset(
                    term
                    for term in terms
                    if _contains_term(path.stem.casefold(), term)
                )
                schema_terms = frozenset(
                    term
                    for term in terms
                    if any(
                        _contains_term(name.casefold(), term) for name, _ in columns
                    )
                )
            entries.append(
                _FileEvidence(
                    path=path,
                    rows=rows,
                    columns=columns,
                    title=str(metadata.get("title", "")),
                    description=str(metadata.get("description", "")),
                    tags=tuple(metadata.get("tags", ())),
                    metadata_scores=metadata_scores,
                    filename_terms=filename_terms,
                    schema_terms=schema_terms,
                )
            )
        entries.sort(key=lambda e: (-e.prefilter_score, e.path.name))
        return entries

    def _probe_counts(
        self, entries: Sequence[_FileEvidence], terms: Sequence[str]
    ) -> list[int]:
        """Read a short row prefix of each file to catch signal that is only in values.

        The prefilter ranks on filenames, schemas, and catalog metadata, so a
        table that mentions a term *only* in its data would otherwise be cut
        before it is ever read. One bounded prefix is cheap enough to ask of
        files just past the cut -- and, in the values-only modality, of every
        file in the lake, since there no other signal may decide.
        """
        return probe_many(
            [entry.path for entry in entries],
            terms,
            max_columns=self.config.grep_max_columns_per_file,
            max_rows=min(
                self.config.grep_probe_rows_per_file,
                self.config.grep_max_scan_rows_per_file,
            ),
            workers=self.config.scan_workers,
            literal=self.values_only,
        )

    def _select(
        self, ranked: list[_FileEvidence], terms: Sequence[str]
    ) -> list[_FileEvidence]:
        """Choose the files that earn a full scan.

        By default there is no choosing: a content-first retriever whose
        candidates are picked by *metadata* would be gated by the very signal it
        exists to bypass, and a cut also changes the IDF the survivors are scored
        with, so a table's score would depend on which other files happened to be
        read. Parallel scanning made the exhaustive path affordable, so it is the
        default and ``grep_max_files`` is an explicit, flagged approximation.
        """
        if self.config.grep_max_files is None:
            return ranked
        if self.values_only:
            return self._select_on_values(ranked, terms)
        max_files = self.config.grep_max_files
        selected = ranked[:max_files]
        # Value-only matches earn a place over candidates the prefilter could not
        # tell apart, while the expensive phase still scans at most max_files.
        window = ranked[max_files : max_files + self.config.grep_probe_files]
        promoted = [
            entry
            for entry, hits in zip(window, self._probe_counts(window, terms))
            if hits
        ]
        if not promoted:
            return selected
        pool = [(entry, False) for entry in selected]
        pool.extend((entry, True) for entry in promoted)
        pool.sort(key=lambda item: (-int(item[1]), -item[0].prefilter_score,
                                    item[0].path.name))
        return [entry for entry, _ in pool[:max_files]]

    def _select_on_values(
        self, ranked: list[_FileEvidence], terms: Sequence[str]
    ) -> list[_FileEvidence]:
        """Pick the files to scan from their cells alone.

        The metadata prefilter is exactly what this modality may not consult, so
        a lake larger than ``grep_max_files`` has no free signal left to order
        it by and a plain cut would keep whichever files happen to sort first by
        name. The bounded prefix read becomes the prefilter instead: every file
        is probed, and the files whose cells actually match are the ones scanned
        in full, most matches first.

        A lake that already fits under the cap skips the probe and is scanned
        whole -- cheaper, and exact where the probe is only a prefix. Past the
        cap the probe is what bounds recall: a match that falls after
        ``grep_probe_rows_per_file`` rows of a table is not seen, so that knob
        trades scan cost for reach into long tables.
        """
        max_files = self.config.grep_max_files
        assert max_files is not None  # only reached on the explicit capped path
        if len(ranked) <= max_files:
            return ranked
        probed = zip(self._probe_counts(ranked, terms), ranked)
        matched = sorted(
            ((hits, entry) for hits, entry in probed if hits),
            key=lambda item: (-item[0], item[1].path.name),
        )
        return [entry for _, entry in matched[:max_files]]

    def _samples(
        self, entries: Sequence[_FileEvidence], terms: Sequence[str]
    ) -> list[list[dict[str, Any]]]:
        """A few matching rows for each hit, from columns already known to match."""
        samples: list[list[dict[str, Any]]] = [[] for _ in entries]
        if self.config.grep_sample_rows <= 0:
            return samples
        limit = self.config.grep_max_columns_per_file
        wanted = [
            (position, columns)
            for position, entry in enumerate(entries)
            if (columns := entry.matched_columns()[:limit])
        ]
        if not wanted:
            return samples
        found = sample_many(
            [(entries[position].path, columns) for position, columns in wanted],
            terms,
            max_rows=self.config.grep_max_scan_rows_per_file,
            limit=self.config.grep_sample_rows,
            workers=self.config.scan_workers,
            literal=self.values_only,
        )
        for (position, _columns), rows in zip(wanted, found):
            samples[position] = rows
        return samples

    def retrieve(
        self, question: str, keywords: Sequence[str], *, top_k: int
    ) -> list[RetrievalHit]:
        if self.values_only:
            # Exactly the values the model listed, each matched whole. Nothing is
            # taken from the question, so a word like "total" cannot match cells.
            primary_terms, secondary_terms = _search_values(keywords), []
        else:
            primary_terms, secondary_terms = _query_terms(question, keywords)
        terms = [*primary_terms, *secondary_terms]
        if not terms:
            return []

        lake = self._candidates(terms)
        candidates = self._select(lake, terms)
        if self.value_weight:
            results = scan_many(
                [entry.path for entry in candidates],
                terms,
                max_columns=self.config.grep_max_columns_per_file,
                max_rows=self.config.grep_max_scan_rows_per_file,
                workers=self.config.scan_workers,
                literal=self.values_only,
            )
            for entry, result in zip(candidates, results):
                if result is None:
                    entry.unscannable = True
                    continue
                counts, scanned, uncastable = result
                entry.term_columns.update(counts)
                entry.scanned_columns = scanned
                entry.uncastable_columns = uncastable
        matched = [
            entry for entry in candidates if any(entry.present(term) for term in terms)
        ]
        if not matched:
            return []

        # IDF is computed per term across the evaluated files, which is the only
        # way it can discriminate: for a single pattern it is one constant and
        # orders nothing.
        population = len(matched)
        document_frequency = {
            term: sum(entry.present(term) for entry in matched) for term in terms
        }

        for entry in matched:
            score = 0.0
            primary_covered = 0
            for term in terms:
                if not entry.present(term):
                    continue
                if term in primary_terms:
                    primary_covered += 1
                    weight = 1.0
                else:
                    weight = 0.25
                idf = math.log((population + 1) / (document_frequency[term] + 1)) + 1.0
                field_score = self.metadata_weight * (
                    (8.0 if term in entry.filename_terms else 0.0)
                    + (5.0 if term in entry.schema_terms else 0.0)
                    + entry.metadata_scores.get(term, 0.0)
                )
                count = entry.term_count(term)
                if count:
                    # Density, not raw count: a 50k-row table with 50 matches
                    # must not outrank a 5-row table that is entirely on topic.
                    density = count / max(1, entry.rows)
                    field_score += self.value_weight * (
                        1.0 + 2.0 * math.sqrt(density)
                    )
                score += weight * idf * field_score
            coverage = primary_covered / max(1, len(primary_terms))
            score += 30.0 * coverage**3
            score -= 6.0 * (len(primary_terms) - primary_covered)
            entry.score = score

        matched.sort(key=lambda entry: (-entry.score, entry.path.name))

        top = matched[:top_k]
        # One batch of sample reads for all hits, rather than a worker call each.
        samples = self._samples(top, terms)
        unscannable = sum(entry.unscannable for entry in candidates)
        hits: list[RetrievalHit] = []
        for entry, entry_samples in zip(top, samples):
            value_terms = sorted(entry.term_columns)
            document = {
                "resource_id": entry.path.name,
                "dataset_id": entry.path.stem,
                "title": entry.title or entry.path.stem.replace("_", " "),
                "description": (
                    ((entry.description + " ") if entry.description else "")
                    + f"Grep evidence: {len(value_terms)} of {len(terms)} terms "
                    f"matched in values across {len(entry.matched_columns())} "
                    f"columns; {entry.rows} total rows."
                ),
                "columns": [
                    {"name": name, "type": dtype} for name, dtype in entry.columns
                ],
                "tags": list(entry.tags),
                "grep_evidence": {
                    "values_only": self.values_only,
                    "terms": terms,
                    "primary_terms": primary_terms,
                    "secondary_terms": secondary_terms,
                    "term_counts": {
                        term: entry.term_count(term) for term in terms
                    },
                    "term_columns": {
                        term: dict(columns)
                        for term, columns in entry.term_columns.items()
                    },
                    "matched_columns": entry.matched_columns(),
                    "filename_terms": sorted(entry.filename_terms),
                    "schema_terms": sorted(entry.schema_terms),
                    "metadata_terms": sorted(entry.metadata_terms),
                    "match_count": sum(entry.term_count(term) for term in terms),
                    "primary_coverage": sum(
                        entry.present(term) for term in primary_terms
                    ) / max(1, len(primary_terms)),
                    "scanned_columns": list(entry.scanned_columns),
                    "uncastable_columns": entry.uncastable_columns,
                    "value_weight": self.value_weight,
                    "metadata_weight": self.metadata_weight,
                    "files_in_lake": len(lake),
                    "files_scanned": len(candidates),
                    "truncated": len(candidates) < len(lake),
                    "files_unscannable": unscannable,
                    "rows_scanned": min(
                        entry.rows, self.config.grep_max_scan_rows_per_file
                    ),
                    "total_rows": entry.rows,
                    "samples": entry_samples,
                },
            }
            hits.append(RetrievalHit(document=document, score=entry.score))

        for rank, hit in enumerate(hits, 1):
            hit.rank = rank
            hit.lexical_rank = rank
            hit.lexical_score = hit.score
        return hits
