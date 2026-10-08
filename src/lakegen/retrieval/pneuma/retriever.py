"""The Pneuma retriever: Pneuma's ranking, and with the Seeker mode the content search and enumeration fused in."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from lakegen.retrieval.config import (
    PneumaContentFusion,
    RetrievalConfig,
    RetrievalMode,
)
from lakegen.retrieval.duckdb_agentic import count_matching_rows
from lakegen.retrieval.models import RetrievalHit, document_key, min_max_normalize
from lakegen.retrieval.pneuma import entities, seeker
from lakegen.retrieval.pneuma.client import PneumaClient, _table_ids
from lakegen.retrieval.pneuma.entities import EntityExtractor
from lakegen.retrieval.pneuma.enumeration import (
    _ENUMERATION_DECAY,
    _ENUMERATION_MAX_SIBLINGS,
    _enumeration_pattern,
)
from lakegen.retrieval.pneuma.resolver import DocumentResolver
from lakegen.retrieval.pneuma.seeker import (
    ScanResult,
    _entity_pattern,
    _scan_terms,
)


@dataclass
class _FileFacts:
    """What a Parquet footer tells us, before any data page is read."""

    path: Path
    stem: str
    columns: tuple[str, ...]
    # The String columns: the only ones the reference searches.
    text_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Content:
    """One file's content-search evidence."""

    score: float
    term_cells: dict[str, int]
    column_name_terms: tuple[str, ...]
    title_terms: tuple[str, ...]


@dataclass
class _Candidate:
    key: str
    document: dict[str, Any]
    path: Path | None = None
    pneuma_rank: int | None = None
    content_score: float | None = None
    content_rank: int | None = None
    normalized_content: float = 0.0
    normalized_pneuma: float = 0.0
    term_cells: dict[str, int] = field(default_factory=dict)
    column_name_terms: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    enumerated_from: str | None = None
    score: float = 0.0


class PneumaRetriever:
    """Use Pneuma's original ranking while returning LakeGen RetrievalHit values.

    ``RetrievalMode.PNEUMA_SEEKER`` selects the augmented modality described in
    the module docstring. The mode is the single source of truth, so a config
    and a retriever cannot disagree about which one is running, and the plain
    ``pneuma`` path stays byte-for-byte what it was.

    Note the slot mapping on the returned hits: the content branch fills
    ``lexical_*`` and the Pneuma branch fills ``semantic_*``. Neither name is
    literally right, but reusing them means the existing run log reports both
    contributions with no schema change; ``pneuma_evidence`` on the document
    carries the same numbers under honest names.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        resolver: DocumentResolver,
        *,
        client: PneumaClient | None = None,
        table_dir: str | Path | None = None,
        entity_extractor: EntityExtractor | None = None,
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.client = client or client.create_pneuma_client(config)
        self.content_search = config.mode is RetrievalMode.PNEUMA_SEEKER
        # Pneuma-Seeker only: entities for callers that pass none (see
        # pneuma_entity_model).
        self.entity_extractor: EntityExtractor | None = None
        if self.content_search and config.pneuma_entity_model:
            self.entity_extractor = entity_extractor or entities.entity_extractor_for(
                config.pneuma_entity_model
            )
        # Plain pneuma's optional value scan of its own pool (see
        # pneuma_value_scan_weight); weight 0 keeps the plain path untouched.
        self.value_scan = (
            config.mode is RetrievalMode.PNEUMA and config.pneuma_value_scan_weight > 0
        )
        self.table_dir: Path | None = None
        if self.value_scan and table_dir is not None:
            self.table_dir = Path(table_dir).resolve()
        if self.content_search:
            if table_dir is None:
                raise ValueError("pneuma_seeker retrieval requires a local table_dir")
            self.table_dir = Path(table_dir).resolve()
            if not self.table_dir.is_dir():
                raise ValueError(
                    f"pneuma_seeker table directory does not exist: {table_dir}"
                )

    # ---------------------------------------------------------------- Pneuma

    def _pneuma_candidates(
        self, question: str, limit: int
    ) -> list[tuple[str, dict[str, Any]]]:
        """Query the index and resolve its ordered table IDs to documents."""
        response = self.client.query_index(
            self.config.pneuma_index_name,
            question,
            k=limit,
            n=self.config.candidate_multiplier,
            alpha=self.config.alpha,
        )
        ordered: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for table_id in _table_ids(response):
            document = self.resolver(table_id)
            if document is None:
                continue
            key = document_key(document)
            if key in seen:
                continue
            seen.add(key)
            ordered.append((table_id, document))
            if len(ordered) >= limit:
                break
        return ordered

    # --------------------------------------------------------- content search

    def _lake(self) -> list[_FileFacts]:
        """List the lake from Parquet footers alone -- no data pages are read."""
        assert self.table_dir is not None
        facts: list[_FileFacts] = []
        paths = sorted(
            (*self.table_dir.glob("*.parquet"), *self.table_dir.glob("*.pq"))
        )
        for path in paths:
            try:
                parquet = pq.ParquetFile(path)
                columns = tuple(field_.name for field_ in parquet.schema_arrow)
                text_columns = tuple(
                    field_.name
                    for field_ in parquet.schema_arrow
                    if pa.types.is_string(field_.type)
                    or pa.types.is_large_string(field_.type)
                    or pa.types.is_string_view(field_.type)
                )
            except (OSError, ValueError):
                continue
            facts.append(
                _FileFacts(
                    path=path,
                    stem=path.stem,
                    columns=columns,
                    text_columns=text_columns,
                )
            )
        return facts

    def _scan_pool(
        self,
        lake: Sequence[_FileFacts],
        terms: Sequence[str],
        anchors: set[str],
        titles: dict[str, str],
    ) -> list[_FileFacts]:
        """Choose which files earn the expensive cell scan, under the file cap.

        Anchors come first: a table Pneuma returned must be scanned, or it would
        score zero on content purely because it was never looked at, and a
        content-only table would outrank it for no reason. Then files whose free
        signals -- catalog title or column names -- already match an entity,
        which the catalog and the footers have told us for nothing. Only if the budget is still open
        does the bounded prefix probe get to promote files whose sole evidence
        is in their cells.

        None of that runs by default. Per-keyword normalization divides by the
        maximum over the pool, so a truncated pool moves every table's score --
        the same defect a truncated IDF population causes in the grep modality --
        and the parallel map made scanning the whole lake affordable.
        """
        budget = self.config.grep_max_files
        if budget is None:
            return list(lake)
        chosen: list[_FileFacts] = []
        taken: set[str] = set()

        def take(facts: _FileFacts) -> None:
            if facts.stem.casefold() not in taken and len(chosen) < budget:
                taken.add(facts.stem.casefold())
                chosen.append(facts)

        for facts in lake:
            if facts.stem.casefold() in anchors:
                take(facts)
        patterns = [re.compile(_entity_pattern(term)) for term in terms]
        named = [
            facts
            for facts in lake
            if facts.stem.casefold() not in taken
            and any(
                pattern.search(titles.get(facts.stem, ""))
                or any(pattern.search(name) for name in facts.columns)
                for pattern in patterns
            )
        ]
        for facts in named:
            take(facts)
        if len(chosen) >= budget:
            return chosen

        remaining = [facts for facts in lake if facts.stem.casefold() not in taken]
        # One batch of probes in worker processes, then the same in-order take.
        # A probe is a prefix by nature, so it keeps its row budget; like the
        # scan, it reads every column.
        counts = seeker.probe_many(
            [facts.path for facts in remaining],
            terms,
            max_columns=None,
            max_rows=self.config.grep_probe_rows_per_file,
            workers=self.config.scan_workers,
            pattern=_entity_pattern,
        )
        for facts, hits in zip(remaining, counts):
            if len(chosen) >= budget:
                break
            if hits:
                take(facts)
        return chosen

    def _content_scores(
        self,
        pool: Sequence[_FileFacts],
        terms: Sequence[str],
        titles: dict[str, str],
    ) -> tuple[dict[str, _Content], int]:
        """Score each scanned file by stem, as the reference implementation does.

        Also returns how many files the scanner could not read at all.

        Per entity, a match in the table's catalog title (once per text column),
        its matching text-column names, and its occurrences in text cells are
        weighted by ``pneuma_table_name_weight``, ``pneuma_column_name_weight``
        and ``pneuma_cell_weight`` and summed, and ``log1p`` damps the sum -- the
        paper's ``log(1 + tf)``: a 100k-row table mentioning a year 100k times
        must not swamp a small table that is exactly on topic. Each entity is
        then divided by its maximum over the pool, which stops one
        high-frequency entity from deciding the ranking on its own. A table's
        content score is the mean over every entity, zeros included, times the
        fraction of entities it matched at all, so a table matching every entity
        beats one matching a single entity many times.

        Every row of every text column is read: the grep modality's per-file
        caps do not apply here.
        """
        table_weight = self.config.pneuma_table_name_weight
        column_weight = self.config.pneuma_column_name_weight
        cell_weight = self.config.pneuma_cell_weight
        patterns = {term: re.compile(_entity_pattern(term)) for term in terms}
        # A family weighted 0 cannot change any ranking, so its work is skipped:
        # with no weight on cells, no cell is read.
        scanned: list[ScanResult | None] = (
            seeker.scan_many(
                [facts.path for facts in pool],
                terms,
                max_columns=None,
                max_rows=None,
                workers=self.config.scan_workers,
                pattern=_entity_pattern,
            )
            if cell_weight
            else [({}, (), 0)] * len(pool)
        )
        damped: dict[str, dict[str, float]] = {}
        evidence: dict[str, tuple[dict[str, int], tuple[str, ...], tuple[str, ...]]] = {}
        unscannable = 0
        for facts, result in zip(pool, scanned):
            # A file that crashed the scanner twice is skipped; its title and
            # column names, known without reading a data page, still count.
            unscannable += result is None
            counts = {} if result is None else result[0]
            title = titles.get(facts.stem, "")
            per_term: dict[str, float] = {}
            cells: dict[str, int] = {}
            named: list[str] = []
            titled: list[str] = []
            for term in terms:
                cell_hits = sum(counts.get(term, {}).values())
                name_hits = (
                    sum(
                        1
                        for name in facts.text_columns
                        if patterns[term].search(name)
                    )
                    if column_weight
                    else 0
                )
                # The reference tests the table name once per text column.
                title_hit = (
                    len(facts.text_columns)
                    if table_weight and title and patterns[term].search(title)
                    else 0
                )
                if cell_hits:
                    cells[term] = cell_hits
                if name_hits:
                    named.append(term)
                if title_hit:
                    titled.append(term)
                per_term[term] = math.log1p(
                    table_weight * title_hit
                    + column_weight * name_hits
                    + cell_weight * cell_hits
                )
            damped[facts.stem] = per_term
            evidence[facts.stem] = (cells, tuple(named), tuple(titled))

        peaks = {
            term: max((values[term] for values in damped.values()), default=0.0)
            for term in terms
        }
        scores: dict[str, _Content] = {}
        for stem, values in damped.items():
            normalized = [
                values[term] / peaks[term] if peaks[term] else 0.0 for term in terms
            ]
            if terms:
                mean = sum(normalized) / len(terms)
                coverage = sum(1 for value in normalized if value > 0) / len(terms)
            else:
                mean = coverage = 0.0
            cells, named, titled = evidence[stem]
            scores[stem] = _Content(mean * coverage, cells, named, titled)
        return scores, unscannable

    # ------------------------------------------------------------ value scan

    def _table_path(self, table_id: str) -> Path | None:
        """The local file behind a Pneuma table ID (a path in this index)."""
        path = Path(table_id)
        if path.is_file():
            return path
        if self.table_dir is not None and (self.table_dir / path.name).is_file():
            return self.table_dir / path.name
        return None

    def _value_scanned(
        self, ordered: Sequence[tuple[str, dict[str, Any]]], terms: Sequence[str]
    ) -> list[RetrievalHit]:
        """Pneuma's own tables, reranked with a DuckDB scan of their values.

        Pneuma-Seeker's content search restricted to the pool Pneuma returned,
        so Pneuma's cost is unchanged. Each term is matched word-bounded, as the
        Seeker does, in every column cast to text; rows are counted within the
        duckdb_* bounds. Per term the count is damped by ``log1p`` and divided
        by its maximum over the pool; a table's value score is the mean over the
        terms times the fraction it matched, and is fused with Pneuma's rank
        score (1/rank, min-max over the pool) by ``pneuma_value_scan_weight``.
        A table whose file cannot be found or read scores 0 on values.
        """
        patterns = [_entity_pattern(term) for term in terms]
        started = time.monotonic()
        rows: list[list[int] | None] = []
        for table_id, _document in ordered:
            path = self._table_path(table_id)
            rows.append(
                count_matching_rows(
                    path,
                    patterns,
                    max_rows=self.config.duckdb_max_scan_rows_per_file,
                    max_columns=self.config.duckdb_max_columns_per_file,
                )
                if path is not None
                else None
            )
        scan_seconds = time.monotonic() - started
        damped = [
            [math.log1p(count) for count in counts] if counts else [0.0] * len(terms)
            for counts in rows
        ]
        peaks = [max((values[i] for values in damped), default=0.0) for i in range(len(terms))]
        value_scores = []
        for values in damped:
            normalized = [value / peak if peak else 0.0 for value, peak in zip(values, peaks)]
            mean = sum(normalized) / len(terms)
            coverage = sum(1 for value in normalized if value > 0) / len(terms)
            value_scores.append(mean * coverage)
        pneuma_scores = min_max_normalize(
            {index: 1.0 / (index + 1) for index in range(len(ordered))}
        )
        weight = self.config.pneuma_value_scan_weight
        fused = [
            weight * value_scores[index] + (1.0 - weight) * pneuma_scores.get(index, 0.0)
            for index in range(len(ordered))
        ]
        value_rank = {
            index: rank
            for rank, index in enumerate(
                sorted(
                    (index for index in range(len(ordered)) if value_scores[index]),
                    key=lambda index: (-value_scores[index], index),
                ),
                1,
            )
        }
        hits: list[RetrievalHit] = []
        for rank, index in enumerate(
            sorted(range(len(ordered)), key=lambda index: (-fused[index], index)), 1
        ):
            table_id, source_document = ordered[index]
            # The resolver hands out shared, cached dicts.
            document = dict(source_document)
            counts = rows[index]
            term_rows = {
                term: count for term, count in zip(terms, counts or []) if count
            }
            if term_rows:
                description = str(document.get("description") or "").strip()
                document["description"] = (
                    (description + " " if description else "")
                    + "Contains the searched values: "
                    + ", ".join(term_rows)
                    + "."
                )
            document["pneuma_evidence"] = {
                "value_scan_terms": list(terms),
                "term_rows": term_rows,
                "value_score": value_scores[index],
                "normalized_pneuma_score": pneuma_scores.get(index, 0.0),
                "value_scan_weight": weight,
                "pneuma_rank": index + 1,
                "scanned": counts is not None,
                "pool_scan_seconds": round(scan_seconds, 6),
            }
            hits.append(
                RetrievalHit(
                    document=document,
                    score=fused[index],
                    rank=rank,
                    lexical_score=value_scores[index],
                    semantic_score=1.0 / (index + 1),
                    normalized_lexical_score=value_scores[index],
                    normalized_semantic_score=pneuma_scores.get(index, 0.0),
                    lexical_rank=value_rank.get(index),
                    semantic_rank=index + 1,
                )
            )
        return hits

    # --------------------------------------------------------------- assembly

    def _enumerate(
        self, ranked: Sequence[_Candidate], lake: Sequence[_FileFacts], top_k: int
    ) -> list[_Candidate]:
        """Pull in the identifier siblings of the strongest tables."""
        by_stem = {facts.stem: facts for facts in lake}
        present = {
            candidate.path.stem for candidate in ranked if candidate.path is not None
        }
        added: list[_Candidate] = []
        for seed in list(ranked)[:top_k]:
            if seed.path is None:
                continue
            pattern = _enumeration_pattern(seed.path.stem)
            if pattern is None:
                continue
            siblings = [
                facts
                for stem, facts in by_stem.items()
                if stem != seed.path.stem
                and re.fullmatch(pattern, stem, re.IGNORECASE)
            ]
            # A pattern matching half the lake describes a naming convention,
            # not a series, so it is dropped rather than trusted.
            if not siblings or len(siblings) > _ENUMERATION_MAX_SIBLINGS:
                continue
            for facts in sorted(siblings, key=lambda item: item.stem):
                if facts.stem in present:
                    continue
                document = self.resolver(str(facts.path))
                if document is None:
                    continue
                present.add(facts.stem)
                added.append(
                    _Candidate(
                        key=document_key(document),
                        document=document,
                        path=facts.path,
                        enumerated_from=seed.path.stem,
                        score=_ENUMERATION_DECAY * seed.score,
                    )
                )
        return added

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        entities: Sequence[str] | None = None,
        keywords: Sequence[str] | None = None,
    ) -> list[RetrievalHit]:
        if not self.content_search:
            ordered = self._pneuma_candidates(question, top_k)
            terms = _scan_terms(entities or keywords) if self.value_scan else []
            if terms:
                return self._value_scanned(ordered, terms)
            hits: list[RetrievalHit] = []
            for rank, (_table_id, document) in enumerate(ordered, 1):
                # Pneuma 0.0.4 exposes ordered table IDs but not final scores.
                hits.append(
                    RetrievalHit(document=document, score=1.0 / rank, rank=rank)
                )
            return hits

        fill = self.config.pneuma_content_fusion is PneumaContentFusion.FILL
        extraction: dict[str, Any] | None = None
        if entities is None and self.entity_extractor is not None:
            started = time.monotonic()
            error = None
            try:
                entities = self.entity_extractor.entities(question)
            except Exception as exc:
                # No entities means no content search, which is Pneuma's own
                # ranking: a degraded answer, not a failed search.
                entities, error = [], f"{type(exc).__name__}: {exc}"
            extraction = {
                "model": self.config.pneuma_entity_model,
                "entities": list(entities),
                "seconds": time.monotonic() - started,
                "error": error,
            }
        # The weighted sum reranks a wider Pneuma pool than the caller asked
        # for, so a table at Pneuma rank 30 can still surface. The reference
        # fill asks Pneuma for exactly top_k, as its code does.
        limit = (
            top_k if fill else max(top_k, top_k * self.config.candidate_multiplier)
        )
        ordered = self._pneuma_candidates(question, limit)
        terms = _scan_terms(entities)

        candidates: dict[str, _Candidate] = {}
        for rank, (table_id, document) in enumerate(ordered, 1):
            key = document_key(document)
            candidates[key] = _Candidate(
                key=key,
                document=document,
                path=Path(table_id) if table_id else None,
                pneuma_rank=rank,
            )

        # With no entity there is nothing to search for, and the reference
        # implementation leaves Pneuma's ranking untouched rather than scanning
        # for words taken from the question. Content search weighted 0 cannot
        # change any ranking either, so in both cases the scan is skipped rather
        # than computed and multiplied away -- which also makes weight 0 exactly
        # equal to plain ``pneuma``, as a boundary should be. The footer listing
        # survives only because enumeration, which needs no entity, still uses it.
        weight = self.config.pneuma_content_weight if terms else 0.0
        # Fill weights nothing, so any entity is enough to search for.
        scans_content = bool(terms) if fill else weight > 0.0
        lake = (
            self._lake()
            if scans_content or self.config.pneuma_enumerate_tables
            else []
        )
        content_by_stem: dict[str, _Content] = {}
        unscannable = 0
        if scans_content and lake:
            # A table's name is the title its catalog document carries: the lakes
            # here name files by opaque id, and the resolver's lookup of a path
            # is deterministic, so the same file always yields the same title.
            documents = {facts.stem: self.resolver(str(facts.path)) for facts in lake}
            titles = {
                stem: str(document.get("title") or "")
                for stem, document in documents.items()
                if document is not None
            }
            anchors = {
                candidate.path.stem.casefold()
                for candidate in candidates.values()
                if candidate.path is not None
            }
            pool = self._scan_pool(lake, terms, anchors, titles)
            content_by_stem, unscannable = self._content_scores(pool, terms, titles)
            for facts in pool:
                content = content_by_stem[facts.stem]
                document = documents[facts.stem]
                if document is None:
                    continue
                key = document_key(document)
                candidate = candidates.get(key)
                if candidate is None:
                    # Now that the whole lake is scanned, most files match
                    # nothing. A table neither branch found evidence for is not a
                    # candidate; admitting it would bury the ranking in zeros and
                    # make every file look already-retrieved to the enumerator.
                    if not content.score:
                        continue
                    candidate = _Candidate(key=key, document=document, path=facts.path)
                    candidates[key] = candidate
                candidate.path = candidate.path or facts.path
                candidate.content_score = content.score
                candidate.term_cells = content.term_cells
                candidate.column_name_terms = content.column_name_terms
                candidate.title_terms = content.title_terms

        normalized_pneuma = min_max_normalize(
            {
                key: 1.0 / candidate.pneuma_rank
                for key, candidate in candidates.items()
                if candidate.pneuma_rank
            }
        )
        # Only the Pneuma branch is min-max normalized. The content score is
        # already on a 0..1 scale by the paper's own recipe -- damped, then
        # normalized per keyword -- and re-normalizing it across the candidates
        # would undo exactly that: min-max forces the weakest table to 0.0 and
        # the strongest to 1.0, discarding the ratio the damping computed.
        content_order = sorted(
            (
                key
                for key, candidate in candidates.items()
                if candidate.content_score
            ),
            key=lambda key: (-(candidates[key].content_score or 0.0), key),
        )
        for rank, key in enumerate(content_order, 1):
            candidates[key].content_rank = rank

        for key, candidate in candidates.items():
            # A branch that did not see this table contributes nothing, which is
            # the zero missing-signal policy the hybrid retriever also uses.
            candidate.normalized_content = candidate.content_score or 0.0
            candidate.normalized_pneuma = normalized_pneuma.get(key, 0.0)
            candidate.score = (
                weight * candidate.normalized_content
                + (1.0 - weight) * candidate.normalized_pneuma
            )

        if fill:
            # The reference merge: Pneuma's tables in Pneuma's order, then the
            # content matches it did not return, strongest first -- only
            # tables that matched an entity, where the reference would pad with
            # zero-score tables in name order. Enumerated siblings come last.
            ranked = [
                *sorted(
                    (c for c in candidates.values() if c.pneuma_rank),
                    key=lambda c: c.pneuma_rank or 0,
                ),
                *(candidates[key] for key in content_order
                  if not candidates[key].pneuma_rank),
            ]
            if self.config.pneuma_enumerate_tables and lake:
                ranked += self._enumerate(ranked, lake, top_k)
            for position, candidate in enumerate(ranked, 1):
                candidate.score = 1.0 / position
        else:
            no_rank = limit + 1

            def order(candidate: _Candidate) -> tuple[Any, ...]:
                return (
                    -candidate.score,
                    candidate.enumerated_from is not None,
                    candidate.pneuma_rank or no_rank,
                    candidate.content_rank or no_rank,
                    candidate.key,
                )

            ranked = sorted(candidates.values(), key=order)
            if self.config.pneuma_enumerate_tables and lake:
                siblings = self._enumerate(ranked, lake, top_k)
                if siblings:
                    ranked = sorted([*ranked, *siblings], key=order)

        hits = []
        for rank, candidate in enumerate(ranked[:top_k], 1):
            # The resolver hands out shared, cached dicts; evidence from one
            # query must not leak into the next.
            document = dict(candidate.document)
            found = [
                term
                for term in terms
                if candidate.term_cells.get(term)
                or term in candidate.column_name_terms
                or term in candidate.title_terms
            ]
            if found:
                # The reference implementation tells the agent which entities a
                # table contains, and the description is what the agent reads.
                description = str(document.get("description") or "").strip()
                document["description"] = (
                    (description + " " if description else "")
                    + "Contains the searched entities: "
                    + ", ".join(found)
                    + "."
                )
            document["pneuma_evidence"] = {
                "scan_terms": list(terms),
                "entity_extraction": extraction,
                "matched_terms": found,
                "content_score": candidate.content_score,
                "normalized_content_score": candidate.normalized_content,
                "normalized_pneuma_score": candidate.normalized_pneuma,
                "content_fusion": self.config.pneuma_content_fusion.value,
                "content_weight": None if fill else weight,
                "table_name_weight": self.config.pneuma_table_name_weight,
                "column_name_weight": self.config.pneuma_column_name_weight,
                "cell_weight": self.config.pneuma_cell_weight,
                "pneuma_rank": candidate.pneuma_rank,
                "term_cells": candidate.term_cells,
                "column_name_terms": list(candidate.column_name_terms),
                "title_terms": list(candidate.title_terms),
                "enumerated_from": candidate.enumerated_from,
                "scanned_files": len(content_by_stem),
                "listed_files": len(lake),
                "truncated": bool(content_by_stem) and len(content_by_stem) < len(lake),
                "unscannable_files": unscannable,
            }
            hits.append(
                RetrievalHit(
                    document=document,
                    score=candidate.score,
                    rank=rank,
                    lexical_score=candidate.content_score,
                    semantic_score=(
                        1.0 / candidate.pneuma_rank if candidate.pneuma_rank else None
                    ),
                    normalized_lexical_score=candidate.normalized_content,
                    normalized_semantic_score=candidate.normalized_pneuma,
                    lexical_rank=candidate.content_rank,
                    semantic_rank=candidate.pneuma_rank,
                )
            )
        return hits
