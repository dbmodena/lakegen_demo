"""Optional adapter for the original MIT-licensed Pneuma implementation.

Two modalities share this module, selected by :class:`RetrievalMode`:

``pneuma``
    The original system: one query against the external Pneuma index, whose
    ordered table IDs become hits.
``pneuma_seeker``
    Pneuma augmented as in *Pneuma-Seeker* (§5.3) -- "Pneuma + content search +
    table enumeration":

    * **content search** (after Octopus): entities likely to appear verbatim in
      tables are scanned for across the local Parquet with case-insensitive
      regexes, scoring matches in column names and cell values; raw counts are
      damped by ``log(1 + tf)`` and normalized per keyword, and the result is
      fused with Pneuma's own ranking;
    * **table enumeration**: once a table is retrieved, siblings whose
      identifiers match the same pattern (``water_body_testing_2020`` ->
      ``water_body_testing_\\d{4}``) are pulled in, which is what top-k
      retrieval structurally cannot do when a question needs a whole family of
      identically-schema'd tables.

Where the paper is silent, the content search follows its reference
implementation (``TheDataStation/pneuma-seeker``, branch
``task-improve-answer-quality``, ``PneumaRetriever.__keyword_relevance_by_table``):

* an entity becomes a word-bounded pattern (:func:`_entity_pattern`), not a
  substring, so ``art`` does not match ``Department``;
* per entity, a table's raw score is ``pneuma_table_name_weight`` x a match in
  its name + ``pneuma_column_name_weight`` x matching column names +
  ``pneuma_cell_weight`` x matching cells, damped as a whole;
* a table's content score is its mean normalized score over the entities times
  the fraction of entities it matched;
* a question that names no entity gets no content search: the ranking stays
  Pneuma's own rather than one guessed from the question's words;
* the entities a table matched are shown to the agent, in its description.

Deviations, all visible in the results rather than hidden:

* **Pneuma exposes no scores.** Its ``/query`` response is an ordered list of
  table IDs (see :func:`_table_ids`), so the paper's "combine with Pneuma's
  retrieval scores" is necessarily a combination with a *rank-derived* score,
  ``1/rank``. A confident first place and a marginal one are indistinguishable
  here, so ``pneuma_content_weight`` has to be swept rather than transplanted
  from the paper.
* **Fusion is a weighted sum.** The paper combines the two scores; the reference
  code instead keeps Pneuma's top-k and lets content-ranked tables fill only the
  slots left over. The weighted sum is the paper's reading, and it lets a table
  Pneuma ranked 30th surface on content alone.
* **Cells, not occurrences.** The reference code counts every regex occurrence
  in text columns; the shared scanner counts matching cells in every column
  cast to text. Both read every row: the grep modality's per-file row and
  column caps do not apply here.
* **A table's name is its catalog title.** The reference code matches the
  DuckDB table name, and counts that match once per text column. The lakes here
  name files by opaque id (``43nn-pn8j``), so the title the resolver returns for
  the file is matched instead, once per table.
* **Enumeration is an agent action in the paper**, invoked on demand by the
  Conductor. Retrieval here is a single call, so enumeration runs automatically
  from the fused ranking instead.

The entities are not extracted by a new LLM call. Under this mode the discovery
prompts ask for them with the reference code's extraction rules -- specific,
named, canonical strings written as the question writes them, never general
concepts -- in ``RetrievalIntent.entities`` or the search tool's ``entities``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol

import pyarrow.parquet as pq
import requests

from lakegen.retrieval.config import RetrievalConfig, RetrievalMode
from lakegen.retrieval.models import RetrievalHit, document_key, min_max_normalize

# The same scanner the grep modality uses: every column cast to text, and one
# column in flight, read a chunk of rows at a time. Content search lifts its row
# and column caps, and builds the pattern from an entity's letters and digits.
from lakegen.retrieval.grep import ScanResult, probe_many, scan_many


class PneumaClient(Protocol):
    def query_index(
        self, index_name: str, queries: str, *, k: int, n: int, alpha: float
    ) -> str | dict[str, Any]: ...


DocumentResolver = Callable[[str], dict[str, Any] | None]


# More entities than a question plausibly names; each one is another regex per
# scanned column.
_MAX_SCAN_TERMS = 12

_NON_ALNUM_RUN = re.compile(r"[^A-Za-z0-9]+")
_DIGIT_RUN = re.compile(r"\d+")
_ALPHA_RUN = re.compile(r"[^\W\d_]+", re.UNICODE)
# An identifier with no real word in it is an opaque id, not a series name.
_ENUMERATION_MIN_WORD = 3
# More siblings than this is a filename convention, not a table family.
_ENUMERATION_MAX_SIBLINGS = 25
# A sibling carries no evidence of its own, so it must stay below its seed.
_ENUMERATION_DECAY = 0.9


def create_pneuma_client(config: RetrievalConfig) -> PneumaClient:
    """Create a lightweight client for the independently hosted backend."""
    return HttpPneumaClient(
        config.pneuma_base_url, timeout=config.pneuma_timeout_seconds
    )


class HttpPneumaClient:
    def __init__(self, base_url: str, *, timeout: float) -> None:
        self.query_url = base_url.rstrip("/") + "/query"
        self.timeout = timeout

    def query_index(
        self, index_name: str, queries: str, *, k: int, n: int, alpha: float
    ) -> dict[str, Any]:
        try:
            response = requests.post(
                self.query_url,
                json={
                    "index_name": index_name,
                    "query": queries,
                    "k": k,
                    "n": n,
                    "alpha": alpha,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"Pneuma service unavailable at {self.query_url}: {exc}"
            ) from exc


def _table_ids(payload: str | dict[str, Any]) -> list[str]:
    try:
        decoded = json.loads(payload) if isinstance(payload, str) else payload
    except json.JSONDecodeError as exc:
        raise RuntimeError("Pneuma returned invalid JSON") from exc
    if not isinstance(decoded, dict) or decoded.get("status") != "SUCCESS":
        message = decoded.get("message") if isinstance(decoded, dict) else None
        raise RuntimeError(f"Pneuma query failed: {message or 'invalid response'}")
    data = decoded.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise RuntimeError("Pneuma returned an unexpected query response")
    tables = data[0].get("retrieved_tables")
    if not isinstance(tables, list):
        raise RuntimeError("Pneuma response has no retrieved_tables list")
    return [str(value) for value in tables if str(value).strip()]


def _entity_pattern(entity: str) -> str:
    """The reference implementation's pattern for one entity, or ``""`` for none.

    The entity is split on every run of characters other than ASCII letters and
    digits, and the parts are joined by exactly one arbitrary character, so
    ``New York`` also matches ``NEW-YORK`` and ``new_york``. A match must start
    and end at a boundary -- the edge of the text, or a character that is not a
    letter, digit, underscore, or hyphen -- so ``art`` does not match
    ``Department`` and ``2020`` does not match ``120201``. One consequence the
    reference code shares: ``borough`` does not match a column named
    ``borough_name``, because an underscore is not a boundary. Case is ignored.

    Only letters and digits reach the pattern, so nothing an agent supplies can
    act as regex syntax, and an entity with neither (``.*``) has no pattern.
    """
    parts = [re.escape(part) for part in _NON_ALNUM_RUN.split(entity) if part]
    if not parts:
        return ""
    return "(?i)(^|[^A-Za-z0-9_-])" + ".".join(parts) + "($|[^A-Za-z0-9_-])"


def _scan_terms(entities: Sequence[str] | None) -> list[str]:
    """The entities to scan for, as listed, each once.

    Only whitespace is normalized: the pattern ignores case anyway, and the
    entity is shown back to the agent as written. Entities sharing a pattern
    (``New York``, ``new-york``) are one entity, or it would count twice in the
    mean and in the coverage.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for value in entities or ():
        term = " ".join(str(value).split())
        pattern = _entity_pattern(term).casefold()
        if pattern and pattern not in seen:
            seen.add(pattern)
            terms.append(term)
    return terms[:_MAX_SCAN_TERMS]


def _enumeration_pattern(stem: str) -> str | None:
    """Generalize the digit runs of a table identifier into a sibling pattern.

    ``water_body_testing_2020`` becomes ``water_body_testing_\\d{4}``, the
    paper's own example. Everything outside a digit run is regex-escaped, so an
    identifier can never act as a pattern.

    Returns ``None`` for identifiers that carry no real word, which is what
    keeps a lake of opaque Socrata ids out of this: ``43nn-pn8j`` would
    otherwise generalize to a shape matching unrelated datasets. The word test
    runs on the original stem, never on the digit-stripped one -- stripping
    joins ``pn`` and ``j`` into a three-letter "word" and invents a family.
    """
    if not _DIGIT_RUN.search(stem):
        return None
    if not any(len(word) >= _ENUMERATION_MIN_WORD for word in _ALPHA_RUN.findall(stem)):
        return None
    parts: list[str] = []
    index = 0
    for match in _DIGIT_RUN.finditer(stem):
        parts.append(re.escape(stem[index : match.start()]))
        parts.append(f"\\d{{{len(match.group())}}}")
        index = match.end()
    parts.append(re.escape(stem[index:]))
    return "".join(parts)


class SolrPneumaDocumentResolver:
    """Map Pneuma's path-like table IDs back to LakeGen catalog documents."""

    FIELDS = (
        "id", "resource_id", "dataset_id", "title", "description", "tags",
        "columns.name", "columns.description", "columns.type", "schema",
        "dataset_url", "download_url", "url", "permalink", "link", "source",
        "portal", "provenance",
    )

    ALIAS_FIELDS = (
        "id", "resource_id", "dataset_id", "download_url", "url", "source"
    )

    def __init__(self, solr: Any) -> None:
        # Indexed by alias rather than scanned per lookup: content search asks
        # this question once per candidate file, not once per Pneuma hit. The
        # catalog position is kept so a contested alias still resolves to the
        # document a linear scan would have found first -- ``_aliases`` returns a
        # set, whose iteration order is not stable across processes.
        self._index: dict[str, tuple[int, dict[str, Any]]] = {}
        for position, document in enumerate(
            solr.iter_documents(fields=self.FIELDS, sort_field="resource_id")
        ):
            for field_name in self.ALIAS_FIELDS:
                for alias in self._aliases(document.get(field_name)):
                    self._index.setdefault(alias, (position, document))
            # Lakes that group resources under a dataset (UK) name each file
            # ``<dataset_id>___<resource_id>``, a stem no single field holds.
            # Pneuma returns that file's path, so the pair is an alias too.
            file_stem = self._file_stem(document)
            if file_stem:
                self._index.setdefault(file_stem, (position, document))

    @staticmethod
    def _file_stem(document: dict[str, Any]) -> str:
        dataset_id = str(document.get("dataset_id") or "").strip()
        resource_id = str(document.get("resource_id") or "").strip()
        if not dataset_id or not resource_id:
            return ""
        return f"{dataset_id}___{resource_id}".casefold()

    @staticmethod
    def _aliases(value: Any) -> set[str]:
        if value is None or not str(value).strip():
            return set()
        text = str(value).strip()
        path = Path(text)
        return {text.casefold(), path.name.casefold(), path.stem.casefold()}

    def __call__(self, table_id: str) -> dict[str, Any] | None:
        found = [
            self._index[alias]
            for alias in self._aliases(table_id)
            if alias in self._index
        ]
        if not found:
            return None
        return min(found, key=lambda item: item[0])[1]


@dataclass
class _FileFacts:
    """What a Parquet footer tells us, before any data page is read."""

    path: Path
    stem: str
    columns: tuple[str, ...]


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
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.client = client or create_pneuma_client(config)
        self.content_search = config.mode is RetrievalMode.PNEUMA_SEEKER
        self.table_dir: Path | None = None
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
            except (OSError, ValueError):
                continue
            facts.append(_FileFacts(path=path, stem=path.stem, columns=columns))
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
        counts = probe_many(
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

        Per entity, a match in the table's catalog title, its matching column
        names, and its matching cells are weighted by
        ``pneuma_table_name_weight``, ``pneuma_column_name_weight`` and
        ``pneuma_cell_weight`` and summed, and ``log1p`` damps the sum -- the
        paper's ``log(1 + tf)``: a 100k-row table mentioning a year 100k times
        must not swamp a small table that is exactly on topic. Each entity is
        then divided by its maximum over the pool, which stops one
        high-frequency entity from deciding the ranking on its own. A table's
        content score is the mean over every entity, zeros included, times the
        fraction of entities it matched at all, so a table matching every entity
        beats one matching a single entity many times.

        Every row and every column is read: the grep modality's per-file caps
        do not apply here.
        """
        table_weight = self.config.pneuma_table_name_weight
        column_weight = self.config.pneuma_column_name_weight
        cell_weight = self.config.pneuma_cell_weight
        patterns = {term: re.compile(_entity_pattern(term)) for term in terms}
        # A family weighted 0 cannot change any ranking, so its work is skipped:
        # with no weight on cells, no cell is read.
        scanned: list[ScanResult | None] = (
            scan_many(
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
                    sum(1 for name in facts.columns if patterns[term].search(name))
                    if column_weight
                    else 0
                )
                title_hit = int(
                    bool(table_weight and title and patterns[term].search(title))
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
    ) -> list[RetrievalHit]:
        if not self.content_search:
            hits: list[RetrievalHit] = []
            for rank, (_table_id, document) in enumerate(
                self._pneuma_candidates(question, top_k), 1
            ):
                # Pneuma 0.0.4 exposes ordered table IDs but not final scores.
                hits.append(
                    RetrievalHit(document=document, score=1.0 / rank, rank=rank)
                )
            return hits

        # A wider Pneuma pool than the caller asked for: content search reranks
        # within it, so a table at Pneuma rank 30 can still surface.
        limit = max(top_k, top_k * self.config.candidate_multiplier)
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
        scans_content = weight > 0.0
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
                "matched_terms": found,
                "content_score": candidate.content_score,
                "normalized_content_score": candidate.normalized_content,
                "normalized_pneuma_score": candidate.normalized_pneuma,
                "content_weight": weight,
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
