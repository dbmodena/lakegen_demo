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
* **Fusion is a weighted sum by default.** The paper combines the two scores;
  the reference code instead keeps Pneuma's top-k and lets content-ranked tables
  fill only the slots its deduplication left over. The weighted sum is the
  paper's reading, and it lets a table Pneuma ranked 30th surface on content
  alone; ``pneuma_content_fusion: fill`` selects the reference merge, except
  that only tables matching an entity fill a slot (the reference pads with
  zero-score tables in name order).
* **A table's name is its catalog title.** The reference code matches the
  DuckDB table name. The lakes here name files by opaque id (``43nn-pn8j``), so
  the title the resolver returns for the file is matched instead -- counted,
  like the reference's name match, once per text column.

As in the reference, only text (String) columns are searched, for cells and
column names alike, every row is read (the grep modality's per-file row and
column caps do not apply), and every regex occurrence counts, so a cell naming
an entity twice counts twice. A table without text columns scores nothing.
* **Enumeration is an agent action in the paper**, invoked on demand by the
  Conductor. Retrieval here is a single call, so enumeration runs automatically
  from the fused ranking instead.

Under this mode the discovery prompts ask the agent for the entities with the
reference code's extraction rules -- specific, named, canonical strings written
as the question writes them, never general concepts -- in
``RetrievalIntent.entities`` or the search tool's ``entities``. A caller that
passes no entities at all, such as the retrieval benchmark, gets them from
``pneuma_entity_model`` instead: one chat call per question with the reference
code's own prompt (:data:`ENTITY_EXTRACTION_PROMPT`), as its retriever makes.

The reference retriever's Pneuma is its hybrid search alone, with the LLM
relevance judge switched off (``RerankingMode.NONE``). The Pneuma service here
always judges, so even ``fill`` with extracted entities is not the reference
retriever exactly.

The package: ``client`` (the Pneuma service and its answer), ``entities``
(what the Seeker scans for), ``seeker`` (the content scan), ``enumeration``
(the table families), ``resolver`` (Pneuma's ids back to the catalogue),
the ranking itself is ``lakegen.retrieval.modes.pneuma``. Scripts that replace a collaborator
patch it on its own module, which the retriever reads at call time.
"""

from lakegen.integrations.pneuma.client import HttpPneumaClient, PneumaClient, create_pneuma_client
from lakegen.integrations.pneuma.entities import (
    ENTITY_EXTRACTION_PROMPT,
    EntityExtractor,
    OCIEntityExtractor,
    TogetherEntityExtractor,
    entity_extractor_for,
)
from lakegen.integrations.pneuma.resolver import DocumentResolver, SolrPneumaDocumentResolver
from lakegen.integrations.pneuma.seeker import probe_many, scan_many

__all__ = [
    "ENTITY_EXTRACTION_PROMPT",
    "DocumentResolver",
    "EntityExtractor",
    "HttpPneumaClient",
    "OCIEntityExtractor",
    "PneumaClient",
    "SolrPneumaDocumentResolver",
    "TogetherEntityExtractor",
    "create_pneuma_client",
    "entity_extractor_for",
    "probe_many",
    "scan_many",
]
