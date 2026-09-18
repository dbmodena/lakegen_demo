"""Configuration shared by keyword, semantic, and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
import os

DEFAULT_TOP_K = 20
PNEUMA_PORTAL_ROUTES = {
    "nyc": ("http://localhost:8765", "lakegen-cohere-v4-1024"),
    "uk": ("http://localhost:8766", "lakegen-cohere-v4-1024"),
}


class RetrievalMode(StrEnum):
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    PNEUMA = "pneuma"
    PNEUMA_SEEKER = "pneuma_seeker"
    DUCKDB_AGENTIC = "duckdb_agentic"
    GREP = "grep"
    GREP_VALUES = "grep_values"

    @property
    def is_grep(self) -> bool:
        """True for the grep family: the full-signal mode and its values-only twin."""
        return self in (RetrievalMode.GREP, RetrievalMode.GREP_VALUES)

    @property
    def is_pneuma(self) -> bool:
        """True for the Pneuma family: the index alone, and the Seeker augmentation."""
        return self in (RetrievalMode.PNEUMA, RetrievalMode.PNEUMA_SEEKER)

    @property
    def value_keywords(self) -> bool:
        """True when search terms should be cell values rather than dataset topics.

        ``grep_values`` matches nothing but cell contents, so a term that names a
        dataset from the outside ("school enrollment") finds little; the values
        its rows actually store ("Brooklyn", "suspension") are what it can find.
        The prompts that produce search terms read this flag.
        """
        return self is RetrievalMode.GREP_VALUES

    @property
    def verbatim_entities(self) -> bool:
        """True when discovery must list entities exactly as the question names them.

        ``pneuma_seeker`` scans table content for those entities, as Pneuma-Seeker
        does: identifiers, codes, and names likely stored verbatim, never general
        concepts. The prompts and the agent's search tool read this flag.
        """
        return self is RetrievalMode.PNEUMA_SEEKER

    @property
    def ranks_question_only(self) -> bool:
        """True when retrieval ranks the question alone and ignores search terms.

        ``semantic`` embeds the question and ``pneuma`` sends it to the Pneuma
        service, so the concepts discovery produces reach neither. The agent is
        still asked for them, which keeps it from telling retrievers apart, but
        records must not present them as what was searched.
        """
        return self in (RetrievalMode.SEMANTIC, RetrievalMode.PNEUMA)

    def split_keywords(self, keywords: Sequence[str] | None) -> tuple[list[str], list[str]]:
        """Discovery's keywords as (searched, not searched) under this mode."""
        listed = list(keywords or ())
        return ([], listed) if self.ranks_question_only else (listed, [])

    @property
    def requires_table_dir(self) -> bool:
        """True for modalities that read local Parquet instead of a Solr index."""
        return (
            self.is_grep
            or self is RetrievalMode.DUCKDB_AGENTIC
            # Plain pneuma stays service-only; only the Seeker scans cells.
            or self is RetrievalMode.PNEUMA_SEEKER
        )


class MissingSignalPolicy(StrEnum):
    ZERO = "zero"
    RESCORE = "rescore"


class FusionMethod(StrEnum):
    WEIGHTED = "weighted"
    RRF = "rrf"


@dataclass(frozen=True)
class RetrievalConfig:
    """Reproducible retrieval settings for a LakeGen experiment."""

    mode: RetrievalMode = RetrievalMode.KEYWORD
    top_k: int = DEFAULT_TOP_K
    alpha: float = 0.5
    candidate_multiplier: int = 5
    representation_version: str = "metadata-v1"
    embedding_model: str = "cohere.embed-v4.0"
    embedding_base_url: str = "http://localhost:11434"
    vector_field: str = "table_embedding"
    lexical_query_fields: str | None = None
    # Keyword mode only. q.op=AND requires every tokenized word across every
    # keyword to appear in one document; a zero-hit AND query normally just
    # returns nothing. When true, retry it once as q.op=OR instead of
    # collapsing to zero candidates. Off by default to preserve the
    # pre-existing strict-AND baseline.
    keyword_or_fallback: bool = False
    missing_signal_policy: MissingSignalPolicy = MissingSignalPolicy.ZERO
    fusion_method: FusionMethod = FusionMethod.WEIGHTED
    rrf_k: int = 60
    pneuma_index_name: str = "lakegen"
    pneuma_base_url: str = "http://localhost:8767"
    pneuma_timeout_seconds: float = 120.0
    # Pneuma-Seeker: weight of the content-search branch against Pneuma's own
    # ranking. ``alpha`` cannot serve here -- it is forwarded to Pneuma's
    # internal hybrid search. The scan shares ``scan_workers`` and the
    # ``grep_max_files`` cut with grep, but reads every row and every column of
    # a file: the per-file ``grep_*`` row and column caps do not apply to it.
    pneuma_content_weight: float = 0.5
    pneuma_enumerate_tables: bool = True
    # Pneuma-Seeker content search, per entity and table: a raw score of
    # table_name_weight x a match in the table's catalog title
    # + column_name_weight x matching column names + cell_weight x matching
    # cells, damped by log(1 + raw). 3, 2 and 1 are upstream Pneuma-Seeker's
    # weights. A family weighted 0 is skipped: a cell weight of 0 reads no cell.
    pneuma_table_name_weight: float = 3.0
    pneuma_column_name_weight: float = 2.0
    pneuma_cell_weight: float = 1.0
    duckdb_max_files: int = 250
    duckdb_max_columns_per_file: int = 40
    duckdb_sample_rows: int = 3
    duckdb_max_scan_rows_per_file: int = 100_000
    duckdb_probe_files: int = 25
    duckdb_probe_rows_per_file: int = 1_000
    # None means no cap: every file in the lake is scanned. A cap is an explicit,
    # flagged approximation -- it changes which files are read AND the IDF the
    # survivors are scored with, so a capped run is not comparable to a full one.
    grep_max_files: int | None = None
    grep_max_columns_per_file: int = 40
    grep_max_scan_rows_per_file: int = 100_000
    grep_sample_rows: int = 3
    grep_probe_files: int = 25
    grep_probe_rows_per_file: int = 1_000
    # Worker processes for the per-file scan map, shared by grep and the Pneuma
    # content search. Every polars query runs in these workers, never in the
    # calling process, so a native polars crash costs a retried batch rather than
    # the application. Each file is an independent bounded read; 16 is the
    # measured optimum on a 32-core host, where the speedup saturates because the
    # work is I/O-bound. 1 still scans outside the calling process, one file at a
    # time.
    scan_workers: int = 8
    # Weight of each grep evidence family in the score. A family weighted 0
    # cannot change any ranking, so the work behind it is skipped rather than
    # computed and multiplied away: 0 for values means no cell is ever read,
    # 0 for metadata means no filename, column name, or catalog field is ever
    # matched. Both at 0 would score every table 0, so that is rejected.
    grep_value_weight: float = 1.0
    grep_metadata_weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", RetrievalMode(self.mode))
        object.__setattr__(
            self,
            "missing_signal_policy",
            MissingSignalPolicy(self.missing_signal_policy),
        )
        object.__setattr__(self, "fusion_method", FusionMethod(self.fusion_method))
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than zero")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        if self.pneuma_timeout_seconds <= 0:
            raise ValueError("pneuma_timeout_seconds must be greater than zero")
        if not 0.0 <= self.pneuma_content_weight <= 1.0:
            raise ValueError("pneuma_content_weight must be between 0 and 1")
        pneuma_weights = (
            "pneuma_table_name_weight",
            "pneuma_column_name_weight",
            "pneuma_cell_weight",
        )
        for name in pneuma_weights:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if not any(getattr(self, name) for name in pneuma_weights):
            raise ValueError(
                "pneuma_table_name_weight, pneuma_column_name_weight and "
                "pneuma_cell_weight must not all be zero: no content evidence "
                "would remain to rank on"
            )
        for name in (
            "duckdb_max_files",
            "duckdb_max_columns_per_file",
            "duckdb_sample_rows",
            "duckdb_max_scan_rows_per_file",
            "duckdb_probe_files",
            "duckdb_probe_rows_per_file",
            "grep_max_columns_per_file",
            "grep_max_scan_rows_per_file",
            "grep_sample_rows",
            "grep_probe_files",
            "grep_probe_rows_per_file",
            "scan_workers",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.grep_max_files is not None and self.grep_max_files <= 0:
            raise ValueError("grep_max_files must be greater than zero or None")
        for name in ("grep_value_weight", "grep_metadata_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        if self.grep_value_weight == 0 and self.grep_metadata_weight == 0:
            raise ValueError(
                "grep_value_weight and grep_metadata_weight must not both be zero: "
                "no evidence would remain to rank on"
            )
        for name in ("representation_version", "embedding_model", "vector_field"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        for name in (
            "pneuma_index_name",
            "pneuma_base_url",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if self.lexical_query_fields is not None:
            query_fields = self.lexical_query_fields.strip()
            object.__setattr__(self, "lexical_query_fields", query_fields or None)

    @property
    def branch_candidate_count(self) -> int:
        return self.top_k * self.candidate_multiplier

    def with_mode(self, mode: RetrievalMode | str) -> "RetrievalConfig":
        return replace(self, mode=RetrievalMode(mode))

    def for_portal(self, portal: str) -> "RetrievalConfig":
        """Select the independent Pneuma service belonging to a portal."""
        normalized = portal.strip().casefold()
        route = PNEUMA_PORTAL_ROUTES.get(normalized)
        if route is None:
            return self
        default_url, default_index = route
        env_prefix = f"LAKEGEN_PNEUMA_{normalized.upper()}"
        return replace(
            self,
            pneuma_index_name=os.environ.get(
                f"{env_prefix}_INDEX_NAME", default_index
            ),
            pneuma_base_url=os.environ.get(f"{env_prefix}_BASE_URL", default_url),
        )

    @classmethod
    def from_env(
        cls,
        *,
        mode: RetrievalMode | str | None = None,
        top_k: int | None = None,
        alpha: float | None = None,
        candidate_multiplier: int | None = None,
    ) -> "RetrievalConfig":
        selected_mode = mode or os.environ.get(
            "LAKEGEN_RETRIEVAL_MODE", RetrievalMode.KEYWORD
        )
        return cls(
            mode=RetrievalMode(selected_mode),
            top_k=top_k
            if top_k is not None
            else int(os.environ.get("LAKEGEN_RETRIEVAL_TOP_K", str(DEFAULT_TOP_K))),
            alpha=alpha
            if alpha is not None
            else float(os.environ.get("LAKEGEN_HYBRID_ALPHA", "0.5")),
            candidate_multiplier=candidate_multiplier
            if candidate_multiplier is not None
            else int(os.environ.get("LAKEGEN_CANDIDATE_MULTIPLIER", "5")),
            representation_version=os.environ.get(
                "LAKEGEN_REPRESENTATION_VERSION", "metadata-v1"
            ),
            embedding_model=os.environ.get(
                "LAKEGEN_EMBEDDING_MODEL", "cohere.embed-v4.0"
            ),
            embedding_base_url=os.environ.get(
                "LAKEGEN_EMBEDDING_BASE_URL", "http://localhost:11434"
            ),
            vector_field=os.environ.get(
                "LAKEGEN_VECTOR_FIELD", "table_embedding"
            ),
            lexical_query_fields=os.environ.get("LAKEGEN_BM25_QUERY_FIELDS"),
            missing_signal_policy=MissingSignalPolicy(
                os.environ.get("LAKEGEN_MISSING_SIGNAL_POLICY", "zero")
            ),
            fusion_method=FusionMethod(
                os.environ.get("LAKEGEN_FUSION_METHOD", "weighted")
            ),
            rrf_k=int(os.environ.get("LAKEGEN_RRF_K", "60")),
            pneuma_index_name=os.environ.get("LAKEGEN_PNEUMA_INDEX_NAME", "lakegen-cohere-v4-1024"),
            pneuma_base_url=os.environ.get(
                "LAKEGEN_PNEUMA_BASE_URL", "http://localhost:8767"
            ),
            pneuma_timeout_seconds=float(
                os.environ.get("LAKEGEN_PNEUMA_TIMEOUT_SECONDS", "120")
            ),
            pneuma_content_weight=float(
                os.environ.get("LAKEGEN_PNEUMA_CONTENT_WEIGHT", "0.5")
            ),
            pneuma_enumerate_tables=os.environ.get(
                "LAKEGEN_PNEUMA_ENUMERATE_TABLES", "1"
            ).strip().casefold() not in ("0", "false", "no"),
            pneuma_table_name_weight=float(
                os.environ.get("LAKEGEN_PNEUMA_TABLE_NAME_WEIGHT", "3.0")
            ),
            pneuma_column_name_weight=float(
                os.environ.get("LAKEGEN_PNEUMA_COLUMN_NAME_WEIGHT", "2.0")
            ),
            pneuma_cell_weight=float(
                os.environ.get("LAKEGEN_PNEUMA_CELL_WEIGHT", "1.0")
            ),
            duckdb_max_files=int(os.environ.get("LAKEGEN_DUCKDB_MAX_FILES", "250")),
            duckdb_max_columns_per_file=int(
                os.environ.get("LAKEGEN_DUCKDB_MAX_COLUMNS_PER_FILE", "40")
            ),
            duckdb_sample_rows=int(
                os.environ.get("LAKEGEN_DUCKDB_SAMPLE_ROWS", "3")
            ),
            duckdb_max_scan_rows_per_file=int(
                os.environ.get("LAKEGEN_DUCKDB_MAX_SCAN_ROWS_PER_FILE", "100000")
            ),
            duckdb_probe_files=int(
                os.environ.get("LAKEGEN_DUCKDB_PROBE_FILES", "25")
            ),
            duckdb_probe_rows_per_file=int(
                os.environ.get("LAKEGEN_DUCKDB_PROBE_ROWS_PER_FILE", "1000")
            ),
            grep_max_files=(
                int(limit)
                if (limit := os.environ.get("LAKEGEN_GREP_MAX_FILES", "").strip())
                else None
            ),
            grep_max_columns_per_file=int(
                os.environ.get("LAKEGEN_GREP_MAX_COLUMNS_PER_FILE", "40")
            ),
            grep_max_scan_rows_per_file=int(
                os.environ.get("LAKEGEN_GREP_MAX_SCAN_ROWS_PER_FILE", "100000")
            ),
            grep_sample_rows=int(os.environ.get("LAKEGEN_GREP_SAMPLE_ROWS", "3")),
            grep_probe_files=int(os.environ.get("LAKEGEN_GREP_PROBE_FILES", "25")),
            scan_workers=int(os.environ.get("LAKEGEN_SCAN_WORKERS", "16")),
            grep_value_weight=float(
                os.environ.get("LAKEGEN_GREP_VALUE_WEIGHT", "1.0")
            ),
            grep_metadata_weight=float(
                os.environ.get("LAKEGEN_GREP_METADATA_WEIGHT", "1.0")
            ),
            grep_probe_rows_per_file=int(
                os.environ.get("LAKEGEN_GREP_PROBE_ROWS_PER_FILE", "1000")
            ),
        )
