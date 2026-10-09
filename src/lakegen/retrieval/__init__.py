from lakegen.config.retrieval import (
    DEFAULT_RERANK_DEPTH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_TOP_K,
    FusionMethod,
    MissingSignalPolicy,
    PneumaContentFusion,
    RetrievalConfig,
    RetrievalMode,
)
from lakegen.domain.retrieval import RetrievalHit, RetrievalRun, min_max_normalize
from lakegen.integrations.llm.embeddings import (
    EmbeddingGenerationError,
    check_embedding_health,
)
from lakegen.integrations.pneuma import SolrPneumaDocumentResolver
from lakegen.integrations.solr.indexing import (
    SolrEmbeddingIndexer,
    ensure_vector_schema,
    validate_stored_replacement_schema,
    validate_vector_schema,
)
from lakegen.integrations.solr.representation import METADATA_V1, represent_table
from lakegen.retrieval.evaluation import (
    RetrievalRunLogger,
    evaluate_ranking,
    mean_metrics,
)
from lakegen.retrieval.modes.duckdb_agentic import DuckDBAgenticRetriever
from lakegen.retrieval.modes.duckdb_semantic import DuckDBSemanticHybridRetriever
from lakegen.retrieval.modes.hybrid import HybridRetriever
from lakegen.retrieval.modes.keyword import KeywordRetriever
from lakegen.retrieval.modes.pneuma import PneumaRetriever
from lakegen.retrieval.modes.semantic import SemanticRetriever
from lakegen.retrieval.registry import (
    Branches,
    ModeRetriever,
    RetrievalRequest,
    RetrievalResult,
    build,
    register,
    registered_modes,
)
from lakegen.retrieval.service import TableRetrievalService

__all__ = [
    "Branches",
    "ModeRetriever",
    "RetrievalRequest",
    "RetrievalResult",
    "build",
    "register",
    "registered_modes",
    "DEFAULT_RERANK_DEPTH",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_TOP_K",
    "HybridRetriever",
    "FusionMethod",
    "EmbeddingGenerationError",
    "check_embedding_health",
    "KeywordRetriever",
    "METADATA_V1",
    "MissingSignalPolicy",
    "PneumaContentFusion",
    "RetrievalConfig",
    "RetrievalHit",
    "RetrievalMode",
    "RetrievalRun",
    "RetrievalRunLogger",
    "PneumaRetriever",
    "DuckDBAgenticRetriever",
    "DuckDBSemanticHybridRetriever",
    "SolrPneumaDocumentResolver",
    "SemanticRetriever",
    "SolrEmbeddingIndexer",
    "TableRetrievalService",
    "ensure_vector_schema",
    "validate_stored_replacement_schema",
    "validate_vector_schema",
    "evaluate_ranking",
    "mean_metrics",
    "min_max_normalize",
    "represent_table",
]
