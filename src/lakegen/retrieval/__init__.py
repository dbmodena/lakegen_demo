"""Table retrieval: rank the lake's tables for a question.

``RetrievalService`` runs one of the registered modes (keyword, semantic,
hybrid, DuckDB, Pneuma; see ``registry`` and ``modes``) and reports each run to
an observer; ``evaluation`` scores rankings and ``run_log`` keeps them. The
names re-exported here are the package's public surface.
"""

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
from lakegen.datastructures.retrieval import RetrievalHit, RetrievalRun, min_max_normalize
from lakegen.llm.embeddings import (
    EmbeddingGenerationError,
    check_embedding_health,
)
from lakegen.retrieval.benchmark import (
    RetrievalRunLogger,
    evaluate_ranking,
    mean_metrics,
)
from lakegen.retrieval.modes.duckdb_agentic import DuckDBAgenticRetriever
from lakegen.retrieval.modes.duckdb_semantic import DuckDBSemanticRetriever
from lakegen.retrieval.modes.hybrid import HybridRetriever
from lakegen.retrieval.modes.keyword import KeywordRetriever
from lakegen.retrieval.modes.pneuma import PneumaRetriever
from lakegen.retrieval.modes.semantic import SemanticRetriever
from lakegen.retrieval.pneuma import SolrPneumaDocumentResolver
from lakegen.retrieval.registry import (
    Branches,
    ModeRetriever,
    RetrievalRequest,
    RetrievalResult,
    build,
    register,
    registered_modes,
)
from lakegen.retrieval.service import RetrievalService

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
    "MissingSignalPolicy",
    "PneumaContentFusion",
    "RetrievalConfig",
    "RetrievalHit",
    "RetrievalMode",
    "RetrievalRun",
    "RetrievalRunLogger",
    "PneumaRetriever",
    "DuckDBAgenticRetriever",
    "DuckDBSemanticRetriever",
    "SolrPneumaDocumentResolver",
    "SemanticRetriever",
    "RetrievalService",
    "evaluate_ranking",
    "mean_metrics",
    "min_max_normalize",
]
