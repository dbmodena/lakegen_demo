"""Where the retrievers used to live; each is its own module now.

Kept for the scripts that import from here: ``keyword``, ``semantic``,
``hybrid``, ``duckdb_semantic``, ``modes`` and ``service``.
"""

from lakegen.retrieval.duckdb_semantic import DuckDBSemanticHybridRetriever  # noqa: F401
from lakegen.retrieval.hybrid import HybridRetriever, MissingScoreResolver  # noqa: F401
from lakegen.retrieval.keyword import KeywordRetriever  # noqa: F401
from lakegen.retrieval.models import min_max_normalize  # noqa: F401
from lakegen.retrieval.modes import (  # noqa: F401
    DuckDBMode,
    DuckDBSemanticMode,
    HybridMode,
    KeywordMode,
    PneumaMode,
    SemanticMode,
)
from lakegen.retrieval.semantic import SemanticRetriever  # noqa: F401
from lakegen.retrieval.service import RunObserver, TableRetrievalService  # noqa: F401
