"""The Solr client of a lake and the retrieval service a run searches with."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from lakegen.config.settings import get_settings
from lakegen.datastructures.retrieval import RetrievalRun
from lakegen.retrieval import RetrievalConfig, RetrievalService, run_log
from lakegen.retrieval.solr.client import SolrClient, resolve_solr_core


@lru_cache(maxsize=8)
def get_solr(core):
    """The Solr client of ``core`` (resolved through the settings), cached per core name."""
    return SolrClient(
        core=resolve_solr_core(core),
        base_url=get_settings().solr_base_url,
    )


def get_table_retrieval_service(
    solr: SolrClient,
    config: RetrievalConfig,
    table_dir: str | Path | None = None,
    *,
    observer: Callable[[RetrievalRun], None] | None = None,
) -> RetrievalService:
    """Build the retrieval service a run searches with.

    Args:
        solr: The lake's Solr client.
        config: The retrieval configuration (mode, top k, fusion, reranking).
        table_dir: The lake's local table directory, for the modes that read the files.
        observer: Called with every retrieval run; defaults to
            ``run_log.log_and_capture_retrieval_run``, read at call time so that
            experiment scripts can replace it.

    Returns:
        The service.
    """
    return RetrievalService(
        solr,
        config,
        observer=observer or run_log.log_and_capture_retrieval_run,
        table_dir=str(table_dir) if table_dir is not None else None,
    )
