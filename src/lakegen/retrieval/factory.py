"""The Solr client of a lake and the retrieval service a run searches with."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from lakegen.config.settings import get_settings
from lakegen.domain.retrieval import RetrievalRun
from lakegen.integrations.solr.client import LocalSolrClient, resolve_solr_core
from lakegen.retrieval import RetrievalConfig, TableRetrievalService, run_log


@lru_cache(maxsize=8)
def get_solr(core):
    return LocalSolrClient(
        core=resolve_solr_core(core),
        base_url=get_settings().solr_base_url,
    )


def get_table_retrieval_service(
    solr: LocalSolrClient,
    config: RetrievalConfig,
    table_dir: str | Path | None = None,
    *,
    observer: Callable[[RetrievalRun], None] | None = None,
) -> TableRetrievalService:
    return TableRetrievalService(
        solr,
        config,
        observer=observer or run_log.log_and_capture_retrieval_run,
        table_dir=str(table_dir) if table_dir is not None else None,
    )
