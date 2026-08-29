"""Programmatic retrieval and stable discovery context construction.

This module is deliberately independent from agent tools: callers receive plain,
serializable data and therefore cannot accidentally expose a Solr callable to the
discovery agent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from lakegen.core.resources import get_table_retrieval_service
from lakegen.core.types import SolrMetadata
from lakegen.phases.utils import match_local_csv, solr_metadata_from_doc
from lakegen.retrieval import RetrievalConfig
from src.client_solr import LocalSolrClient


class PreparedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    retrieval_rank: int
    prepared_position: int
    dataset: str
    scores: dict[str, Any]
    missing_signals: list[str]
    metadata: dict[str, Any]


class PreparedDiscoveryContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    retrieval_mode: str
    candidates: list[PreparedCandidate]
    retrieved_hit_count: int
    prepared_candidate_count: int

    def stable_json(self) -> str:
        """Serialize the complete technical context for telemetry and logs."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def agent_payload(self) -> dict[str, Any]:
        """Return a stable, mode-neutral view containing no retrieval signals."""
        candidates: list[dict[str, Any]] = []
        for item in self.candidates:
            metadata = item.metadata
            names = list(metadata.get("columns.name", []) or [])
            descriptions = list(metadata.get("columns.description", []) or [])
            types = list(metadata.get("columns.type", []) or [])
            columns = []
            for index, name in enumerate(names):
                column: dict[str, Any] = {"name": name}
                if index < len(descriptions) and descriptions[index]:
                    column["description"] = descriptions[index]
                if index < len(types) and types[index]:
                    column["type"] = types[index]
                columns.append(column)
            candidates.append({
                "position": item.prepared_position,
                "dataset": item.dataset,
                "metadata": {
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "tags": list(metadata.get("tags", []) or []),
                    "columns": columns,
                },
            })
        return {"query": self.query, "candidates": candidates}

    def agent_json(self) -> str:
        """Serialize only the mode-neutral context shown to an agent."""
        return json.dumps(
            self.agent_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def prepare_discovery_context(
    *,
    query: str,
    keywords: list[str],
    solr_client: LocalSolrClient,
    all_files: list[str],
    retrieval_config: RetrievalConfig,
    table_dir: Path | None = None,
) -> tuple[PreparedDiscoveryContext, SolrMetadata]:
    """Run the existing retriever and map its ranked hits to local datasets."""

    retriever = get_table_retrieval_service(
        solr_client,
        retrieval_config,
        *([table_dir] if retrieval_config.mode.value == "duckdb_agentic" else []),
    )
    hits = retriever.retrieve(
        question=query,
        keywords=keywords,
        top_k=retrieval_config.top_k,
        lexical_fetch_k=15,
        q_op="AND",
    )
    candidates: list[PreparedCandidate] = []
    metadata_by_dataset: SolrMetadata = {}
    seen: set[str] = set()
    for hit in hits:
        dataset = match_local_csv(hit.document, all_files)
        if dataset is None or dataset in seen:
            continue
        seen.add(dataset)
        metadata = solr_metadata_from_doc(hit.document)
        hit_log = hit.to_log_dict()
        metadata["retrieval"] = hit_log
        metadata_by_dataset[dataset] = metadata
        scores = {
            key: hit_log.get(key)
            for key in ("score", "lexical_score", "semantic_score")
            if hit_log.get(key) is not None
        }
        missing = [
            key
            for key in ("lexical_score", "semantic_score")
            if hit_log.get(key) is None
        ]
        candidates.append(PreparedCandidate(
            retrieval_rank=hit.rank,
            prepared_position=len(candidates) + 1,
            dataset=dataset,
            scores=scores,
            missing_signals=missing,
            metadata=metadata,
        ))
        if len(candidates) >= retrieval_config.top_k:
            break

    context = PreparedDiscoveryContext(
        query=query,
        retrieval_mode=retrieval_config.mode.value,
        candidates=candidates,
        retrieved_hit_count=len(hits),
        prepared_candidate_count=len(candidates),
    )
    return context, metadata_by_dataset
