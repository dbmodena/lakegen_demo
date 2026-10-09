"""What the engine calls: the phases, the resources and the sinks, injectable.

The engine never imports a phase or a resource itself; it is handed a
``PipelineDeps``. ``PipelineDeps.default()`` holds the real functions. Each
surface (the batch service, ``lakegen ask``, the Chainlit app) builds its own
from the names it imports, at the start of a run, so a test that replaces a
name on the surface replaces what the engine calls; a test of the engine
itself passes a ``PipelineDeps`` of fakes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineDeps:
    get_llm: Callable[..., Any]
    get_solr: Callable[..., Any]
    get_prompt_manager: Callable[..., Any]
    get_all_table_files: Callable[..., Any]
    phase1_generate_keywords: Callable[..., Any]
    phase2_select_tables: Callable[..., Any]
    phase12_agent: Callable[..., Any]
    phase3_generate_and_execute: Callable[..., Any]
    phase4_synthesize: Callable[..., Any]
    run_unified_orchestrated_discovery: Callable[..., Any]
    prepare_discovery_context: Callable[..., Any]
    select_from_prepared_context: Callable[..., Any]
    judge_semantic_code_result: Callable[..., Any]
    save_experiment_log: Callable[..., Any]
    log_retrieval_decision: Callable[..., Any]
    persist_manifest: Callable[..., Any]

    @classmethod
    def default(cls) -> PipelineDeps:
        """The real phases, resources and sinks."""
        from lakegen.integrations.llm import get_llm
        from lakegen.integrations.table_io import get_all_table_files
        from lakegen.retrieval.factory import get_solr
        from lakegen.retrieval.run_log import log_retrieval_decision
        from lakegen.workflow.answer.synthesize import phase4_synthesize
        from lakegen.workflow.coding.agent import phase3_generate_and_execute
        from lakegen.workflow.discovery.context import prepare_discovery_context
        from lakegen.workflow.discovery.divided import phase2_select_tables
        from lakegen.workflow.discovery.keywords import phase1_generate_keywords
        from lakegen.workflow.discovery.orchestrated import (
            run_unified_orchestrated_discovery,
            select_from_prepared_context,
        )
        from lakegen.workflow.discovery.unified import phase12_agent
        from lakegen.workflow.evaluation.semantic_judge import judge_semantic_code_result
        from lakegen.workflow.observability import csv_log as core_logger
        from lakegen.workflow.observability.manifest import persist_manifest
        from lakegen.workflow.prompts import get_prompt_manager

        return cls(
            get_llm=get_llm,
            get_solr=get_solr,
            get_prompt_manager=get_prompt_manager,
            get_all_table_files=get_all_table_files,
            phase1_generate_keywords=phase1_generate_keywords,
            phase2_select_tables=phase2_select_tables,
            phase12_agent=phase12_agent,
            phase3_generate_and_execute=phase3_generate_and_execute,
            phase4_synthesize=phase4_synthesize,
            run_unified_orchestrated_discovery=run_unified_orchestrated_discovery,
            prepare_discovery_context=prepare_discovery_context,
            select_from_prepared_context=select_from_prepared_context,
            judge_semantic_code_result=judge_semantic_code_result,
            save_experiment_log=core_logger.save_experiment_log,
            log_retrieval_decision=log_retrieval_decision,
            persist_manifest=persist_manifest,
        )

