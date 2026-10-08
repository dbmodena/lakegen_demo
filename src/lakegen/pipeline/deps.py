"""What the engine calls: the phases, the resources and the sinks, injectable.

The engine never imports a phase or a resource itself; it is handed a
``PipelineDeps``. ``PipelineDeps.default()`` holds the real functions.
``PipelineDeps.from_module(module)`` reads them from a surface module
(``lakegen.service`` today) by name, at call time, so the tests that patch a
name on that module keep patching what the engine calls. ``from_module`` is
transitional: once the surfaces build their deps explicitly, the tests will
patch the deps and it goes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from types import ModuleType
from typing import Any

# The attribute name of each dependency on a surface module.
DEPENDENCY_NAMES = (
    "get_llm",
    "get_solr",
    "get_prompt_manager",
    "get_all_table_files",
    "phase1_generate_keywords",
    "phase2_select_tables",
    "phase12_agent",
    "phase3_generate_and_execute",
    "phase4_synthesize",
    "run_unified_orchestrated_discovery",
    "prepare_discovery_context",
    "select_from_prepared_context",
    "judge_semantic_code_result",
    "save_experiment_log",
    "log_retrieval_decision",
    "persist_manifest",
)


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
    def from_module(cls, module: ModuleType) -> PipelineDeps:
        """The dependencies as ``module`` names them now (patches included)."""
        return cls(**{name: getattr(module, name) for name in DEPENDENCY_NAMES})

    @classmethod
    def default(cls) -> PipelineDeps:
        """The real phases, resources and sinks."""
        from lakegen.core import logger as core_logger
        from lakegen.discovery.orchestrated import (
            run_unified_orchestrated_discovery,
            select_from_prepared_context,
        )
        from lakegen.lakes import get_all_table_files
        from lakegen.llm import get_llm
        from lakegen.manifest import persist_manifest
        from lakegen.orchestrated_context import prepare_discovery_context
        from lakegen.phases import (
            phase1_generate_keywords,
            phase2_select_tables,
            phase3_generate_and_execute,
            phase4_synthesize,
            phase12_agent,
        )
        from lakegen.prompts import get_prompt_manager
        from lakegen.retrieval.factory import get_solr
        from lakegen.retrieval.run_log import log_retrieval_decision
        from lakegen.semantic_code_judge import judge_semantic_code_result

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


assert tuple(field.name for field in fields(PipelineDeps)) == DEPENDENCY_NAMES
