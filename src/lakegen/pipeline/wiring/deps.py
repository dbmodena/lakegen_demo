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
    """The phases, resources and sinks one run calls.

    Attributes:
        get_llm: Returns the model and its token counter, by model name.
        get_solr: Returns the lake's Solr client.
        get_prompt_manager: Returns the prompt manager.
        get_all_table_files: Lists the lake's table files.
        write_keywords: The divided architecture's keyword writer.
        select_tables: The divided architecture's table selection.
        discover_tables: The unified agent: search, inspection and selection.
        write_and_run_code: The coder: writes and runs the analysis code.
        phase4_synthesize: Writes the answer from the code's result.
        judge_semantic_code_result: The semantic judge of a code result.
        save_experiment_log: Writes the run's log row.
        log_retrieval_decision: Writes the run's retrieval decision.
        persist_manifest: Writes the run's manifest.
    """

    get_llm: Callable[..., Any]
    get_solr: Callable[..., Any]
    get_prompt_manager: Callable[..., Any]
    get_all_table_files: Callable[..., Any]
    write_keywords: Callable[..., Any]
    select_tables: Callable[..., Any]
    discover_tables: Callable[..., Any]
    write_and_run_code: Callable[..., Any]
    phase4_synthesize: Callable[..., Any]
    judge_semantic_code_result: Callable[..., Any]
    save_experiment_log: Callable[..., Any]
    log_retrieval_decision: Callable[..., Any]
    persist_manifest: Callable[..., Any]

    @classmethod
    def default(cls) -> PipelineDeps:
        """The real phases, resources and sinks."""
        from lakegen.llm import get_llm
        from lakegen.llm.prompts import get_prompt_manager
        from lakegen.pipeline.answer import phase4_synthesize
        from lakegen.pipeline.coding.agent import write_and_run_code
        from lakegen.pipeline.discovery.divided import select_tables
        from lakegen.pipeline.discovery.keywords import write_keywords
        from lakegen.pipeline.discovery.unified import discover_tables
        from lakegen.pipeline.evaluation.semantic_judge import judge_semantic_code_result
        from lakegen.pipeline.observability import experiment_log as core_logger
        from lakegen.pipeline.observability.manifest import persist_manifest
        from lakegen.retrieval.factory import get_solr
        from lakegen.retrieval.run_log import log_retrieval_decision
        from lakegen.utils.table_io import get_all_table_files

        return cls(
            get_llm=get_llm,
            get_solr=get_solr,
            get_prompt_manager=get_prompt_manager,
            get_all_table_files=get_all_table_files,
            write_keywords=write_keywords,
            select_tables=select_tables,
            discover_tables=discover_tables,
            write_and_run_code=write_and_run_code,
            phase4_synthesize=phase4_synthesize,
            judge_semantic_code_result=judge_semantic_code_result,
            save_experiment_log=core_logger.save_experiment_log,
            log_retrieval_decision=log_retrieval_decision,
            persist_manifest=persist_manifest,
        )
