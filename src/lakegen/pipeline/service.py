"""Non-interactive LakeGen workflow used by API and batch clients."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from lakegen.config.experiment import (
    DiscoveryArchitecture,
    ExperimentConfig,
    InteractionMode,
    load_experiment_config,
)
from lakegen.config.paths import resolve_portal_tables_dir
from lakegen.config.registry import get_lake
from lakegen.config.runtime import MODEL_OPTIONS, SOLR_CORE_OPTIONS, RuntimeSettings
from lakegen.datastructures.results import QueryResult, QuestionSource, extract_questions  # noqa: F401
from lakegen.llm import get_llm  # noqa: F401
from lakegen.llm.prompts import get_prompt_manager  # noqa: F401
from lakegen.pipeline.answer import phase4_synthesize
from lakegen.pipeline.coding.agent import write_and_run_code
from lakegen.pipeline.discovery.divided import select_tables
from lakegen.pipeline.discovery.keywords import write_keywords
from lakegen.pipeline.discovery.unified import discover_tables

# The constants and helpers live in the engine; they are re-exported for the
# tests and callers that import them from here.
from lakegen.pipeline.engine import (  # noqa: F401
    MAX_CODE_ATTEMPTS,
    MAX_TABLE_ATTEMPTS,
    _record_semantic_plan_telemetry,
    _rejected_selection_signature,
    _selection_plan_signature,
    _selection_retry_feedback,
    run_pipeline,
)
from lakegen.pipeline.evaluation.semantic_judge import judge_semantic_code_result  # noqa: F401

# The phases, resources and sinks the engine calls, imported here by name:
# pipeline_deps() binds them when a run starts, so a test patching
# `lakegen.pipeline.service.<name>` patches what the engine calls.
from lakegen.pipeline.observability.experiment_log import save_experiment_log  # noqa: F401
from lakegen.pipeline.observability.manifest import (
    ExperimentManifest,
    persist_manifest,  # noqa: F401
)
from lakegen.pipeline.replay import Replay
from lakegen.pipeline.run.context import RunContext
from lakegen.pipeline.wiring.deps import PipelineDeps
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalConfig, RetrievalMode
from lakegen.retrieval.factory import get_solr  # noqa: F401
from lakegen.retrieval.run_log import log_retrieval_decision  # noqa: F401
from lakegen.utils.table_io import get_all_table_files  # noqa: F401

logger = logging.getLogger(__name__)


def make_runtime_settings(
    *,
    core: str,
    model: str,
    use_unified_agent: bool = True,
    retrieval_mode: RetrievalMode | str = RetrievalMode.KEYWORD,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = 0.5,
    candidate_multiplier: int = 5,
    interaction_mode: InteractionMode | str = InteractionMode.AUTONOMOUS,
    experiment_id: str = "default",
    seed: int = 0,
    config: ExperimentConfig | None = None,
) -> RuntimeSettings:
    """The runtime settings of a run on ``core`` with ``model``.

    With a ``config``, its retrieval settings and the config itself are
    used; without one, the retrieval settings come from the arguments and
    the environment, and the experiment config is loaded with them as
    overrides. Either way the retrieval is adjusted for the portal.

    Args:
        core: The lake (Solr core).
        model: The model name.
        use_unified_agent: Run the unified discovery agent rather than the divided one.
        retrieval_mode: The retrieval mode, when no config is given.
        top_k: How many tables retrieval returns, when no config is given.
        alpha: The hybrid weight, when no config is given.
        candidate_multiplier: The hybrid modes retrieve ``top_k`` times this many
            candidates per branch before fusing, when no config is given.
        interaction_mode: The interaction mode, when no config is given.
        experiment_id: The experiment id, when no config is given.
        seed: The seed, when no config is given.
        config: The resolved experiment config, if there is one.

    Returns:
        The runtime settings.

    Raises:
        ValueError: When the core or the model is unknown.
    """
    if core not in SOLR_CORE_OPTIONS:
        raise ValueError(f"Unknown core {core!r}. Expected one of: {', '.join(SOLR_CORE_OPTIONS)}")
    if model not in MODEL_OPTIONS:
        raise ValueError(f"Unknown model {model!r}. Expected one of: {', '.join(MODEL_OPTIONS)}")

    retrieval = (
        config.retrieval.to_runtime()
        if config is not None
        else RetrievalConfig.from_env(
            mode=retrieval_mode,
            top_k=top_k,
            alpha=alpha,
            candidate_multiplier=candidate_multiplier,
        )
    )
    retrieval = retrieval.for_portal(core)
    resolved_config = config or load_experiment_config(
        overrides={
            "model": model,
            "core": core,
            "discovery_architecture": (
                DiscoveryArchitecture.UNIFIED.value
                if use_unified_agent
                else DiscoveryArchitecture.DIVIDED.value
            ),
            "interaction_mode": str(interaction_mode),
            "experiment_id": experiment_id,
            "seed": seed,
            **{f"retrieval.{key}": value for key, value in retrieval.__dict__.items()},
        }
    )
    return RuntimeSettings(
        model_name=model,
        solr_core=core,
        csv_dir=resolve_portal_tables_dir(core),
        db_path=get_lake(core).blend_db_path,
        use_unified_agent=use_unified_agent,
        retrieval=retrieval,
        experiment=resolved_config,
    )


def pipeline_deps() -> PipelineDeps:
    """The phases, resources and sinks this surface runs the engine with.

    Built from this module's own names at the start of a run, so a test that
    replaces one here replaces what the engine calls.

    Returns:
        The dependencies, bound to this module's names.
    """
    return PipelineDeps(
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
        save_experiment_log=save_experiment_log,
        log_retrieval_decision=log_retrieval_decision,
        persist_manifest=persist_manifest,
    )


def run_question(
    question: str,
    runtime: RuntimeSettings,
    *,
    question_id: str | int | None = None,
    log_context: Mapping[str, Any] | None = None,
    manifest: ExperimentManifest | None = None,
    replay: Replay | None = None,
    deps: PipelineDeps | None = None,
) -> QueryResult:
    """Run the autonomous LakeGen workflow for one question.

    Args:
        question: The question.
        runtime: The run's runtime settings.
        question_id: The question's id, if it has one.
        log_context: The benchmark fields of the question, if any.
        manifest: The run's manifest; one is created when None.
        replay: What an earlier run recorded that this run starts from.
        deps: The phases and resources the run calls. Defaults to this
            module's ``pipeline_deps()``: the real phases and resources, or
            what a test put in their place here.

    Returns:
        The run's result.
    """
    return run_pipeline(
        question,
        RunContext.from_runtime(runtime),
        deps or pipeline_deps(),
        question_id=question_id,
        log_context=log_context,
        manifest=manifest,
        replay=replay,
    )
