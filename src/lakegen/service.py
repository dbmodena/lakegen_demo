"""Non-interactive LakeGen workflow used by API and batch clients."""

from __future__ import annotations

import logging
import sys
from typing import Any, Mapping

from lakegen.config.registry import get_lake
from lakegen.core.config import resolve_portal_tables_dir
from lakegen.experiment_config import (
    DiscoveryArchitecture,
    ExperimentConfig,
    InteractionMode,
    load_experiment_config,
)
from lakegen.manifest import ExperimentManifest
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalConfig, RetrievalMode
from lakegen.runtime import MODEL_OPTIONS, SOLR_CORE_OPTIONS, RuntimeSettings
from lakegen.service_models import QueryResult, QuestionSource, extract_questions

# The phases, resources and sinks the engine calls, imported here by name so
# that a test patching `lakegen.service.<name>` patches what the engine calls
# (PipelineDeps.from_module reads them from this module when a run starts).
from lakegen.core.logger import save_experiment_log  # noqa: F401
from lakegen.lakes import get_all_table_files  # noqa: F401
from lakegen.llm import get_llm  # noqa: F401
from lakegen.prompts import get_prompt_manager  # noqa: F401
from lakegen.retrieval.factory import get_solr  # noqa: F401
from lakegen.retrieval.run_log import log_retrieval_decision  # noqa: F401
from lakegen.manifest import persist_manifest  # noqa: F401
from lakegen.orchestrated_context import prepare_discovery_context  # noqa: F401
from lakegen.phases import (  # noqa: F401
    phase1_generate_keywords,
    phase2_select_tables,
    phase3_generate_and_execute,
    phase4_synthesize,
    phase12_agent,
)
from lakegen.discovery.orchestrated import (  # noqa: F401
    run_unified_orchestrated_discovery,
    select_from_prepared_context,
)
from lakegen.semantic_code_judge import judge_semantic_code_result  # noqa: F401

# The constants and helpers live in the engine; they are re-exported for the
# tests and callers that import them from here.
from lakegen.pipeline.context import RunContext
from lakegen.pipeline.deps import PipelineDeps
from lakegen.pipeline.engine import (  # noqa: F401
    MAX_CODE_ATTEMPTS,
    MAX_TABLE_ATTEMPTS,
    _record_semantic_plan_telemetry,
    _rejected_selection_signature,
    _selection_plan_signature,
    _selection_retry_feedback,
    run_pipeline,
)
from lakegen.pipeline.replay import Replay

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
    if core not in SOLR_CORE_OPTIONS:
        raise ValueError(
            f"Unknown core {core!r}. Expected one of: {', '.join(SOLR_CORE_OPTIONS)}"
        )
    if model not in MODEL_OPTIONS:
        raise ValueError(
            f"Unknown model {model!r}. Expected one of: {', '.join(MODEL_OPTIONS)}"
        )

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
    resolved_config = config or load_experiment_config(overrides={
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
    })
    return RuntimeSettings(
        model_name=model,
        solr_core=core,
        csv_dir=resolve_portal_tables_dir(core),
        db_path=get_lake(core).blend_db_path,
        use_unified_agent=use_unified_agent,
        retrieval=retrieval,
        experiment=resolved_config,
    )


def run_question(
    question: str,
    runtime: RuntimeSettings,
    *,
    question_id: str | int | None = None,
    log_context: Mapping[str, Any] | None = None,
    manifest: ExperimentManifest | None = None,
    replay: Replay | None = None,
) -> QueryResult:
    """Run the autonomous LakeGen workflow for one question.

    The phases and resources are taken from this module by name when the
    run starts, so a name patched here is what the engine calls.
    """
    return run_pipeline(
        question,
        RunContext.from_runtime(runtime),
        PipelineDeps.from_module(sys.modules[__name__]),
        question_id=question_id,
        log_context=log_context,
        manifest=manifest,
        replay=replay,
    )
