"""``lakegen ask``: one question through the workflow, step by step in the terminal.

Basic example (using the default model and core)
lakegen ask "What are the top 3 districts by student suspensions?"

Using a specific core:
lakegen ask --core bologna "Which areas are the most polluted?"

Using a specific core and model
lakegen ask --core nyc --model openai.gpt-oss-120b "How many parks are there?"

Using the divided architecture (separate Phase 1 & 2) instead of the default unified architecture:
lakegen ask --divided "What are the top 3 districts by student suspensions?"

``python src/cli.py`` still runs the same command.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lakegen.config.bootstrap import bootstrap_nltk_data
from lakegen.config.experiment import InteractionMode, load_experiment_config
from lakegen.config.files import parse_set_values
from lakegen.config.locations import resolve_portal_tables_dir
from lakegen.config.registry import get_lake
from lakegen.config.runtime import MODEL_OPTIONS, SOLR_CORE_OPTIONS, RuntimeSettings
from lakegen.discovery.context import prepare_discovery_context  # noqa: F401
from lakegen.discovery.orchestrated import (  # noqa: F401
    run_unified_orchestrated_discovery,
    select_from_prepared_context,
)
from lakegen.evaluation.semantic_judge import judge_semantic_code_result  # noqa: F401
from lakegen.interfaces.cli import terminal
from lakegen.interfaces.cli.driver import TerminalDriver
from lakegen.lakes import get_all_table_files  # noqa: F401
from lakegen.llm import get_llm  # noqa: F401

# The phases, resources and sinks the engine calls; pipeline_deps() binds
# them when a run starts, so a test patching a name here patches the engine.
from lakegen.observability.csv_log import save_experiment_log  # noqa: F401
from lakegen.observability.manifest import persist_manifest  # noqa: F401
from lakegen.phases import (  # noqa: F401
    phase1_generate_keywords,
    phase2_select_tables,
    phase3_generate_and_execute,
    phase4_synthesize,
    phase12_agent,
)
from lakegen.pipeline.context import MemoryScope, RunContext, SurfaceOptions
from lakegen.pipeline.deps import PipelineDeps
from lakegen.pipeline.engine import MAX_CODE_ATTEMPTS, MAX_TABLE_ATTEMPTS, Pipeline
from lakegen.prompts import get_prompt_manager  # noqa: F401
from lakegen.retrieval import DEFAULT_TOP_K, RetrievalMode
from lakegen.retrieval.factory import get_solr  # noqa: F401
from lakegen.retrieval.run_log import log_retrieval_decision  # noqa: F401
from lakegen.runner import ExperimentRunner

# ── Helpers ──────────────────────────────────────────────────────────────────
# The prompts live in lakegen.interfaces.cli.terminal; these names stay for the
# tests and callers that import them from here.

COLORS = terminal.COLORS
_c = terminal.colour
_header = terminal.header
_ask_yes_no = terminal.ask_yes_no
_ask_input = terminal.ask_input
_keyword_list = terminal.keyword_list

# ── Workflow ─────────────────────────────────────────────────────────────────

MAX_RETRIES = MAX_CODE_ATTEMPTS
MAX_KEYWORD_RETRIES = MAX_TABLE_ATTEMPTS

def pipeline_deps() -> PipelineDeps:
    """The phases, resources and sinks this surface runs the engine with.

    Built from this module's own names at the start of a run, so a test that
    replaces one here replaces what the engine calls.
    """
    return PipelineDeps(
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
        save_experiment_log=save_experiment_log,
        log_retrieval_decision=log_retrieval_decision,
        persist_manifest=persist_manifest,
    )


def run_cli_workflow(question: str, runtime: RuntimeSettings, deps: PipelineDeps | None = None) -> None:
    """Run one question step by step in the terminal, on the batch service's engine."""
    ctx = RunContext.from_runtime(
        runtime,
        SurfaceOptions(human_gated=True, memory_scope=MemoryScope.NONE, answer_on_failure=True),
    )
    pipeline = Pipeline(ctx, deps or pipeline_deps())
    try:
        TerminalDriver(pipeline, ctx).run(question)
    except RuntimeError as exc:
        print(_c(str(exc), "red"))


def resolve_cli_experiment(
    *,
    config_path: Path | None = None,
    core: str | None = None,
    model: str | None = None,
    unified: bool | None = None,
    retrieval_mode: str | None = None,
    top_k: int | None = None,
    hybrid_alpha: float | None = None,
    candidate_multiplier: int | None = None,
    set_values: list[str] | None = None,
):
    """Translate CLI options into the canonical experiment configuration."""

    overrides = {}
    if config_path is None:
        # Preserve the CLI defaults even when LAKEGEN_* retrieval
        # environment variables are present; files remain authoritative.
        overrides.update({
            "core": SOLR_CORE_OPTIONS[0],
            "model": MODEL_OPTIONS[0],
            "discovery_architecture": "unified",
            "retrieval.mode": RetrievalMode.KEYWORD.value,
            "retrieval.top_k": DEFAULT_TOP_K,
            "retrieval.alpha": 0.5,
            "retrieval.candidate_multiplier": 5,
        })
    cli_values = {
        "core": core,
        "model": model,
        "discovery_architecture": (
            "unified" if unified is True else "divided" if unified is False else None
        ),
        "retrieval.mode": retrieval_mode,
        "retrieval.top_k": top_k,
        "retrieval.alpha": hybrid_alpha,
        "retrieval.candidate_multiplier": candidate_multiplier,
    }
    overrides.update({key: value for key, value in cli_values.items() if value is not None})
    overrides.update(parse_set_values(set_values))
    return load_experiment_config(config_path, overrides=overrides)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lakegen ask",
        description="LakeGen CLI – run the full workflow from the terminal.",
    )
    parser.add_argument("question", help="The natural-language question to answer.")
    parser.add_argument(
        "--core",
        choices=SOLR_CORE_OPTIONS,
        default=None,
        help=f"Solr core / dataset portal (default: {SOLR_CORE_OPTIONS[0]}).",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_OPTIONS,
        default=None,
        help=f"OCI Generative AI model name (default: {MODEL_OPTIONS[0]}).",
    )
    parser.add_argument(
        "--divided",
        dest="unified",
        action="store_false",
        default=None,
        help="Use the divided agent (separate Phase 1 & 2) instead of the default unified agent.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=[mode.value for mode in RetrievalMode],
        default=None,
        help="Table retriever: keyword BM25, semantic KNN, or hybrid fusion.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--hybrid-alpha", type=float, default=None)
    parser.add_argument("--candidate-multiplier", type=int, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML or JSON experiment configuration file.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Override one config value (repeatable), e.g. retrieval.top_k=20.",
    )
    args = parser.parse_args(argv)

    nltk_err = bootstrap_nltk_data()
    if nltk_err:
        print(_c(f"NLTK error: {nltk_err}", "red"))
        sys.exit(1)

    try:
        experiment = resolve_cli_experiment(
            config_path=args.config,
            core=args.core,
            model=args.model,
            unified=args.unified,
            retrieval_mode=args.retrieval_mode,
            top_k=args.top_k,
            hybrid_alpha=args.hybrid_alpha,
            candidate_multiplier=args.candidate_multiplier,
            set_values=args.set,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    runtime = RuntimeSettings(
        model_name=experiment.model,
        solr_core=experiment.core,
        csv_dir=resolve_portal_tables_dir(experiment.core),
        db_path=get_lake(experiment.core).blend_db_path,
        use_unified_agent=experiment.use_unified_agent,
        retrieval=experiment.retrieval.to_runtime(),
        experiment=experiment,
    )

    if experiment.interaction_mode == InteractionMode.AUTONOMOUS:
        result = ExperimentRunner(experiment).run(args.question)
        if result.answer:
            print(result.answer)
        elif result.error:
            print(_c(result.error, "red"))
        return
    run_cli_workflow(args.question, runtime)


if __name__ == "__main__":
    main()
