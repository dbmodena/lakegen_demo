"""
LakeGen CLI - Run the full workflow from the terminal.

Basic example (using the default model and core)
uv run python src/cli.py "What are the top 3 districts by student suspensions?"

Using a specific core:
uv run python src/cli.py --core bologna "Which areas are the most polluted?"

Using a specific core and model
uv run python src/cli.py --core nyc --model openai.gpt-oss-120b "How many parks are there?"

Using the divided architecture (separate Phase 1 & 2) instead of the default unified architecture:
uv run python src/cli.py --divided "What are the top 3 districts by student suspensions?"
"""

import argparse
import sys
import uuid
import time
from pathlib import Path
import yaml

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from lakegen.core.bootstrap import bootstrap_nltk_data
from lakegen.core.resources import get_all_table_files, get_llm, get_prompt_manager, get_solr
from lakegen.phases import (
    phase1_generate_keywords,
    phase2_select_tables,
    phase12_agent,
    phase3_generate_and_execute,
    phase4_synthesize,
)
from lakegen.core.logger import save_experiment_log
from lakegen.core.config import BASE_DIR, LOG_DIR, resolve_portal_tables_dir
from lakegen.ui.state import RuntimeSettings, SOLR_CORE_OPTIONS, MODEL_OPTIONS
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.experiment_config import load_experiment_config
from lakegen.manifest import create_manifest, persist_manifest
from lakegen.reproducibility import initialize_reproducibility
from lakegen.tracing import (
    HumanGate,
    HumanInterventionRecorder,
    PhaseName,
    build_llm_phase_records,
    summarize_tool_calls,
    normalize_hint,
)
from lakegen.runner import ExperimentRunner
from lakegen.experiment_config import InteractionMode


# ── Helpers ──────────────────────────────────────────────────────────────────

COLORS = {
    "bold":    "\033[1m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "cyan":    "\033[96m",
    "red":     "\033[91m",
    "reset":   "\033[0m",
    "dim":     "\033[2m",
}


def _c(text: str, color: str) -> str:
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(_c(f"  {title}", "bold"))
    print(f"{'─' * 60}")


def _ask_yes_no(
    prompt: str,
    default: bool = True,
    *,
    recorder: HumanInterventionRecorder,
    phase: PhaseName,
    gate: HumanGate,
) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    started = time.monotonic()
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        approved = default
    else:
        approved = answer in ("y", "yes", "si", "sì")
    recorder.record_approval(
        phase=phase,
        gate=gate,
        approved=approved,
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    return approved


def _ask_input(
    prompt: str,
    allow_empty: bool = True,
    *,
    recorder: HumanInterventionRecorder,
    phase: PhaseName,
    gate: HumanGate,
) -> str:
    started = time.monotonic()
    value = normalize_hint(input(f"{prompt}: "))
    recorder.record_hint(
        phase=phase,
        gate=gate,
        provided=bool(value),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    if not allow_empty and not value:
        return _ask_input(
            prompt,
            allow_empty,
            recorder=recorder,
            phase=phase,
            gate=gate,
        )
    return value


def _keyword_list(keywords: list[str]) -> str:
    return ", ".join(keywords) if keywords else "(none)"


# ── Workflow ─────────────────────────────────────────────────────────────────

MAX_RETRIES = 3
MAX_KEYWORD_RETRIES = 3


def _stream_to_terminal(delta: str) -> None:
    """Stream callback that prints directly to stdout."""
    print(delta, end="", flush=True)


def run_cli_workflow(question: str, runtime: RuntimeSettings) -> None:
    workflow_started = time.monotonic()
    reproducibility = initialize_reproducibility(runtime.experiment.seed)
    interventions = HumanInterventionRecorder()
    llm, _token_counter = get_llm(runtime.model_name)
    solr = get_solr(runtime.solr_core)
    pm = get_prompt_manager()
    all_files = get_all_table_files(runtime.csv_dir)

    if not all_files:
        print(_c(f"No CSV or Parquet files found in {runtime.csv_dir}", "red"))
        return

    print(_c(f"Portal: {runtime.portal_name}", "cyan"))
    print(_c(f"Model:  {runtime.model_name}", "cyan"))
    print(_c(f"Tables: {len(all_files)} files", "cyan"))

    tokens = {"p1": 0, "p2": 0, "p3": 0, "p4": 0}
    keyword_hint = ""
    attempted_keywords = []
    run_id = uuid.uuid4().hex
    manifest = create_manifest(
        runtime.experiment,
        base_dir=BASE_DIR,
        question=question,
        run_id=run_id,
    )
    persist_manifest(manifest, LOG_DIR / "manifests")
    keywords = []
    selected = []
    solr_meta = {}
    reasoning = ""
    trace = ""
    phase_seconds = {"discovery": 0.0, "code": 0.0, "result": 0.0}
    llm_call_counts = {"discovery": 0, "code": 0, "result": 0}

    # ── Phase 1 & 2 ────────────
    keyword_retries = 0
    while True:
        if runtime.use_unified_agent:
            _header("Phase 1 & 2 – Unified Architect & Search")

            phase_started = time.monotonic()
            selected, keywords, solr_meta, reasoning, trace, tok = phase12_agent(
                query=question,
                llm=llm,
                pm=pm,
                all_files=all_files,
                solr_client=solr,
                csv_dir=runtime.csv_dir,
                hint=keyword_hint,
                portal_name=runtime.portal_name,
                stream_callback=_stream_to_terminal,
                retrieval_config=runtime.retrieval,
            )
            phase_seconds["discovery"] += time.monotonic() - phase_started
            llm_call_counts["discovery"] += 1

            tokens["p1"] += tok

            print(f"\n\n{_c('Keywords:', 'bold')} {_keyword_list(keywords)}")
            print(f"{_c('Tables:', 'bold')}   {', '.join(selected) if selected else '(none)'}")
            print(f"{_c('Reasoning:', 'bold')} {reasoning}")
            print(f"{_c('Tokens:', 'dim')}    {tok}")

            if _ask_yes_no(
                f"\n{_c('Approve this selection?', 'yellow')}",
                recorder=interventions,
                phase="discovery",
                gate=HumanGate.DATASET_APPROVAL,
            ):
                break

            keyword_hint = _ask_input(
                "Hint for the agent (or press Enter to retry without hint)",
                recorder=interventions,
                phase="discovery",
                gate=HumanGate.DATASET_HINT,
            )
            continue

        # ── Two-phase flow: Keywords → Search + Judge ────────────
        # ── Phase 1: Generate AND keywords ────────────────────────────
        _header(f"Phase 1 – Keyword Selection (AND logic)")

        phase_started = time.monotonic()
        keywords, raw_content, tok1, reasoning_p1 = phase1_generate_keywords(
            query=question,
            llm=llm,
            pm=pm,
            hint=keyword_hint,
            portal_name=runtime.portal_name,
            avoid_keywords=attempted_keywords,
        )
        phase_seconds["discovery"] += time.monotonic() - phase_started
        llm_call_counts["discovery"] += 1
        tokens["p1"] += tok1

        print(f"\n\n{_c('AND Keywords:', 'bold')} {_keyword_list(keywords)}")
        print(f"{_c('Tokens:', 'dim')}    {tok1}")

        if _ask_yes_no(
            f"\n{_c('Approve these keywords?', 'yellow')}",
            recorder=interventions,
            phase="discovery",
            gate=HumanGate.KEYWORD_APPROVAL,
        ):
            pass  # proceed to Phase 2
        else:
            attempted_keywords.extend(keywords)
            attempted_keywords = list(dict.fromkeys(attempted_keywords))
            keyword_hint = _ask_input(
                "Hint for keyword selection (or press Enter to retry without hint)",
                recorder=interventions,
                phase="discovery",
                gate=HumanGate.KEYWORD_HINT,
            )
            continue

        # ── Phase 2: configured retrieval + table judge ──────────────
        _header(
            "Phase 2 – Table Search & Selection "
            f"({runtime.retrieval.mode})"
        )

        phase_started = time.monotonic()
        selected, candidates, solr_meta, reasoning, trace, tok2 = phase2_select_tables(
            query=question,
            llm=llm,
            pm=pm,
            all_files=all_files,
            keywords=keywords,
            solr_client=solr,
            csv_dir=runtime.csv_dir,
            hint=keyword_hint,
            portal_name=runtime.portal_name,
            stream_callback=_stream_to_terminal,
            retrieval_config=runtime.retrieval,
        )
        phase_seconds["discovery"] += time.monotonic() - phase_started
        llm_call_counts["discovery"] += 1
        tokens["p2"] += tok2

        # Check if Phase 2 rejected the keywords
        if reasoning.startswith("REJECT_KEYWORDS:"):
            reject_reason = reasoning.replace("REJECT_KEYWORDS:", "").strip()
            print(_c(f"\n⚠ Keywords rejected by table judge: {reject_reason}", "yellow"))
            attempted_keywords.extend(keywords)
            attempted_keywords = list(dict.fromkeys(attempted_keywords))
            keyword_retries += 1
            if keyword_retries >= MAX_KEYWORD_RETRIES:
                print(_c(f"Max keyword retries ({MAX_KEYWORD_RETRIES}) reached. Using best available.", "red"))
                selected = candidates[:3] if candidates else all_files[:3]
                reasoning = f"Fallback after {MAX_KEYWORD_RETRIES} keyword rejections."
                break
            keyword_hint = (
                f"The previous keywords led to bad tables. "
                f"Architect feedback: {reject_reason}. "
                f"Generate completely different keywords."
            )
            continue

        print(f"\n\n{_c('Keywords:', 'bold')} {_keyword_list(keywords)}")
        print(f"{_c('Tables:', 'bold')}   {', '.join(selected) if selected else '(none)'}")
        print(f"{_c('Reasoning:', 'bold')} {reasoning}")
        print(f"{_c('Tokens:', 'dim')}    {tok2}")

        if _ask_yes_no(
            f"\n{_c('Approve this selection?', 'yellow')}",
            recorder=interventions,
            phase="discovery",
            gate=HumanGate.DATASET_APPROVAL,
        ):
            break

        attempted_keywords.extend(keywords)
        attempted_keywords = list(dict.fromkeys(attempted_keywords))
        keyword_hint = _ask_input(
            "Hint for the agent (or press Enter to retry)",
            recorder=interventions,
            phase="discovery",
            gate=HumanGate.DATASET_HINT,
        )



    # ── Phase 3 – Code Generation & Execution ────────────────────────
    candidates = selected
    architect_reasoning = reasoning
    force_execution = False
    run_dir = BASE_DIR / "coding" / run_id

    retries = 0
    error_msg = ""
    final_code = ""
    raw_result = None
    err = None

    while True:
        _header(f"Phase 3 – Code Generation (attempt {retries + 1}/{MAX_RETRIES})")

        phase_started = time.monotonic()
        phase3_result = phase3_generate_and_execute(
            question,
            selected,
            candidates,
            solr_meta,
            architect_reasoning,
            llm,
            pm,
            runtime.csv_dir,
            retries=retries,
            error_msg=error_msg,
            force_execution=force_execution,
            stream_placeholder=None,
            reasoning_placeholder=None,
            run_dir=run_dir,
            seed=reproducibility.effective_seed,
        )
        phase_seconds["code"] += time.monotonic() - phase_started
        llm_call_counts["code"] += 1

        tokens["p3"] += phase3_result.tokens
        final_code = phase3_result.clean_code
        raw_result = phase3_result.raw_result
        err = phase3_result.error

        if phase3_result.rejected_reason:
            print(_c(f"\n⚠ Tables rejected: {phase3_result.rejected_reason}", "yellow"))
            if _ask_yes_no(
                "Force execution anyway?",
                default=False,
                recorder=interventions,
                phase="code",
                gate=HumanGate.FORCE_EXECUTION_CONFIRMATION,
            ):
                force_execution = True
                continue
            # Re-run Phase 1 & 2
            attempted_keywords.extend(keywords)
            attempted_keywords = list(dict.fromkeys(attempted_keywords))
            keyword_hint = f"Previous tables rejected by coder: {phase3_result.rejected_reason}"
            if runtime.use_unified_agent:
                _header("Re-running Phase 1 & 2 (Unified)")
                phase_started = time.monotonic()
                selected, keywords, solr_meta, reasoning, trace, tok = phase12_agent(
                    query=question,
                    llm=llm,
                    pm=pm,
                    all_files=all_files,
                    solr_client=solr,
                    csv_dir=runtime.csv_dir,
                    hint=keyword_hint,
                    portal_name=runtime.portal_name,
                    stream_callback=_stream_to_terminal,
                    retrieval_config=runtime.retrieval,
                )
                phase_seconds["discovery"] += time.monotonic() - phase_started
                llm_call_counts["discovery"] += 1
                tokens["p1"] += tok
                architect_reasoning = reasoning
            else:
                _header("Re-running Phase 1 – Keyword Selection")
                phase_started = time.monotonic()
                keywords, raw_content, tok1, reasoning_p1 = phase1_generate_keywords(
                    query=question,
                    llm=llm,
                    pm=pm,
                    hint=keyword_hint,
                    portal_name=runtime.portal_name,
                    avoid_keywords=attempted_keywords,
                )
                phase_seconds["discovery"] += time.monotonic() - phase_started
                llm_call_counts["discovery"] += 1
                tokens["p1"] += tok1

                _header("Re-running Phase 2 – Table Search & Selection")
                phase_started = time.monotonic()
                selected, candidates, solr_meta, reasoning, trace, tok2 = phase2_select_tables(
                    query=question,
                    llm=llm,
                    pm=pm,
                    all_files=all_files,
                    keywords=keywords,
                    solr_client=solr,
                    csv_dir=runtime.csv_dir,
                    hint=keyword_hint,
                    portal_name=runtime.portal_name,
                    stream_callback=_stream_to_terminal,
                    retrieval_config=runtime.retrieval,
                )
                phase_seconds["discovery"] += time.monotonic() - phase_started
                llm_call_counts["discovery"] += 1
                tokens["p2"] += tok2
                architect_reasoning = reasoning
            print(f"\n{_c('New tables:', 'bold')} {', '.join(selected)}")
            if not _ask_yes_no(
                "Approve?",
                recorder=interventions,
                phase="discovery",
                gate=HumanGate.DATASET_APPROVAL,
            ):
                print(_c("Workflow cancelled.", "red"))
                return
            retries = 0
            error_msg = ""
            force_execution = False
            continue

        if err is None:
            print(_c("\n✓ Code executed successfully!", "green"))
            print(f"{_c('Output:', 'bold')}\n{raw_result}")
            break

        print(_c(f"\n✗ Execution error: {err}", "red"))
        retries += 1
        if retries >= MAX_RETRIES:
            print(_c(f"Failed after {MAX_RETRIES} attempts.", "red"))
            break
        error_msg = err
        print(f"Retrying ({retries}/{MAX_RETRIES})...")

    if raw_result is None:
        raw_result = f"Execution failed after {MAX_RETRIES} attempts. Last error: {error_msg}"

    # ── Phase 4 – Synthesis ──────────────────────────────────────────────
    _header("Phase 4 – Answer Synthesis")
    phase_started = time.monotonic()
    answer, tok4 = phase4_synthesize(question, raw_result, llm, pm)
    phase_seconds["result"] += time.monotonic() - phase_started
    llm_call_counts["result"] += 1
    tokens["p4"] = tok4

    print(f"\n{_c('Final Answer:', 'bold')}")
    print(answer)

    # ── Summary ──────────────────────────────────────────────────────────
    _header("Summary")
    print(f"  Keywords:  {_keyword_list(keywords)}")
    print(f"  Tables:    {', '.join(selected)}")
    print(f"  Tokens:    P1={tokens['p1']}  P2={tokens['p2']}  P3={tokens['p3']}  P4={tokens['p4']}")
    print(f"  Total:     {sum(tokens.values())}")

    save_experiment_log(
        question=question,
        code=final_code,
        result=raw_result if raw_result else "",
        retries=retries,
        reasoning=architect_reasoning,
        tables=selected,
        raw_keywords="",
        final_keywords=keywords,
        debug_raw="",
        final_result=answer,
        full_trace=trace,
        tokens_phase1=tokens["p1"],
        tokens_phase2=tokens["p2"],
        tokens_phase3=tokens["p3"],
        tokens_phase4=tokens["p4"],
        error=err if err is not None else "",
        model=runtime.model_name,
        architecture=runtime.experiment.architecture_name,
        elapsed_seconds=round(time.monotonic() - workflow_started, 3),
        extra_fields={
            "MANIFEST_JSON": manifest.model_dump(mode="json"),
            "RUN_TRACE_JSON": {
                "discovery": {"keywords": keywords, "selected_datasets": selected},
                "llm_calls": build_llm_phase_records(
                    total_tokens={
                        "discovery": tokens["p1"] + tokens["p2"],
                        "code": tokens["p3"],
                        "result": tokens["p4"],
                    },
                    phase_invocations=llm_call_counts,
                ),
                "phase_metrics": {
                    phase: {"latency_seconds": round(seconds, 6)}
                    for phase, seconds in phase_seconds.items()
                },
                "tool_calls": summarize_tool_calls(trace),
                "retries": retries,
                "errors": [str(err)] if err else [],
                "code": final_code,
                "execution_outcome": {
                    "succeeded": err is None,
                    "raw_result": raw_result,
                },
                "human_interventions": interventions.to_list(),
                "configuration": manifest.resolved_config,
                "reproducibility": reproducibility.telemetry(
                    generated_code_instructions_applied=True
                ),
            },
        },
    )
    print(_c("\nExperiment log saved.", "dim"))


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
        # Preserve the historical CLI defaults even when LAKEGEN_* retrieval
        # environment variables are present; files remain authoritative.
        overrides.update({
            "core": SOLR_CORE_OPTIONS[0],
            "model": MODEL_OPTIONS[0],
            "discovery_architecture": "unified",
            "retrieval.mode": RetrievalMode.KEYWORD.value,
            "retrieval.top_k": 10,
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
    for raw_override in set_values or []:
        if "=" not in raw_override:
            raise ValueError(f"invalid --set {raw_override!r}; expected PATH=VALUE")
        key, raw_value = raw_override.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw_value)
    return load_experiment_config(config_path, overrides=overrides)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LakeGen CLI – run the full workflow from the terminal."
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
    args = parser.parse_args()

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
        db_path=BASE_DIR / f"data/blend_{experiment.core}.db",
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
