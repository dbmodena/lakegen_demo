"""
LakeGen CLI - Run the full workflow from the terminal.

Basic example (using the default model and core)
uv run python src/cli.py "What are the top 3 districts by student suspensions?"

Using a specific core:
uv run python src/cli.py --core bologna "Quali sono le zone più inquinate?"

Using a specific core and model
uv run python src/cli.py --core nyc --model qwen3.5:latest "How many parks are there?"

Using a custom Ollama URL
uv run python src/cli.py --core bologna --ollama-url http://192.168.1.10:11434 "Quali sono le zone più inquinate?"

Using the divided architecture (separate Phase 1 & 2) instead of the default unified architecture:
uv run python src/cli.py --divided "What are the top 3 districts by student suspensions?"
"""

import argparse
import sys
import uuid
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from lakegen.core.bootstrap import bootstrap_nltk_data
from lakegen.core.resources import get_all_csv_files, get_llm, get_prompt_manager, get_solr
from lakegen.phases import (
    phase1_generate_keywords,
    phase2_select_tables,
    phase12_agent,
    phase3_generate_and_execute,
    phase4_synthesize,
)
from lakegen.core.logger import save_experiment_log
from lakegen.core.config import BASE_DIR
from lakegen.ui.state import RuntimeSettings, SOLR_CORE_OPTIONS, MODEL_OPTIONS


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


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{prompt} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "si", "sì")


def _ask_input(prompt: str, allow_empty: bool = True) -> str:
    value = input(f"{prompt}: ").strip()
    if not allow_empty and not value:
        return _ask_input(prompt, allow_empty)
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
    llm, _token_counter = get_llm(runtime.model_name, runtime.ollama_url)
    solr = get_solr(runtime.solr_core)
    pm = get_prompt_manager()
    all_csv = get_all_csv_files(runtime.csv_dir)

    if not all_csv:
        print(_c(f"No CSV files found in {runtime.csv_dir}", "red"))
        return

    print(_c(f"Portal: {runtime.portal_name}", "cyan"))
    print(_c(f"Model:  {runtime.model_name}", "cyan"))
    print(_c(f"CSVs:   {len(all_csv)} files", "cyan"))

    tokens = {"p1": 0, "p2": 0, "p3": 0, "p4": 0}
    keyword_hint = ""
    attempted_keywords = []
    run_id = uuid.uuid4().hex
    keywords = []
    selected = []
    solr_meta = {}
    reasoning = ""
    trace = ""

    # ── Phase 1 & 2 ────────────
    keyword_retries = 0
    while True:
        if runtime.use_unified_agent:
            _header("Phase 1 & 2 – Unified Architect & Search")

            selected, keywords, solr_meta, reasoning, trace, tok = phase12_agent(
                query=question,
                llm=llm,
                pm=pm,
                all_files=all_csv,
                solr_client=solr,
                csv_dir=runtime.csv_dir,
                hint=keyword_hint,
                portal_name=runtime.portal_name,
                stream_callback=_stream_to_terminal,
            )

            tokens["p1"] += tok

            print(f"\n\n{_c('Keywords:', 'bold')} {_keyword_list(keywords)}")
            print(f"{_c('Tables:', 'bold')}   {', '.join(selected) if selected else '(none)'}")
            print(f"{_c('Reasoning:', 'bold')} {reasoning}")
            print(f"{_c('Tokens:', 'dim')}    {tok}")

            if _ask_yes_no(f"\n{_c('Approve this selection?', 'yellow')}"):
                break

            keyword_hint = _ask_input("Hint for the agent (or press Enter to retry without hint)")
            continue

        # ── Two-phase flow: Keywords → Search + Judge ────────────
        # ── Phase 1: Generate AND keywords ────────────────────────────
        _header(f"Phase 1 – Keyword Selection (AND logic)")

        keywords, raw_content, tok1, reasoning_p1 = phase1_generate_keywords(
            query=question,
            llm=llm,
            pm=pm,
            hint=keyword_hint,
            portal_name=runtime.portal_name,
            avoid_keywords=attempted_keywords,
        )
        tokens["p1"] += tok1

        print(f"\n\n{_c('AND Keywords:', 'bold')} {_keyword_list(keywords)}")
        print(f"{_c('Tokens:', 'dim')}    {tok1}")

        if _ask_yes_no(f"\n{_c('Approve these keywords?', 'yellow')}"):
            pass  # proceed to Phase 2
        else:
            attempted_keywords.extend(keywords)
            attempted_keywords = list(dict.fromkeys(attempted_keywords))
            keyword_hint = _ask_input("Hint for keyword selection (or press Enter to retry without hint)")
            continue

        # ── Phase 2: Solr AND search + table judge ────────────────────
        _header("Phase 2 – Table Search & Selection (Solr AND)")

        selected, candidates, solr_meta, reasoning, trace, tok2 = phase2_select_tables(
            query=question,
            llm=llm,
            pm=pm,
            all_files=all_csv,
            keywords=keywords,
            solr_client=solr,
            csv_dir=runtime.csv_dir,
            hint=keyword_hint,
            portal_name=runtime.portal_name,
            stream_callback=_stream_to_terminal,
        )
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
                selected = candidates[:3] if candidates else all_csv[:3]
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

        if _ask_yes_no(f"\n{_c('Approve this selection?', 'yellow')}"):
            break

        attempted_keywords.extend(keywords)
        attempted_keywords = list(dict.fromkeys(attempted_keywords))
        keyword_hint = _ask_input("Hint for the agent (or press Enter to retry)")



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
        )

        tokens["p3"] += phase3_result.tokens
        final_code = phase3_result.clean_code
        raw_result = phase3_result.raw_result
        err = phase3_result.error

        if phase3_result.rejected_reason:
            print(_c(f"\n⚠ Tables rejected: {phase3_result.rejected_reason}", "yellow"))
            if _ask_yes_no("Force execution anyway?", default=False):
                force_execution = True
                continue
            # Re-run Phase 1 & 2
            attempted_keywords.extend(keywords)
            attempted_keywords = list(dict.fromkeys(attempted_keywords))
            keyword_hint = f"Previous tables rejected by coder: {phase3_result.rejected_reason}"
            if runtime.use_unified_agent:
                _header("Re-running Phase 1 & 2 (Unified)")
                selected, keywords, solr_meta, reasoning, trace, tok = phase12_agent(
                    query=question,
                    llm=llm,
                    pm=pm,
                    all_files=all_csv,
                    solr_client=solr,
                    csv_dir=runtime.csv_dir,
                    hint=keyword_hint,
                    portal_name=runtime.portal_name,
                    stream_callback=_stream_to_terminal,
                )
                tokens["p1"] += tok
                architect_reasoning = reasoning
            else:
                _header("Re-running Phase 1 – Keyword Selection")
                keywords, raw_content, tok1, reasoning_p1 = phase1_generate_keywords(
                    query=question,
                    llm=llm,
                    pm=pm,
                    hint=keyword_hint,
                    portal_name=runtime.portal_name,
                    avoid_keywords=attempted_keywords,
                )
                tokens["p1"] += tok1

                _header("Re-running Phase 2 – Table Search & Selection")
                selected, candidates, solr_meta, reasoning, trace, tok2 = phase2_select_tables(
                    query=question,
                    llm=llm,
                    pm=pm,
                    all_files=all_csv,
                    keywords=keywords,
                    solr_client=solr,
                    csv_dir=runtime.csv_dir,
                    hint=keyword_hint,
                    portal_name=runtime.portal_name,
                    stream_callback=_stream_to_terminal,
                )
                tokens["p2"] += tok2
                architect_reasoning = reasoning
            print(f"\n{_c('New tables:', 'bold')} {', '.join(selected)}")
            if not _ask_yes_no("Approve?"):
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
    answer, tok4 = phase4_synthesize(question, raw_result, llm, pm)
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
    )
    print(_c("\nExperiment log saved.", "dim"))


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LakeGen CLI – run the full workflow from the terminal."
    )
    parser.add_argument("question", help="The natural-language question to answer.")
    parser.add_argument(
        "--core",
        choices=SOLR_CORE_OPTIONS,
        default=SOLR_CORE_OPTIONS[0],
        help=f"Solr core / dataset portal (default: {SOLR_CORE_OPTIONS[0]}).",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_OPTIONS,
        default=MODEL_OPTIONS[0],
        help=f"Ollama model name (default: {MODEL_OPTIONS[0]}).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama server URL (default: http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--divided",
        dest="unified",
        action="store_false",
        default=True,
        help="Use the divided agent (separate Phase 1 & 2) instead of the default unified agent.",
    )
    args = parser.parse_args()

    nltk_err = bootstrap_nltk_data()
    if nltk_err:
        print(_c(f"NLTK error: {nltk_err}", "red"))
        sys.exit(1)

    runtime = RuntimeSettings(
        ollama_url=args.ollama_url,
        model_name=args.model,
        solr_core=args.core,
        csv_dir=BASE_DIR / f"data/{args.core}/datasets/csv",
        db_path=BASE_DIR / f"data/blend_{args.core}.db",
        use_unified_agent=args.unified,
    )

    run_cli_workflow(args.question, runtime)


if __name__ == "__main__":
    main()
