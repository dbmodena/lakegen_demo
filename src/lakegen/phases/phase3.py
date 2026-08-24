import os
import re
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from llama_index.core.llms import ChatMessage, LLM

from lakegen.core.types import SolrMetadata
from lakegen.experiment_config import CoderContextLevel
from prompts.prompt_manager import PromptManager
from lakegen.core.config import BASE_DIR
from lakegen.core.table_io import read_table, table_load_command
from lakegen.column_resolution import resolve_generated_code_columns
from lakegen.code_evaluation import (
    evaluation_output_instruction,
    extract_evaluation_payload,
)
from lakegen.core.token_usage import (
    extract_total_tokens,
    get_llm_token_usage,
    reset_llm_token_usage,
)

from .phase1 import split_thinking_blocks


@dataclass
class Phase3Result:
    code_raw: str
    tokens: int
    raw_result: str | None = None
    error: str | None = None
    clean_code: str = ""
    rejected_reason: str = ""
    structured_result: object | None = None
    structured_result_error: str = ""


_ERROR_PATTERNS = [
    "error:",
    "exception:",
    "traceback",
    "errno",
    "no such file",
    "filenotfounderror",
    "permissionerror",
    "modulenotfounderror",
    "importerror",
    "keyerror",
    "valueerror",
    "typeerror",
    "indexerror",
    "zerodivisionerror",
    # Generated scripts sometimes handle schema validation themselves, print a
    # failure, and return normally.  The process then exits with code 0 even
    # though no analysis result was produced, so these messages must trigger a
    # correction attempt too.
    "missing required column",
    "required column is missing",
    "required columns are missing",
]


_TABPFN_INTENT_KEYWORDS = {
    "causal": [
        "causal", "cause", "causing", "treatment effect",
        "causale", "causa", "causare", "causando", "effetto del trattamento",
        "causalité", "causalite", "causer", "provoque", "effet du traitement",
        "causalidad", "causar", "causando", "efecto del tratamiento",
    ],
    "forecasting": [
        "forecast", "forecasting", "future forecast", "next year", "next month",
        "previsione", "previsionale", "anno prossimo", "mese prossimo",
        "prévision", "prevision", "année prochaine", "mois prochain",
        "pronóstico", "pronostico", "previsión", "próximo año", "proximo ano",
    ],
    "classification": [
        "classify", "classification", "classifier", "likely", "probability",
        "classifica", "classificare", "classificazione", "probabile", "probabilità",
        "classer", "classificateur", "probable", "probabilité", "probabilite",
        "clasificar", "clasificación", "clasificacion", "clasificador", "probabilidad",
    ],
    "regression": [
        "regression", "regressor", "expected", "estimate",
        "regressione", "regressore", "atteso", "attesa", "stimare", "stima",
        "régression", "regression", "régresseur", "regresseur", "attendu", "attendue", "estimer", "estimation",
        "regresión", "regresion", "regresor", "esperado", "esperada", "estimar", "estimación", "estimacion",
    ],
    "prediction": [
        "predict", "predicts", "predicted", "predicting", "prediction", "predictive",
        "predici", "predire", "prevedi", "prevedere", "predizione", "predittivo",
        "prédire", "prédir", "predire", "predir", "prédiction", "prédictif", "predictif",
        "predecir", "predicción", "prediccion", "predictivo",
    ],
}


def _tabpfn_enabled() -> bool:
    """Return whether generated analyses may use the optional TabPFN stack."""
    return os.getenv("LAKEGEN_ENABLE_TABPFN", "").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _detect_tabpfn_intent(query: str) -> str | None:
    """Return the most specific multilingual TabPFN intent found in a query."""
    normalized_query = query.casefold()
    for intent in ("causal", "forecasting", "classification", "regression", "prediction"):
        if any(
            re.search(rf"(?<!\w){re.escape(keyword.casefold())}(?!\w)", normalized_query)
            for keyword in _TABPFN_INTENT_KEYWORDS[intent]
        ):
            return intent
    return None


def _extract_code(code_raw: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", code_raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return code_raw.replace("```python", "").replace("```", "").strip()


def _rejected_tables_reason(code_raw: str) -> str:
    if "REJECT_TABLES" not in code_raw:
        return ""
    for line in code_raw.splitlines():
        if "REJECT_TABLES" in line:
            # Extract just the reason part from the line
            return line.replace("print(", "").replace(")", "").replace('"', '').replace("'", "").replace("REJECT_TABLES:", "").replace("REJECT_TABLES", "").strip()
    return "Tables rejected by model."


def _execute_code(code_raw: str, run_dir: Path | None = None):
    code = _extract_code(code_raw)

    forbidden = ["import os", "import sys", "import shutil", "subprocess", "eval(", "exec("]
    forbidden_match = next((fragment for fragment in forbidden if fragment in code), None)
    if forbidden_match is not None:
        return (
            None,
            f"Security Error: forbidden code fragment {forbidden_match!r}. "
            "Remove it completely; use only data-analysis libraries required by the task.",
            code,
        )

    coding_dir = run_dir or BASE_DIR / "coding" / uuid.uuid4().hex
    coding_dir.mkdir(parents=True, exist_ok=True)
    fp = coding_dir / "script.py"
    fp.write_text(code, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, str(fp)],
            capture_output=True,
            text=True,
            timeout=180, # Increased from 15s to 180s to allow TabPFN to run and download weights
        )
        if result.returncode == 0:
            stdout_lower = result.stdout.lower()
            if any(pat in stdout_lower for pat in _ERROR_PATTERNS):
                return None, result.stdout.strip(), code
            return result.stdout.strip(), None, code

        detail = result.stderr.strip() or result.stdout.strip()
        
        # Enhanced Error Parsing for Pandas
        if "KeyError:" in detail:
            match = re.search(r"KeyError:\s*(.*)", detail)
            col_name = match.group(1).strip() if match else "Unknown"
            error_msg = (
                f"FATAL ERROR: KeyError for column {col_name}. "
                "You tried to use a column that does NOT exist in the dataframe. "
                "Look strictly at the AVAILABLE TABLES schema provided above, and use the EXACT column name found there. "
                f"\n\nFull Traceback:\n{detail[-500:]}"
            )
        elif "FileNotFoundError:" in detail:
            error_msg = (
                "FATAL ERROR: FileNotFoundError. "
                "You hallucinated a file path or name. "
                "Use the EXACT file paths provided in the AVAILABLE TABLES section. "
                f"\n\nFull Traceback:\n{detail[-500:]}"
            )
        else:
            error_msg = f"[Exit code {result.returncode}] {detail[-800:]}"
            
        return None, error_msg, code
    except subprocess.TimeoutExpired:
        return None, "Execution timed out (180s limit).", code
    except Exception as e:
        return None, f"[{type(e).__name__}] {e}", code


def _resolve_and_validate_columns(
    code_raw: str,
    tables: list[str],
    csv_dir: Path,
) -> tuple[str, str | None]:
    """Resolve generated aliases against the exact selected-table schemas."""

    code = _extract_code(code_raw)
    schemas: dict[str, list[str]] = {}
    try:
        for table in tables:
            path = Path(csv_dir) / table.strip()
            schemas[table] = [str(column) for column in read_table(path, nrows=0).columns]
        resolution = resolve_generated_code_columns(code, schemas)
    except (SyntaxError, ValueError) as exc:
        return code, f"Column preflight failed: {type(exc).__name__}: {exc}"

    if resolution.unresolved_required:
        available = sorted({column for columns in schemas.values() for column in columns})
        return resolution.code, (
            "Column preflight failed. Required generated column names do not match "
            f"the selected schemas: {list(resolution.unresolved_required)}. "
            f"Available exact columns: {available}"
        )
    return resolution.code, None


def _exact_column_labels(frame) -> list[str]:
    """Render the executable table schema without substituting Solr aliases."""

    return [f"{column}({frame[column].dtype})" for column in frame.columns]


def _build_coder_tables_info(
    tables: list[str],
    csv_dir: Path,
    context_level: CoderContextLevel,
) -> str:
    """Build the selected-table context exposed to the code generator."""

    max_detail_cols = 25
    max_sample_cols = 15
    max_sample_rows = 2
    max_cell_width = 40

    info_lines = ["AVAILABLE TABLES:"]
    for idx, fn in enumerate(tables, 1):
        filepath = Path(csv_dir) / fn.strip()
        load_cmd = table_load_command(filepath)
        info_lines.append(f"{idx}. LOAD: {load_cmd}")

        if context_level == CoderContextLevel.MINIMAL:
            continue

        rows_to_read = max_sample_rows + 1 if context_level == CoderContextLevel.FULL else 0
        df = read_table(filepath, nrows=rows_to_read)

        # The local file is the execution schema and therefore the sole source
        # of truth for exact names. Solr may expose normalized API aliases.
        try:
            col_typed = _exact_column_labels(df)
        except Exception:
            col_typed = ["Unknown columns"]

        total_cols = len(col_typed)
        shown = col_typed[:max_detail_cols]
        info_lines.append(f"   Columns ({total_cols}): {', '.join(shown)}")
        if total_cols > max_detail_cols:
            rest_names = [
                column.split("(")[0] if "(" in column else column
                for column in col_typed[max_detail_cols:]
            ]
            info_lines.append(
                f"   +{total_cols - max_detail_cols} more: {', '.join(rest_names[:15])}"
            )

        if context_level == CoderContextLevel.SCHEMA_ONLY:
            continue

        sample_df = df.head(max_sample_rows).copy()
        if len(sample_df.columns) > max_sample_cols:
            sample_df = sample_df.iloc[:, :max_sample_cols].copy()
        for column in sample_df.columns:
            sample_df[column] = sample_df[column].astype(str).str.slice(0, max_cell_width)
        info_lines.append(f"   Sample:\n{sample_df.to_string(index=False)}")

    return "\n".join(info_lines)


def phase3_generate_code(
    query, 
    tables, 
    candidates, 
    solr_meta: SolrMetadata, 
    reasoning,
    llm: LLM, 
    pm: PromptManager, 
    csv_dir, 
    retries=0,
    error_msg="", 
    previous_code="",
    force_execution: bool = False,
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
    cancel_check: Callable[[], None] | None = None,
    seed: int = 0,
    seed_instruction_recorder: Callable[[], None] | None = None,
    coder_context_level: CoderContextLevel = CoderContextLevel.FULL,
    evaluation_result_type: str | None = None,
):
    context_level = CoderContextLevel(coder_context_level)
    tables_info = _build_coder_tables_info(tables, Path(csv_dir), context_level)

    system_prompt = pm.render("code_generator", "system_prompt")
    if retries == 0:
        user_prompt = pm.render("code_generator", "initial_prompt",
                                question=query, arch_reasoning=reasoning,
                                tables_info=tables_info)
    else:
        user_prompt = pm.render("code_generator", "correction_prompt",
                                question=query, error_message=error_msg,
                                previous_code=previous_code,
                                arch_reasoning=reasoning,
                                tables_info=tables_info)

    user_prompt += (
        "\n\n[REPRODUCIBILITY]\n"
        f"The effective seed for this run is {seed}. Use exactly this value for "
        "every stochastic operation that accepts a seed (including pandas "
        "sampling, train/test split, cross-validation splitters, and model "
        "random_state parameters). Do not add a seed to deterministic operations. "
        "Do not use another fixed seed.\n"
    )
    if evaluation_result_type:
        user_prompt += evaluation_output_instruction(evaluation_result_type)

    # --- Optional TabPFN intent routing and task-specific hint injection ---
    tabpfn_intent = _detect_tabpfn_intent(query)
    if tabpfn_intent and not _tabpfn_enabled():
        user_prompt += (
            "\n\n[OPTIONAL ML BACKEND DISABLED]\n"
            "TabPFN is not installed in this environment. Do not import `tabpfn` or "
            "`torch`. Prefer deterministic Pandas analysis when it answers the "
            "question; otherwise use an appropriate scikit-learn estimator and "
            "clearly report its validation limits.\n"
        )
    elif tabpfn_intent:
        common_hint = (
            f"\n\n[TABPFN REQUIREMENT — {tabpfn_intent.upper()}]\n"
            f"The detected task is `{tabpfn_intent}`. You MUST use the `tabpfn` library "
            "and follow the rules for this specific intent. Use this API:\n"
            "```python\n"
            "import torch\n"
            "from tabpfn import TabPFNRegressor, TabPFNClassifier\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "# For regression: model = TabPFNRegressor(device=device)\n"
            "# For classification: model = TabPFNClassifier(device=device)\n"
            "```\n"
            "COMMON DATAFRAME AND MODELING RULES:\n"
            "1. SPATIAL ALIGNMENT (NEIGHBORHOODS/DISTRICTS): If the user asks for predictions by neighborhood/district ('nei quartieri'), you MUST aggregate and merge the datasets by neighborhood. Map station names or coordinates (latitude, longitude) of the sensors/loops to neighborhoods using spatial proximity or lookup tables. Do not just aggregate globally.\n"
            "2. TEMPORAL ALIGNMENT & FALLBACK: If merging on absolute dates (YYYY-MM-DD) results in 0 rows (e.g., due to different years like 2024 vs 2026), DO NOT fail. Fall back to merge the datasets by seasonal/weekly profiles: group both datasets by day of week (e.g., `.dt.dayofweek` or `.dt.strftime('%A')`) and/or hour/month, then merge on these profile keys.\n"
            "3. PREDICTIONS VS AVERAGES: When showing aggregated predictions (e.g. average monthly accidents per borough), DO NOT feed the average of features to the model (i.e. model.predict(X_mean)). Instead, feed the actual raw samples (X) to the model to get individual predictions, and then calculate the average of those predictions (e.g. y_pred = model.predict(X) and then calculate averages grouped by borough/district).\n"
            "4. FUTURE FORECASTS (DUPLICATION AVOIDANCE): When preparing a DataFrame for future predictions (e.g. forecasting the values of the next year for each district), make sure you only have one row per unique entity (e.g. use `.drop_duplicates()` on the district/neighborhood column). Otherwise, you will generate duplicate predictions for the same entity and sum/aggregate them incorrectly.\n"
            f"5. TABPFN DATA LIMITS (CUDA OOM AVOIDANCE): TabPFN models are strictly designed for small-to-medium datasets (up to 10,000 samples). If the prepared training dataset (X) exceeds 10,000 rows, you MUST downsample it to exactly 10,000 rows (or fewer) using `df = df.sample(n=min(10000, len(df)), random_state={seed})` and aligning X and y accordingly. This is critical to prevent CUDA Out of Memory errors.\n"
            "6. STRING-NUMERIC CLEANING: For columns containing mixed text/units and numbers (e.g. 'Strade 30 km/h o inferiore'), DO NOT use simple pd.to_numeric() with errors='coerce' directly because this turns all values into NaN. Instead, you MUST clean them by extracting the digits using regular expressions (e.g., `.str.extract(r'(\\d+)')`) before converting to numeric, ensuring features do not become constant.\n"
            "7. DATA SPLITTING: Use a reproducible 80/20 train/test split. For classification, pass `stratify=y` whenever every class has enough examples. If rows share the same entity or coordinates, use a group-aware split so duplicates cannot appear in both train and test.\n"
            "8. SMALL-DATA EVALUATION: When computationally feasible, supplement the holdout metric with at most 3-fold cross-validation. Use stratified folds for classification and group-aware folds when duplicate entities or coordinates exist. If cross-validation is not feasible, print that limitation instead of inventing a score.\n"
            "9. SPATIAL EXPLANATION: For a coordinate-based prediction, also calculate and print a concise comparison with the nearest observed locations and their target values. Treat this as supporting evidence, not as model confidence."
        )

        intent_hints = {
            "classification": (
                "\nCLASSIFICATION RULES:\n"
                "- Use TabPFNClassifier. Normalize equivalent target labels before fitting.\n"
                "- Print the predicted class and, when supported, `predict_proba` for that specific prediction.\n"
                "- Clearly label test accuracy, cross-validation results, and prediction probability as different quantities. Never describe test accuracy as the confidence of one prediction."
            ),
            "regression": (
                "\nREGRESSION RULES:\n"
                "- Use TabPFNRegressor and print the predicted numeric value with its unit.\n"
                "- Evaluate with R2 and an absolute-error metric when the test set permits it. Do not present an evaluation metric as prediction confidence."
            ),
            "forecasting": (
                "\nFORECASTING RULES:\n"
                "- Parse the time column with `errors='coerce'`, drop invalid timestamps, and ALWAYS call `sort_values(time_column).reset_index(drop=True)` before splitting, creating lags, or fitting. Never assume the input row order is chronological.\n"
                "- A date-only inclusive cutoff such as 2025-09-30 MUST include the entire day. Filter with `< cutoff + pd.Timedelta(days=1)` (or normalize dates); do not use `<= '2025-09-30'`, which excludes timestamps later that day.\n"
                "- Match the aggregation grain to the question. For an average monthly trend, aggregate observations to exact year-month periods first. Never calculate a requested month using only `dt.month == N`, because that mixes the same month across different years; filter by both year and month or by a Period value.\n"
                "- Use TabPFNRegressor with explicitly time-ordered features and leakage-safe lag values. Never randomly shuffle future observations into training, and never train on later dates to evaluate earlier dates.\n"
                "- Evaluate with a chronological holdout or walk-forward split and actually print at least one error metric (for example MAE). State the last observed date and forecast horizon. If evaluation is impossible, print that limitation explicitly.\n"
                "- Before forecasting a calendar month or season, inspect which months/seasons exist historically. Do not forecast an unseen month as though it were supported by seasonal evidence. If the next calendar period has no historical coverage, use TabPFN for a chronological backtest on the latest historically supported period, report the observed direction from the latest exact periods, and clearly state that a next-period forecast is unsupported.\n"
                "- Determine `upward` or `downward` only from comparable quantities at the same aggregation grain (for example September 2025 observed average versus October 2025 forecast). Print the exact periods and values used so the direction can be verified."
            ),
            "causal": (
                "\nCAUSAL-INFERENCE RULES:\n"
                "- Explicitly define treatment, outcome, and pre-treatment confounders. Before string conversion, drop missing values. Keep nominal categories as strings/native categories or one-hot encode them; never use LabelEncoder ordinals as numeric distances.\n"
                "- LEAKAGE CHECK: A subtype/detail column must not adjust its parent-category outcome (for example `factype` or `facsubgrp` predicting `facgroup`). Check a categorical crosstab; if one value maps to one outcome at least 95% of the time, treat it as outcome leakage, exclude it from causal features, and use it only in a separate composition table.\n"
                "- Always print raw treatment/control outcome counts, denominators, and rates before modeling. For a composition explanation, print within-group percentages, not only absolute counts.\n"
                "- Use a TabPFN outcome model only with valid non-leaking covariates. Estimate treatment=1 versus treatment=0 while preserving identical feature names and order. If no credible confounders remain, label the result an unadjusted predictive association and state that a causal effect is not identifiable.\n"
                "- Report near-zero rounded effects as `no detectable difference`, never as an increase/decrease of 0.0. Do not claim that a subtype `drives` an association unless the printed proportions support it. Always state observational limitations."
            ),
            "prediction": (
                "\nGENERIC PREDICTION RULES:\n"
                "- Infer from the target whether TabPFNClassifier or TabPFNRegressor is appropriate, then follow the corresponding evaluation and output conventions above."
            ),
        }
        user_prompt += common_hint + intent_hints[tabpfn_intent]

    original_temperature = getattr(llm, "temperature", 0.1)
    if retries > 0:
        # Dynamic temperature adjustment to break out of reasoning local minimums
        if hasattr(llm, "temperature"):
            llm.temperature = min(original_temperature + (0.1 * retries), 0.8)

    if force_execution:
        user_prompt += (
            "\n\nFORCE EXECUTION OVERRIDE\n"
            "The user explicitly chose to continue with these tables despite "
            "the previous table-quality warning. Do not return REJECT_TABLES. "
            "Write the best executable Pandas script possible using only the "
            "available paths and columns. If the data is insufficient, the "
            "script must print a concise explanation of what is missing."
        )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    if seed_instruction_recorder is not None:
        seed_instruction_recorder()

    raw_stream = ""
    structured_reasoning = ""
    tokens = 0
    reset_llm_token_usage(llm)

    def update_placeholders() -> None:
        visible_stream, tagged_reasoning = split_thinking_blocks(raw_stream)
        reasoning_parts = [
            part.strip()
            for part in (structured_reasoning, tagged_reasoning)
            if part.strip()
        ]
        reasoning_stream = "\n\n".join(reasoning_parts)

        if stream_placeholder is not None:
            stream_placeholder.markdown(visible_stream or raw_stream)
        if reasoning_placeholder is not None and reasoning_stream:
            reasoning_placeholder.markdown(reasoning_stream)

    print("[phase3 code stream] ", end="", flush=True)

    stream_kwargs = {"think": True} if stream_reasoning else {}

    loop_detected = False

    try:
        chunk_stream = llm.stream_chat(messages, **stream_kwargs)
        for chunk in chunk_stream:
            if cancel_check is not None:
                cancel_check()
            thinking_delta = chunk.additional_kwargs.get("thinking_delta")
            if thinking_delta:
                structured_reasoning += thinking_delta
                print(thinking_delta, end="", flush=True)
                cleaned_reasoning = structured_reasoning.strip()
                if len(cleaned_reasoning) > 400:
                    tail = cleaned_reasoning[-150:]
                    if cleaned_reasoning.count(tail) >= 4:
                        print("\n[!] Block repetition loop detected in reasoning. Forcing stream break.")
                        loop_detected = True
                        break
                
                update_placeholders()

            delta = chunk.delta or ""
            if delta:
                raw_stream += delta
                print(delta, end="", flush=True)
                update_placeholders()

            tokens = max(
                tokens,
                extract_total_tokens(chunk.raw),
                extract_total_tokens(chunk.additional_kwargs),
            )

    except Exception as e:
        print(f"\n[!] Error during stream: {e}")
        if raw_stream or structured_reasoning or not stream_reasoning:
            raise
        chunk_stream = llm.stream_chat(messages)
        for chunk in chunk_stream:
            if cancel_check is not None:
                cancel_check()
            delta = chunk.delta or ""
            if delta:
                raw_stream += delta
                print(delta, end="", flush=True)
                update_placeholders()

            tokens = max(
                tokens,
                extract_total_tokens(chunk.raw),
                extract_total_tokens(chunk.additional_kwargs),
            )

    print("", flush=True)
    visible_content, _tagged_reasoning = split_thinking_blocks(raw_stream)
    
    final_code = visible_content.strip()
    if not final_code:
        if loop_detected:
            final_code = "__GENERATION_ERROR__: The model was forcefully stopped due to an infinite reasoning loop."
        else:
            final_code = "__GENERATION_ERROR__: No code was generated by the model."
    elif loop_detected:
        # Some partial code may have been emitted before the loop was detected.
        # If the code looks incomplete (no newlines, no import, no print), treat it as an error.
        if final_code.count('\n') < 2 or not any(kw in final_code for kw in ('import ', 'print(', 'pd.', 'df')):
            final_code = "__GENERATION_ERROR__: The model entered an infinite reasoning loop. Partial output was not valid Python."

    if hasattr(llm, "temperature"):
        llm.temperature = original_temperature

    tokens = max(tokens, get_llm_token_usage(llm))
    return final_code, tokens


def phase3_generate_and_execute(
    query: str,
    tables: list[str],
    candidates: list[str],
    solr_meta: SolrMetadata,
    reasoning: str,
    llm: LLM,
    pm: PromptManager,
    csv_dir: Path,
    retries: int = 0,
    error_msg: str = "",
    previous_code: str = "",
    force_execution: bool = False,
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
    cancel_check: Callable[[], None] | None = None,
    run_dir: Path | None = None,
    seed: int = 0,
    seed_instruction_recorder: Callable[[], None] | None = None,
    coder_context_level: CoderContextLevel = CoderContextLevel.FULL,
    evaluation_result_type: str | None = None,
) -> Phase3Result:
    code_raw, tokens = phase3_generate_code(
        query,
        tables,
        candidates,
        solr_meta,
        reasoning,
        llm,
        pm,
        csv_dir,
        retries=retries,
        error_msg=error_msg,
        previous_code=previous_code,
        force_execution=force_execution,
        stream_placeholder=stream_placeholder,
        reasoning_placeholder=reasoning_placeholder,
        stream_reasoning=stream_reasoning,
        cancel_check=cancel_check,
        seed=seed,
        seed_instruction_recorder=seed_instruction_recorder,
        coder_context_level=coder_context_level,
        evaluation_result_type=evaluation_result_type,
    )

    # Detect generation errors (loop, empty output) before attempting execution
    if code_raw.startswith("__GENERATION_ERROR__:"):
        error_detail = code_raw.replace("__GENERATION_ERROR__:", "").strip()
        return Phase3Result(
            code_raw=code_raw,
            tokens=tokens,
            error=f"Code generation failed: {error_detail}",
            clean_code="",
        )

    rejected_reason = "" if force_execution else _rejected_tables_reason(code_raw)
    if rejected_reason:
        return Phase3Result(
            code_raw=code_raw,
            tokens=tokens,
            rejected_reason=rejected_reason,
        )

    if cancel_check is not None:
        cancel_check()

    resolved_code, preflight_error = _resolve_and_validate_columns(
        code_raw, tables, csv_dir
    )
    if preflight_error:
        return Phase3Result(
            code_raw=code_raw,
            tokens=tokens,
            error=preflight_error,
            clean_code=resolved_code,
        )

    raw_result, error, clean_code = _execute_code(resolved_code, run_dir=run_dir)
    structured_result = None
    structured_result_error = ""
    if error is None and raw_result is not None and evaluation_result_type:
        raw_result, structured_result, payload_error = extract_evaluation_payload(
            raw_result
        )
        structured_result_error = payload_error or ""
    return Phase3Result(
        code_raw=code_raw,
        tokens=tokens,
        raw_result=raw_result,
        error=error,
        clean_code=clean_code,
        structured_result=structured_result,
        structured_result_error=structured_result_error,
    )
