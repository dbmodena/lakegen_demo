import os
import json
import re
import subprocess
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
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
from lakegen.coder_context import CoderContext

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
    execution_error: dict[str, object] | None = None
    coder_review: dict[str, str] | None = None
    coder_runs: int = 0
    coder_lifecycle: str = ""
    stop_reason: str = ""
    finalization_mode: str = ""
    operation_trace: dict[str, object] | None = None
    coder_context_audit: dict[str, object] | None = None
    rejection_details: dict[str, object] | None = None
    coder_attempt_trace: list[dict[str, object]] = field(default_factory=list)


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


def _recover_fenced_agent_code(response: str, manager, state) -> bool:
    """Execute and inspect a complete program emitted without a run_code call."""
    if (
        state.run_count != 0
        or state.error is not None
        or state.rejected_reason
        or "```python" not in response.casefold()
    ):
        return False
    fallback_code = _extract_code(response)
    if (
        "__LAKEGEN_EVAL_JSON__" not in fallback_code
        or not re.search(r"\b(?:import|from)\s+\w+", fallback_code)
    ):
        return False
    if not state.analysis_contract:
        manager.infer_analysis_contract()
    fallback_run = json.loads(manager.run_code(fallback_code))
    if not fallback_run.get("ok"):
        return False
    manager.inspect_result()
    state.stop_reason = "recovered_fenced_code_without_run_code_call"
    return True


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
    table_metadata: SolrMetadata | None = None,
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

        metadata = (table_metadata or {}).get(
            fn, (table_metadata or {}).get(Path(fn).stem, {})
        )
        title = " ".join(str(metadata.get("title") or "").split())[:160]
        description = " ".join(
            str(metadata.get("description") or "").split()
        )[:320]
        if title:
            info_lines.append(f"   Resource title: {title}")
        if description:
            info_lines.append(f"   Resource description: {description}")

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

        structured_columns = metadata.get("columns", [])
        if isinstance(structured_columns, list):
            descriptions = {
                re.sub(r"[^a-z0-9]", "", str(column.get("name", "")).casefold()):
                    " ".join(str(column.get("description") or "").split())[:180]
                for column in structured_columns
                if isinstance(column, dict) and column.get("name") and column.get("description")
            }
            described = []
            for column in df.columns:
                description_hint = descriptions.get(
                    re.sub(r"[^a-z0-9]", "", str(column).casefold())
                )
                if description_hint:
                    described.append(f"{column}: {description_hint}")
                if len(described) >= 12:
                    break
            if described:
                info_lines.append("   Column semantics:")
                info_lines.extend(f"     - {item}" for item in described)

        temporal_candidates = [
            str(column) for column in df.columns
            if re.search(r"(?:^|[_\W])(year|date|time|period|fy|sy)(?:$|[_\W])", str(column), re.IGNORECASE)
            or re.search(r"(?:year|date)$", str(column), re.IGNORECASE)
        ]
        if len(temporal_candidates) > 1:
            info_lines.append(
                "   Temporal ambiguity: " + ", ".join(temporal_candidates[:8])
                + ". Choose by resource/column meaning, not name alone."
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
    coder_context = CoderContext.build(
        question=query, selected_tables=tables, table_metadata=solr_meta,
        selection_plan={}, execution_error={"message": error_msg} if error_msg else {},
    )
    solr_meta = dict(coder_context.table_metadata)
    context_level = CoderContextLevel(coder_context_level)
    tables_info = _build_coder_tables_info(
        tables, Path(csv_dir), context_level, solr_meta
    )

    system_prompt = pm.render("code_generator", "system_prompt")
    if retries == 0:
        user_prompt = pm.render("code_generator", "initial_prompt",
                                question=query, arch_reasoning="",
                                tables_info=tables_info)
    else:
        user_prompt = pm.render("code_generator", "correction_prompt",
                                question=query, error_message=error_msg,
                                previous_code=previous_code,
                                arch_reasoning="",
                                tables_info=tables_info)

    user_prompt += (
        "\n\n[REPRODUCIBILITY]\n"
        f"The effective seed for this run is {seed}. Use exactly this value for "
        "every stochastic operation that accepts a seed (including pandas "
        "sampling, train/test split, cross-validation splitters, and model "
        "random_state parameters). Do not add a seed to deterministic operations. "
        "Do not use another fixed seed.\n"
    )
    user_prompt += evaluation_output_instruction(
        str(coder_context.output_shape["result_type"])
    )

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
    max_run_calls: int = 3,
    selection_plan: dict[str, object] | None = None,
    source_field_names: list[str] | None = None,
    require_semantic_plan: bool = True,
) -> Phase3Result:
    # Keep retrieval/discovery and the existing sandbox unchanged: only the
    # coder's generate/execute retry loop becomes a bounded tool-using agent.
    from lakegen.agent_tools.tools_p3 import P3State, Phase3ToolsManager
    from lakegen.agents.agent_runner import run_agent_workflow
    from lakegen.phases.logging import Phase2AgentStall

    coder_context = CoderContext.build(
        question=query,
        selected_tables=tables,
        table_metadata=solr_meta,
        selection_plan=selection_plan,
        source_payload={name: None for name in (source_field_names or [])},
        execution_error={"message": error_msg} if error_msg else {},
    )
    # From this point onward Phase 3 sees only the DTO's explicit allowlist.
    selection_plan = dict(coder_context.selection_plan)
    solr_meta = dict(coder_context.table_metadata)
    context_level = CoderContextLevel(coder_context_level)
    tables_info = _build_coder_tables_info(
        tables, Path(csv_dir), context_level, solr_meta
    )
    system_prompt = pm.render("code_generator", "agentic_system_prompt") + (
        "\n\nYou are a bounded coding agent. A semantic plan is authoritative only "
        "after deterministic runtime validation reports it locked. An invalid or "
        "insufficiently evidenced plan must be inspected, revised through the "
        "analysis contract, reported with plan_conflict, or rejected with concrete "
        "evidence. For legacy selections without a validated "
        "semantic plan, set_analysis_contract may record filters, measures, "
        "grouping, distinct counts, joins, ordering, limit, and output columns. "
        "Treat a year that names a dataset edition or release (for example, "
        "'using the 2024 ... dataset') as resource provenance, not as a row-level "
        "filter: omit it from filters and do not require a year/date column. "
        "The contract must preserve every semantic "
        "qualifier from the question (for example Community, completed, distinct, "
        "and the exact measure being averaged). Objective contract violations block "
        "finalization and require corrected code. "
        "Do not add filters, limits, distinct-count semantics, deduplication, or "
        "temporal interpretations that the question does not request. Preserve every "
        "requested measure and identifying dimension in the final structured output. "
        "Do not invent bucket boundaries with qcut/cut, tertiles, or quantiles unless "
        "the question explicitly defines that bucketing rule. Do not replace a named "
        "measure with a nearby total, count, or identifier merely because it is easier "
        "to compute. For a qualified filter, inspect the exact column values when its "
        "encoding is unclear; a non-empty value is not evidence that a semantic "
        "qualifier is satisfied. "
        "For distinct counts of categories or types, use the column whose meaning "
        "represents the requested category; do not automatically prefer an identifier "
        "code when an explicit descriptive column is available. "
        "Then write a complete Python program and "
        "call run_code with it. If execution fails, use the structured error to "
        "correct the program. After every successful run, call inspect_result and "
        "verify that the result answers the whole question. A warning-free "
        "inspection is finalized automatically; do not spend another turn on a "
        "closure protocol. You have at most "
        f"{min(2, max_run_calls)} "
        "run_code calls. You may call inspect_table once per selected table when "
        "you need exact columns, category values, null counts, or temporal coverage; "
        "the same cached sample supports progressive profiling of up to 8 columns. "
        "Do this instead of spending run_code on diagnostic prints. After inspection "
        "you MUST choose: run_code if the data is sufficient, or reject_tables with "
        "specific missing requirements and evidence if it is insufficient. If a "
        "run reports diagnostic_output, do not print more diagnostics: use "
        "inspect_table, then correct the analysis or reject the tables. Never "
        "infer correctness from benchmark gold: it is not available. "
        "Do not import sys; it is unnecessary and forbidden by the execution sandbox. "
        "Do not merely describe code in chat."
    )
    if retries == 0:
        user_prompt = pm.render(
            "code_generator", "agentic_initial_prompt", question=query,
            arch_reasoning="", tables_info=tables_info,
        )
    else:
        user_prompt = pm.render(
            "code_generator", "agentic_correction_prompt", question=query,
            error_message=error_msg, previous_code=previous_code,
            arch_reasoning="", tables_info=tables_info,
        )
    user_prompt += (
        "\n\nQUESTION-DERIVED OUTPUT SHAPE (non-gold):\n"
        + json.dumps(coder_context.output_shape, ensure_ascii=False, sort_keys=True)
    )
    user_prompt += (
        "\n\n[REPRODUCIBILITY]\n"
        f"Use seed {seed} for every stochastic operation. "
        "Before finishing, inspect the actual output and check filters, measures, "
        "group coverage, ordering/limits, and output shape."
    )
    coder_result_type = str(coder_context.output_shape["result_type"])
    user_prompt += evaluation_output_instruction(coder_result_type)
    if force_execution:
        user_prompt += (
            "\nThe selected tables must be used for the best possible executable "
            "answer; do not reject them."
        )
    if seed_instruction_recorder is not None:
        seed_instruction_recorder()

    # One initial analysis plus at most one evidence-directed revision.
    state = P3State(max_runs=max(1, min(2, max_run_calls)))
    manager = Phase3ToolsManager(
        state,
        tables=tables,
        csv_dir=Path(csv_dir),
        run_dir=run_dir,
        evaluation_result_type=coder_result_type,
        question=query,
        table_metadata={
            table: solr_meta.get(table, solr_meta.get(Path(table).stem, {}))
            for table in tables
        },
        selection_plan=selection_plan,
        resolve_code=_resolve_and_validate_columns,
        execute_code=_execute_code,
        extract_payload=extract_evaluation_payload,
        require_semantic_plan=require_semantic_plan,
        # The coder brief is the sole semantic contract. Runtime code/result
        # inspection supplies independent evidence without a duplicate model-
        # authored manifest.
        require_analysis_manifest=False,
    )
    plan_view = manager.coder_plan_view()
    initial_plan_status = str(plan_view.get("status") or "missing")
    runnable_plan_statuses = {"verified", "executable_with_obligations"}
    if require_semantic_plan and plan_view.get("status") not in runnable_plan_statuses:
        status = str(plan_view.get("status") or "invalid")
        audit = {
            **coder_context.audit(),
            **plan_view,
            "semantic_plan_status": status,
            "semantic_plan_initial_status": initial_plan_status,
            "semantic_plan_final_status": status,
            "semantic_plan_coder_start_status": "not_started",
            "validation_diagnostics": list(plan_view.get("diagnostics") or []),
            "evidence_count": len(plan_view.get("evidence") or []),
            "coder_started_after_verified_plan": False,
        }
        return Phase3Result(
            code_raw="", tokens=0,
            error=f"Selection contract gate blocked coder startup: {status}.",
            execution_error={
                "stage": "selection_contract_gate",
                "category": "coder_brief_not_verified",
                "status": status,
                "retryable": status in {"missing", "invalid", "needs_runtime_evidence"},
                "diagnostics": audit["validation_diagnostics"],
            },
            coder_runs=0, coder_lifecycle="blocked_before_start",
            stop_reason="selection_contract_gate",
            coder_context_audit=audit,
        )
    user_prompt += (
        "\n\nSELECTION CONTRACT RUNTIME VALIDATION (non-gold):\n"
        + json.dumps(plan_view or {
            "valid": False,
            "locked": False,
            "status": "missing",
            "required_action": "inspect_table_then_set_analysis_contract_or_reject_tables",
        }, ensure_ascii=False, sort_keys=True, default=str)
    )
    reset_llm_token_usage(llm)

    def emit_stream(delta: str) -> None:
        if not delta:
            return
        print(delta, end="", flush=True)
        if stream_placeholder is not None:
            stream_placeholder.markdown(delta)

    try:
        response = run_agent_workflow(
            llm=llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            agent_name="coder",
            emit_stream=emit_stream,
            cancel_check=cancel_check,
            tools=[
                tool for tool in manager.get_tools()
                if tool.metadata.name != "finish_code"
            ],
            max_iterations=10,
            # inspect_result has no arguments and may legitimately follow both
            # the initial execution and its single revision.
            # Repeated inspect_table calls are served from a cache and no longer
            # justify aborting the whole coder. The global tool budget still
            # bounds unproductive loops.
            max_repeats=8,
            max_tool_calls=10,
            timeout_seconds=600,
        )
    except Phase2AgentStall as exc:
        response = ""
        state.stop_reason = str(exc)
        if not state.error and state.raw_result is None:
            state.error = f"Coder agent stalled: {exc}"
    except Exception as exc:
        response = ""
        if not state.error:
            state.error = f"Coder agent failed: {type(exc).__name__}: {exc}"

    # Some model turns emit the complete program in a fenced response instead
    # of invoking run_code. Route that program through the exact same bounded
    # preflight, sandbox, result extraction and inspection path as a tool call.
    _recover_fenced_agent_code(response, manager, state)

    # inspect_result is deterministic and does not consume a run_code attempt.
    # If the model stops immediately after a valid structured execution, apply
    # the same inspection tool programmatically so closure/revision routing is
    # based on the actual latest result.
    if (
        state.error is None
        and state.raw_result is not None
        and state.structured_result is not None
        and state.inspected_version != state.result_version
    ):
        manager.inspect_result()
        state.stop_reason = state.stop_reason or "auto_inspected_latest_successful_result"

    if force_execution:
        rejected_reason = ""
    elif state.rejected_reason:
        missing = state.rejection_details.get("missing_requirements", [])
        evidence = state.rejection_details.get("inspected_evidence", "")
        rejected_reason = (
            f"{state.rejected_reason} Missing requirements: {missing}. "
            f"Inspected evidence: {evidence}"
        )
    else:
        rejected_reason = _rejected_tables_reason(response)
    # A model can exhaust its normal reasoning turn immediately after inspection.
    # Inspection is deterministic bookkeeping over the latest runtime result.
    # Do it when the model executed successfully but omitted the protocol call.
    if (
        not state.finished
        and state.lifecycle.value == "needs_inspection"
        and state.raw_result is not None
        and not rejected_reason
    ):
        try:
            manager.inspect_result()
        except Exception as exc:
            state.stop_reason = (
                "automatic_inspection_failed: " f"{type(exc).__name__}: {exc}"
            )

    # Deterministic validation is the terminal gate; no second LLM is needed.
    if not state.finished and state.ready_for_finalization() and not rejected_reason:
        manager.finalize_validated_result(
            "Automatically finalized after a valid, structured, warning-free inspection."
        )

    # A latest inspected structured result remains useful even when semantic
    # advisories could not be resolved within the bounded coder loop. Preserve
    # it as a degraded completion; downstream evaluation still decides its
    # correctness and the advisories remain attached to the review telemetry.
    if (
        not state.finished
        and state.ready_for_degraded_finalization()
        and state.run_count >= state.max_runs
        and not rejected_reason
    ):
        manager.recover_degraded_finish(
            "System recovery preserved the latest inspected structured result "
            "after the coder exhausted its correction/finalization budget."
        )

    tokens = get_llm_token_usage(llm)
    if not state.finished and state.error is None and not rejected_reason:
        state.error = {
            "needs_inspection": "Coder stopped after execution without inspecting the latest result.",
            "needs_revision": "Coder stopped while the inspected result still required revision.",
            "ready_to_finish": "Coder stopped before mandatory finalization.",
        }.get(
            state.lifecycle.value,
            "Coder stopped without reaching a terminal coding state.",
        )
    if (
        not state.finished
        and state.lifecycle.value == "needs_revision"
        and not state.execution_error
    ):
        state.execution_error = {
            "stage": "result_validation",
            "category": "result_needs_revision",
            "message": state.error or "The inspected result requires revision.",
            "retryable": state.run_count < state.max_runs,
            "coverage_warnings": list(state.coverage_warnings),
        }
    return Phase3Result(
        code_raw=state.code_raw or response,
        tokens=tokens,
        raw_result=state.raw_result,
        error=state.error,
        clean_code=state.clean_code,
        rejected_reason=rejected_reason,
        structured_result=state.structured_result,
        structured_result_error=state.structured_result_error,
        execution_error=state.execution_error or None,
        coder_review=state.review or None,
        coder_runs=state.run_count,
        coder_lifecycle=state.lifecycle.value,
        stop_reason=state.stop_reason,
        finalization_mode=state.finalization_mode,
        operation_trace=state.operation_trace or None,
        coder_context_audit={
            **coder_context.audit(),
            **{
                key: manager.coder_plan_view().get(key)
                for key in (
                    "semantic_plan_present", "semantic_plan_valid",
                    "semantic_plan_locked", "semantic_plan_revised",
                    "semantic_plan_rejected", "semantic_plan_missing",
                )
            },
            "semantic_plan_validated_before_lock": bool(state.plan_validation.get("valid")),
            "semantic_plan_status": state.plan_validation.get("status"),
            "semantic_plan_initial_status": initial_plan_status,
            "semantic_plan_final_status": state.plan_validation.get("status"),
            "semantic_plan_coder_start_status": initial_plan_status,
            "validation_diagnostics": list(state.plan_validation.get("diagnostics", [])),
            "evidence_count": len(state.plan_validation.get("evidence", [])),
            "coder_started_after_verified_plan": (
                not require_semantic_plan or initial_plan_status in runnable_plan_statuses
            ),
            "verified_requirements": list(state.coverage_requirements),
            "inspect_result_executed": state.inspected_version > 0,
        },
        rejection_details=state.rejection_details or None,
        coder_attempt_trace=list(state.execution_attempts),
    )
