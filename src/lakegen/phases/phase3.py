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
from prompts.prompt_manager import PromptManager
from lakegen.core.config import BASE_DIR

from .phase1 import split_thinking_blocks


@dataclass
class Phase3Result:
    code_raw: str
    tokens: int
    raw_result: str | None = None
    error: str | None = None
    clean_code: str = ""
    rejected_reason: str = ""


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
]


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
    if any(f in code for f in forbidden):
        return None, "Security Error: Forbidden libraries used.", code

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


def _detect_separator(filepath: str) -> str:
    """Helper to detect whether a CSV uses ',' or ';' as delimiter."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        semicolons = first_line.count(";")
        commas = first_line.count(",")
        if semicolons > commas:
            return ";"
    except Exception:
        pass
    return ","


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
):
    MAX_DETAIL_COLS = 25    # columns shown with type info
    MAX_SAMPLE_COLS = 15    # columns shown in sample rows
    MAX_SAMPLE_ROWS = 2     # rows in sample preview
    MAX_CELL_WIDTH = 40     # max chars per cell in sample

    info_lines = ["AVAILABLE TABLES:"]
    for idx, fn in enumerate(tables, 1):
        filepath = os.path.join(csv_dir, fn.strip())
        meta = solr_meta.get(fn, {})
        cn = meta.get("columns.name", [])
        ct = meta.get("columns.type", [])

        sep = _detect_separator(filepath)
        load_cmd = f"pd.read_csv('{filepath}', sep={repr(sep)})"
        df = pd.read_csv(filepath, sep=sep, nrows=MAX_SAMPLE_ROWS + 1)

        # Build column info from metadata or actual dataframe
        if cn and len(cn) == len(ct):
            col_typed = [f"{n}({t})" for n, t in zip(cn, ct)]
        elif cn:
            col_typed = list(cn)
        else:
            try:
                col_typed = [f"{c}({df[c].dtype})" for c in df.columns]
            except Exception:
                col_typed = ["Unknown columns"]

        total_cols = len(col_typed)
        shown = col_typed[:MAX_DETAIL_COLS]

        # Sample preview: limit columns and truncate wide cells
        sample_df = df.head(MAX_SAMPLE_ROWS).copy()
        if len(sample_df.columns) > MAX_SAMPLE_COLS:
            sample_df = sample_df.iloc[:, :MAX_SAMPLE_COLS].copy()
        for col in sample_df.columns:
            sample_df[col] = sample_df[col].astype(str).str.slice(0, MAX_CELL_WIDTH)
        sample_str = sample_df.to_string(index=False)

        info_lines.append(f"{idx}. LOAD: {load_cmd}")
        info_lines.append(f"   Columns ({total_cols}): {', '.join(shown)}")
        if total_cols > MAX_DETAIL_COLS:
            rest_names = [c.split("(")[0] if "(" in c else c for c in col_typed[MAX_DETAIL_COLS:]]
            info_lines.append(f"   +{total_cols - MAX_DETAIL_COLS} more: {', '.join(rest_names[:15])}")
        info_lines.append(f"   Sample:\n{sample_str}")

    tables_info = "\n".join(info_lines)

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

    # --- TabPFN Dynamic Hint Injection ---
    tabpfn_keywords = [
        # English
        "predict", "regression", "causal", "forecast", "prediction", "predictive",
        # Italian
        "predici", "regressione", "causale", "previsione", "predittivo", "predizione",
        # French
        "prédir", "predir", "régression", "regression", "prévision", "prevision", "prédictif", "predictif",
        # Spanish
        "predecir", "regresión", "regresion", "pronóstico", "pronostico", "predicción", "prediccion", "predictivo"
    ]
    if any(kw in query.lower() for kw in tabpfn_keywords):
        tabpfn_hint = (
            "\n\n[TABPFN REQUIREMENT]\n"
            "The user query requires prediction, forecasting, or causal inference. "
            "You MUST use the `tabpfn` library for this task. Small models MUST follow this exact API:\n"
            "```python\n"
            "import torch\n"
            "from tabpfn import TabPFNRegressor, TabPFNClassifier\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "# For regression: model = TabPFNRegressor(device=device)\n"
            "# For classification: model = TabPFNClassifier(device=device)\n"
            "```\n"
            "DATAFRAME PREPARATION RULES:\n"
            "1. SPATIAL ALIGNMENT (NEIGHBORHOODS/DISTRICTS): If the user asks for predictions by neighborhood/district ('nei quartieri'), you MUST aggregate and merge the datasets by neighborhood. Map station names or coordinates (latitude, longitude) of the sensors/loops to neighborhoods using spatial proximity or lookup tables (e.g. `quartieri-di-bologna.csv` or `zone_urbanistiche.csv`). Do not just aggregate globally.\n"
            "2. TEMPORAL ALIGNMENT & FALLBACK: If merging on absolute dates (YYYY-MM-DD) results in 0 rows (e.g., due to different years like 2024 vs 2026), DO NOT fail. Fall back to merge the datasets by seasonal/weekly profiles: group both datasets by day of week (e.g., `.dt.dayofweek` or `.dt.strftime('%A')`) and/or hour/month, then merge on these profile keys.\n"
            "3. PREDICTIONS VS AVERAGES: When showing aggregated predictions (e.g. average monthly accidents per borough), DO NOT feed the average of features to the model (i.e. model.predict(X_mean)). Instead, feed the actual raw samples (X) to the model to get individual predictions, and then calculate the average of those predictions (e.g. y_pred = model.predict(X) and then calculate averages grouped by borough/district).\n"
            "4. FUTURE FORECASTS (DUPLICATION AVOIDANCE): When preparing a DataFrame for future predictions (e.g. forecasting the values of the next year for each district), make sure you only have one row per unique entity (e.g. use `.drop_duplicates()` on the district/neighborhood column). Otherwise, you will generate duplicate predictions for the same entity and sum/aggregate them incorrectly.\n"
            "5. TABPFN DATA LIMITS (CUDA OOM AVOIDANCE): TabPFN models are strictly designed for small-to-medium datasets (up to 10,000 samples). If the prepared training dataset (X) exceeds 10,000 rows, you MUST downsample it to exactly 10,000 rows (or fewer) using `df = df.sample(n=min(10000, len(df)), random_state=42)` and aligning X and y accordingly. This is critical to prevent CUDA Out of Memory errors.\n"
            "6. STRING-NUMERIC CLEANING: For columns containing mixed text/units and numbers (e.g. 'Strade 30 km/h o inferiore'), DO NOT use simple pd.to_numeric() with errors='coerce' directly because this turns all values into NaN. Instead, you MUST clean them by extracting the digits using regular expressions (e.g., `.str.extract(r'(\\d+)')`) before converting to numeric, ensuring features do not become constant.\n"
            "7. MODELING: Prepare features (X) and target (y), perform a train_test_split (80/20), train the TabPFN model, and print metrics (MSE or R2 for regression, accuracy for classification) along with a sample of predictions."
        )
        user_prompt += tabpfn_hint

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

    raw_stream = ""
    structured_reasoning = ""
    tokens = 0

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

            if chunk.raw:
                prompt_tokens = chunk.raw.get("prompt_eval_count") or 0
                completion_tokens = chunk.raw.get("eval_count") or 0
                if prompt_tokens or completion_tokens:
                    tokens = prompt_tokens + completion_tokens

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

            if chunk.raw:
                prompt_tokens = chunk.raw.get("prompt_eval_count") or 0
                completion_tokens = chunk.raw.get("eval_count") or 0
                if prompt_tokens or completion_tokens:
                    tokens = prompt_tokens + completion_tokens

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

    raw_result, error, clean_code = _execute_code(code_raw, run_dir=run_dir)
    return Phase3Result(
        code_raw=code_raw,
        tokens=tokens,
        raw_result=raw_result,
        error=error,
        clean_code=clean_code,
    )
