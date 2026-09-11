import os
import csv
import datetime
import json
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lakegen.core.config import LOG_DIR

CSV_LOG_COLUMNS = [
    "ID", "TIMESTAMP", "MODEL", "ARCHITECTURE", "QUESTION",
    "TABLES_SELECTED", "KEYWORDS_RAW", "KEYWORDS_FINAL", "RETRIES",
    "SUCCESS", "REASONING", "DEBUG_RAW", "RAW_RESULT", "FINAL_RESULT",
    "TOKENS_PHASE1", "TOKENS_PHASE2", "TOKENS_PHASE3", "TOKENS_PHASE4",
    "ERROR",
]

API_CSV_LOG_COLUMNS = [
    "ID", "TIMESTAMP", "JOB_ID", "SOURCE_PATH", "SOURCE_ID",
    "EXECUTION_ATTEMPT", "IS_FINAL_ATTEMPT", "EXPERIMENT_ID", "MANIFEST_ID", "MODEL",
    "ARCHITECTURE", "CORE", "PORTAL_NAME", "STATUS", "QUESTION",
    "TABLES_SELECTED", "KEYWORDS_FINAL", "SOURCE_RELEVANT_TABLE_IDS",
    "RETRIEVAL_MODE", "TOP_K", "HYBRID_ALPHA", "CANDIDATE_MULTIPLIER",
    "REPRESENTATION_VERSION", "EMBEDDING_MODEL", "EMBEDDING_BASE_URL",
    "VECTOR_FIELD", "LEXICAL_QUERY_FIELDS", "MISSING_SIGNAL_POLICY",
    "FUSION_METHOD", "RRF_K",
    "PIPELINE_STAGES_JSON", "ANSWER_DISPOSITION", "RETRIES", "SUCCESS",
    "TOKENS_PHASE1", "TOKENS_PHASE2", "TOKENS_PHASE3",
    "TOKENS_PHASE4", "TOKENS_PHASE5", "ELAPSED_SECONDS", "ERROR",
    "HIT_AT_1", "HIT_AT_5", "HIT_AT_10", "RECALL_AT_1", "RECALL_AT_5",
    "RECALL_AT_10", "MRR", "NDCG_AT_1", "NDCG_AT_5", "NDCG_AT_10",
]

API_CSV_EXCLUDED_COLUMNS = (
    "FULL_TRACE", "SOURCE_JSON", "KEYWORDS_RAW", "REASONING", "DEBUG_RAW",
    "RAW_RESULT", "FINAL_RESULT", "RETRIEVAL_RUNS_JSON", "LLM_THINKING",
    "AGENT_THINKING", "CODE", "MANIFEST_JSON", "RUN_TRACE_JSON",
    "SOURCE_CODE", "SOURCE_RESPONSE", "SOURCE_JUDGE_FEEDBACK",
    "SOURCE_TABLES", "SOURCE_REFERENCE_CODE", "SOURCE_REFERENCE_RESULT",
    "SOURCE_EXPECTED_RESULT_DESCRIPTION",
)

_CSV_LOCK = threading.Lock()


def _raise_csv_field_limit() -> None:
    """Allow intentionally lossless API log fields to be read back safely."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_raise_csv_field_limit()


def _csv_value(value: Any) -> Any:
    """Return a lossless, CSV-safe representation for structured values."""

    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_csv_columns(
    csv_path: str,
    required_columns: list[str],
    *,
    excluded_columns: tuple[str, ...] = (),
) -> list[str]:
    """Evolve a CSV schema, adding fields and removing explicitly private ones."""

    excluded = set(excluded_columns)
    required_columns = [
        column for column in required_columns if column not in excluded
    ]
    if not os.path.exists(csv_path):
        return required_columns

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        existing_columns = list(reader.fieldnames or [])
        rows = list(reader)

    retained_columns = [
        column for column in existing_columns if column not in excluded
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in retained_columns and column not in excluded
    ]
    evolved_columns = retained_columns + missing_columns
    if evolved_columns == existing_columns:
        return existing_columns

    temporary_path = f"{csv_path}.tmp"
    with open(temporary_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=evolved_columns, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, csv_path)
    return evolved_columns

def save_experiment_log(
    question: str, 
    code: str, 
    result: str, 
    retries: int, 
    reasoning: str = "", 
    tables: list = None, 
    raw_keywords: str = "", 
    final_keywords: list = None, 
    debug_raw: str = "", 
    final_result: str = "", 
    full_trace: str = "", 
    tokens_phase1: int = 0, 
    tokens_phase2: int = 0, 
    tokens_phase3: int = 0, 
    tokens_phase5: int = 0, 
    tokens_phase4: int = 0, 
    llm_thinking: str = "", 
    agent_thinking: str = "", 
    error: str = "",
    csv_filename: str = "api_experiments_log.csv",
    model: str = "",
    architecture: str = "",
    status: str = "",
    elapsed_seconds: float = 0.0,
    extra_fields: Mapping[str, Any] | None = None,
):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    synthesis_tokens = tokens_phase4 or tokens_phase5

    # --- TXT log (human-readable) ---
    txt_path = os.path.join(LOG_DIR, "experiments_log.txt")
    tables_str = f"\nTABLES SELECTED: {', '.join(tables)}" if tables else ""
    model_str = f"\nMODEL: {model}" if model else ""
    architecture_str = f"\nARCHITECTURE: {architecture}" if architecture else ""
    raw_kw_str = f"\nKEYWORDS (model raw output): {raw_keywords}" if raw_keywords else ""
    final_kw_str = f"\nKEYWORDS (final elaborated): {', '.join(final_keywords)}" if final_keywords else ""
    final_result_str = f"\nFINAL RESULT (Phase 4):\n{final_result}" if final_result else ""
    debug_raw_str = f"\nDEBUG RAW:\n{debug_raw}" if debug_raw else ""
    llm_thinking_str = f"\n{'-'*40}\nMODEL THINKING (Phase 3 - Code Generator):\n{llm_thinking}\n{'-'*40}" if llm_thinking else ""
    error_str = f"\nERROR:\n{error}" if error else ""
    
    reasoning_parts = []
    if full_trace:
        reasoning_parts.append(f"=== WORKFLOW TRACE ===\n{full_trace.strip()}")
    if agent_thinking:
        reasoning_parts.append(f"=== FULL UNTRUNCATED REASONING ===\n{agent_thinking.strip()}")
        
    reasoning_txt = "\n\n".join(reasoning_parts) if reasoning_parts else reasoning
    
    tokens_str = f"\nTOKENS: Phase1={tokens_phase1} | Phase2={tokens_phase2} | Phase3={tokens_phase3} | Phase4={synthesis_tokens}" if any([tokens_phase1, tokens_phase2, tokens_phase3, synthesis_tokens]) else ""
    log_entry = f"\n{'='*50}\nDATA: {timestamp}{model_str}{architecture_str}\nQUESTION: {question}{tables_str}{raw_kw_str}{final_kw_str}\nMODEL REASONING (Agent Trace):\n{reasoning_txt}{debug_raw_str}{tokens_str}\nRETRIES: {retries}{llm_thinking_str}\nCODE (extracted):\n{code}\n\nRAW OUTPUT (Phase 3):\n{result}{final_result_str}{error_str}\n{'='*50}\n"
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # --- CSV log (structured, for analysis) ---
    csv_path = os.path.join(LOG_DIR, csv_filename)
    success = (
        status == "completed"
        if status
        else not result.startswith("[EXECUTION ERROR]")
        and not result.startswith("[CRITICAL ERROR]")
        and not error
    )
    is_api_log = csv_filename == "api_experiments_log.csv"
    row: dict[str, Any] = {
        "TIMESTAMP":       timestamp,
        "EXECUTION_ATTEMPT": 1,
        "IS_FINAL_ATTEMPT": True,
        "MODEL":           model,
        "ARCHITECTURE":    architecture,
        "STATUS":          status,
        "QUESTION":        question,
        "TABLES_SELECTED": ", ".join(tables) if tables else "",
        "KEYWORDS_RAW":    raw_keywords,
        "KEYWORDS_FINAL":  ", ".join(final_keywords) if final_keywords else "",
        "RETRIES":         retries,
        "SUCCESS":         success,
        "REASONING":       reasoning,
        "DEBUG_RAW":       debug_raw[:100].replace("'", "").replace('"', "").replace("\n", " "),
        "RAW_RESULT":      result[:500].replace("\n", "  "),
        "FINAL_RESULT":    final_result[:500].replace("\n", "  ") if final_result else "",
        "TOKENS_PHASE1":   tokens_phase1,
        "TOKENS_PHASE2":   tokens_phase2,
        "TOKENS_PHASE3":   tokens_phase3,
        "TOKENS_PHASE4":   synthesis_tokens,
        "TOKENS_PHASE5":   synthesis_tokens,
        "ERROR":           error.replace("\n", "  "),
    }
    if is_api_log:
        row.update({
            "DEBUG_RAW": debug_raw,
            "LLM_THINKING": llm_thinking,
            "AGENT_THINKING": agent_thinking,
            "CODE": code,
            "RAW_RESULT": result,
            "FINAL_RESULT": final_result,
            "TOKENS_PHASE5": synthesis_tokens,
            "ELAPSED_SECONDS": elapsed_seconds,
            "ERROR": error,
        })
    if extra_fields:
        row.update({str(key): _csv_value(value) for key, value in extra_fields.items()})

    base_columns = API_CSV_LOG_COLUMNS if is_api_log else CSV_LOG_COLUMNS
    extra_columns = (
        [column for column in (extra_fields or {}) if column in API_CSV_LOG_COLUMNS]
        if is_api_log
        else list(extra_fields or {})
    )
    required_columns = list(dict.fromkeys([*base_columns, *extra_columns]))
    with _CSV_LOCK:
        is_new_file = not os.path.exists(csv_path)
        fieldnames = _ensure_csv_columns(
            csv_path,
            required_columns,
            excluded_columns=(
                API_CSV_EXCLUDED_COLUMNS if is_api_log else ("FULL_TRACE",)
            ),
        )

        next_id = 1
        if not is_new_file:
            try:
                df = pd.read_csv(csv_path, usecols=["ID"])
                numeric_ids = pd.to_numeric(df["ID"], errors="coerce")
                if not numeric_ids.isna().all():
                    next_id = int(numeric_ids.max()) + 1
                else:
                    next_id = len(df) + 1
            except Exception:
                pass
        row["ID"] = next_id

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if is_new_file:
                writer.writeheader()
            writer.writerow(row)
