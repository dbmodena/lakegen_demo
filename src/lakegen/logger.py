import os
import csv
import datetime
import pandas as pd

from lakegen.config import LOG_DIR

CSV_LOG_COLUMNS = [
    "ID", "TIMESTAMP", "QUESTION", "TABLES_SELECTED", "KEYWORDS_RAW", "KEYWORDS_FINAL", 
    "RETRIES", "SUCCESS", "REASONING", "DEBUG_RAW", "RAW_RESULT", "FINAL_RESULT", 
    "TOKENS_PHASE1", "TOKENS_PHASE2", "TOKENS_PHASE3", "TOKENS_PHASE4", "ERROR"
]

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
    error: str = ""
):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    synthesis_tokens = tokens_phase4 or tokens_phase5

    # --- TXT log (human-readable) ---
    txt_path = os.path.join(LOG_DIR, "experiments_log.txt")
    tables_str = f"\nTABLES SELECTED: {', '.join(tables)}" if tables else ""
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
    log_entry = f"\n{'='*50}\nDATA: {timestamp}\nQUESTION: {question}{tables_str}{raw_kw_str}{final_kw_str}\nMODEL REASONING (Agent Trace):\n{reasoning_txt}{debug_raw_str}{tokens_str}\nRETRIES: {retries}{llm_thinking_str}\nCODE (extracted):\n{code}\n\nRAW OUTPUT (Phase 3):\n{result}{final_result_str}{error_str}\n{'='*50}\n"
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # --- CSV log (structured, for analysis) ---
    csv_path = os.path.join(LOG_DIR, "experiments_log.csv")
    is_new_file = not os.path.exists(csv_path)
    success = not result.startswith("[EXECUTION ERROR]") and not result.startswith("[CRITICAL ERROR]") and not error
    
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

    row = {
        "ID":              next_id,
        "TIMESTAMP":       timestamp,
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
    fieldnames = CSV_LOG_COLUMNS
    if not is_new_file:
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                existing_header = next(csv.reader(f), None)
            if existing_header:
                fieldnames = existing_header
        except Exception:
            pass
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
