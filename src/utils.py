import os
import sys
import io
import re
import csv
import uuid
import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

CURRENT_DIR = Path(__file__).parent.resolve()

if CURRENT_DIR.name == "src":
    BASE_DIR = CURRENT_DIR.parent
else:
    BASE_DIR = CURRENT_DIR

percorso_src = str(BASE_DIR / "src")
percorso_blend = str(BASE_DIR / "src" / "blend")
percorso_sloth = str(BASE_DIR / "src" / "Sloth")

# Inserting at index 0, Python will search HERE FIRST
if percorso_sloth not in sys.path:
    sys.path.insert(0, percorso_sloth)
if percorso_blend not in sys.path:
    sys.path.insert(0, percorso_blend)
if percorso_src in sys.path:
    sys.path.remove(percorso_src)
sys.path.insert(0, percorso_src)

# Add project root so top-level packages (e.g. prompts/) are importable
percorso_root = str(BASE_DIR)
if percorso_root not in sys.path:
    sys.path.insert(0, percorso_root)

DATA_DIR = BASE_DIR / "Data"
CSV_DIR = DATA_DIR / "data_csv"
DB_PATH = DATA_DIR / "blend_index.db"
INDEXES_DIR = DATA_DIR / "indexes"
LOG_DIR = BASE_DIR / "logs"

# ==========================================
# UTILITIES
# ==========================================
def get_filtered_tables_info(selected_files: list) -> str:
    info_lines = [f"AVAILABLE SELECTED TABLES IN '{CSV_DIR}/':"]
    if not selected_files:
        selected_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    for idx, filename in enumerate(selected_files, 1):
        filepath = os.path.join(CSV_DIR, filename.strip())
        if not os.path.exists(filepath): continue
        try:
            df = pd.read_csv(filepath, nrows=3)
            cols_with_types = [f"'{col}' ({dtype})" for col, dtype in df.dtypes.items()]
            info_lines.append(f"{idx}. '{filepath}'")
            info_lines.append(f"   Columns: [" + ", ".join(cols_with_types) + "]")
        except Exception as e:
            pass
    return "\n".join(info_lines)

def extract_query_keywords(query: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = re.findall(r'\b\w+\b', query.lower())
    extracted_keywords = [lemmatizer.lemmatize(w) for w in words if w not in ENGLISH_STOP_WORDS]
    return ", ".join(list(dict.fromkeys(extracted_keywords)))

CSV_LOG_COLUMNS = ["ID", "TIMESTAMP", "QUESTION", "TABLES_SELECTED", "KEYWORDS_RAW", "KEYWORDS_FINAL", "RETRIES", "SUCCESS", "REASONING", "DEBUG_RAW", "RAW_RESULT", "FINAL_RESULT", "TOKENS_PHASE1", "TOKENS_PHASE2", "TOKENS_PHASE5", "ERROR"]

def save_experiment_log(question: str, code: str, result: str, retries: int, reasoning: str = "", tables: list = None, raw_keywords: str = "", final_keywords: list = None, debug_raw: str = "", final_result: str = "", full_trace: str = "", tokens_phase1: int = 0, tokens_phase2: int = 0, tokens_phase5: int = 0, error: str = ""):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- TXT log (human-readable) ---
    txt_path = os.path.join(LOG_DIR, "experiments_log.txt")
    tables_str = f"\nTABLES SELECTED: {', '.join(tables)}" if tables else ""
    raw_kw_str = f"\nKEYWORDS (model raw output): {raw_keywords}" if raw_keywords else ""
    final_kw_str = f"\nKEYWORDS (final elaborated): {', '.join(final_keywords)}" if final_keywords else ""
    final_result_str = f"\nFINAL RESULT (Phase 5):\n{final_result}" if final_result else ""
    debug_raw_str = f"\nDEBUG RAW:\n{debug_raw}" if debug_raw else ""
    error_str = f"\nERROR:\n{error}" if error else ""
    reasoning_txt = full_trace if full_trace else reasoning
    tokens_str = f"\nTOKENS: Phase1={tokens_phase1} | Phase2={tokens_phase2} | Phase5={tokens_phase5}" if any([tokens_phase1, tokens_phase2, tokens_phase5]) else ""
    log_entry = f"\n{'='*50}\nDATA: {timestamp}\nQUESTION: {question}{tables_str}{raw_kw_str}{final_kw_str}\nMODEL REASONING (Agent Trace):\n{reasoning_txt}{debug_raw_str}{tokens_str}\nRETRIES: {retries}\nCODE:\n{code}\n\nRAW OUTPUT (Phase 4):\n{result}{final_result_str}{error_str}\n{'='*50}\n"
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
        "TOKENS_PHASE5":   tokens_phase5,
        "ERROR":           error.replace("\n", "  "),
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_LOG_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)

class DualLogger:
    def __init__(self, terminal):
        self.terminal = terminal
        self.log_str = io.StringIO()
    def write(self, message):
        self.terminal.write(message)
        self.log_str.write(message)
    def flush(self):
        self.terminal.flush()
