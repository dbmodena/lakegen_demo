import os
import sys
import io
import re
import csv
import json
import uuid
import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from typing import List, Any
from pydantic import Field

# LlamaIndex Instrumentation for Thinking Capture
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.agent import AgentRunStepEndEvent

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

CONFIG_FILE = BASE_DIR / "config.json"
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, "r") as f:
        config_data = json.load(f)
else:
    config_data = {"paths": {}}

paths = config_data.get("paths", {})

DATA_DIR = BASE_DIR / paths.get("data_dir", "Data")
CSV_DIR = BASE_DIR / paths.get("csv_dir", "Data/bologna_update/datasets/csv")
JSON_DIR = BASE_DIR / paths.get("json_metadata_dir", "Data/bologna_update/metadata")
DB_PATH = BASE_DIR / paths.get("blend_db_path", "Data/blend_index.db")
INDEXES_DIR = BASE_DIR / paths.get("indexes_dir", "Data/indexes")
LOG_DIR = BASE_DIR / paths.get("logs_dir", "logs")

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

CSV_LOG_COLUMNS = ["ID", "TIMESTAMP", "QUESTION", "TABLES_SELECTED", "KEYWORDS_RAW", "KEYWORDS_FINAL", "RETRIES", "SUCCESS", "REASONING", "DEBUG_RAW", "RAW_RESULT", "FINAL_RESULT", "TOKENS_PHASE1", "TOKENS_PHASE2", "TOKENS_PHASE3", "TOKENS_PHASE5", "ERROR"]

def save_experiment_log(question: str, code: str, result: str, retries: int, reasoning: str = "", tables: list = None, raw_keywords: str = "", final_keywords: list = None, debug_raw: str = "", final_result: str = "", full_trace: str = "", tokens_phase1: int = 0, tokens_phase2: int = 0, tokens_phase3: int = 0, tokens_phase5: int = 0, llm_thinking: str = "", agent_thinking: str = "", error: str = ""):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- TXT log (human-readable) ---
    txt_path = os.path.join(LOG_DIR, "experiments_log.txt")
    tables_str = f"\nTABLES SELECTED: {', '.join(tables)}" if tables else ""
    raw_kw_str = f"\nKEYWORDS (model raw output): {raw_keywords}" if raw_keywords else ""
    final_kw_str = f"\nKEYWORDS (final elaborated): {', '.join(final_keywords)}" if final_keywords else ""
    final_result_str = f"\nFINAL RESULT (Phase 5):\n{final_result}" if final_result else ""
    debug_raw_str = f"\nDEBUG RAW:\n{debug_raw}" if debug_raw else ""
    llm_thinking_str = f"\n{'-'*40}\nMODEL THINKING (Phase 3 - Code Generator):\n{llm_thinking}\n{'-'*40}" if llm_thinking else ""
    error_str = f"\nERROR:\n{error}" if error else ""
    
    reasoning_parts = []
    if full_trace:
        reasoning_parts.append(f"=== WORKFLOW TRACE ===\n{full_trace.strip()}")
    if agent_thinking:
        reasoning_parts.append(f"=== FULL UNTRUNCATED REASONING ===\n{agent_thinking.strip()}")
        
    reasoning_txt = "\n\n".join(reasoning_parts) if reasoning_parts else reasoning
    
    tokens_str = f"\nTOKENS: Phase1={tokens_phase1} | Phase2={tokens_phase2} | Phase3={tokens_phase3} | Phase5={tokens_phase5}" if any([tokens_phase1, tokens_phase2, tokens_phase3, tokens_phase5]) else ""
    log_entry = f"\n{'='*50}\nDATA: {timestamp}\nQUESTION: {question}{tables_str}{raw_kw_str}{final_kw_str}\nMODEL REASONING (Agent Trace):\n{reasoning_txt}{debug_raw_str}{tokens_str}\nRETRIES: {retries}{llm_thinking_str}\nCODE (extracted):\n{code}\n\nRAW OUTPUT (Phase 4):\n{result}{final_result_str}{error_str}\n{'='*50}\n"
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

class ThinkingCapture(BaseEventHandler):
    parts: List[str] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "ThinkingCapture"

    def handle(self, event) -> None:
        event_type = type(event).__name__
        
        def extract_thinking(msg):
            # Extract from LlamaIndex structured thinking blocks
            for block in getattr(msg, 'blocks', []):
                if getattr(block, 'block_type', '') == 'thinking':
                    content = getattr(block, 'content', '')
                    if content and content not in self.parts:
                        self.parts.append(content)
                        
            # Extract from raw text using <think> tags (fallback for some Ollama models)
            content = getattr(msg, 'content', '')
            if content and isinstance(content, str):
                matches = re.findall(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    text = match.strip()
                    if text and text not in self.parts:
                        self.parts.append(text)
                        
        # Capture from LLMChatEndEvent
        if event_type == "LLMChatEndEvent":
            response = getattr(event, 'response', None)
            if response:
                msg = getattr(response, 'message', None)
                if msg:
                    extract_thinking(msg)
                                
        # Capture from AgentRunStepEndEvent or Workflow Step Events
        elif event_type in ["AgentRunStepEndEvent", "StepEndEvent", "AgentOutput"]:
            output = getattr(event, 'step_output', None) or getattr(event, 'output', None) or event
            if output is None:
                return
            msg = getattr(output, 'output', None) or getattr(output, 'response', None)
            if msg:
                extract_thinking(msg)
