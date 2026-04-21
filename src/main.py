# Download wordnet and omw-1.4 before running the program (only the first time)
# uv run python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

import os
import sys
import io
import re
import json
import asyncio
import datetime
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from thefuzz import fuzz
from valentine import valentine_match
from valentine.algorithms import JaccardDistanceMatcher

# from llama_index.llms.openai_like import OpenAILike
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Workflow, step, StartEvent, StopEvent, Event
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

from utils import (
    BASE_DIR,
    DATA_DIR,
    CSV_DIR,
    INDICI_DIR,
    DB_PATH,
    LOG_DIR,
    get_filtered_tables_info,
    extract_query_keywords,
    save_experiment_log,
    DualLogger
)
from tools import agent_tools

# ==========================================
# WORKFLOW DEFINITION (Events and Class)
# ==========================================
class TableSelectionEvent(Event): 
    selected_files: list
    reasoning: str
    full_trace: str
class ExecutionEvent(Event): 
    code: str
    retries: int
class CodeErrorEvent(Event): 
    error_message: str
    retries: int
class FinalResultEvent(Event): 
    raw_result: Any

class RobustLakeGenWorkflow(Workflow):
    def __init__(self, llm_instant, llm_versatile, inverted_index: dict, table_kw: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_instant = llm_instant
        self.llm_versatile = llm_versatile
        self.inverted_index = inverted_index
        self.table_kw = table_kw
        self.max_retries = 3
        self.all_available_files = list(set([f for files in self.inverted_index.values() for f in files]))

    @step
    async def select_tables(self, ev: StartEvent) -> TableSelectionEvent:
        """PHASE 1 and 2: Generate keywords and select tables."""
        self.question = ev.question

        # PHASE 1: KEYWORD GENERATION WITH LOOP
        raw_keywords_str = extract_query_keywords(self.question)
        default_keywords = [k.strip() for k in raw_keywords_str.split(',') if k.strip()]
        
        print(f"\nPHASE 1: KEYWORD GENERATION")
        print(f"    Question: '{self.question}'")
        print(f"    Base keywords: {default_keywords}")

        keyword_hint = ""
        while True:
            hint_section = f'\nUSER HINT (CRITICAL): "{keyword_hint}"' if keyword_hint else ""

            system_prompt = system_prompt = """You are a Data Processing API.
            TASK: Generate MAX 5 NEW domain-specific synonyms or related technical terms.
            STRICT RULES:
            1. DO NOT repeat the words provided in the input.
            2. Reply ONLY with a single line of comma-separated lowercase words.
            3. NO conversational text.
            """

            user_prompt = f"""Question: What are the best restaurants in Rome?
            Answer: restaurants, Rome, best
            Input Question: "{self.question}"
            Existing Words (DO NOT REPEAT THESE): "{raw_keywords_str}"
            {hint_section}
            New Synonyms:"""

            res = await self.llm_instant.achat([
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ])

            if res.raw:
                in_t = res.raw.get('prompt_eval_count', 0)
                out_t = res.raw.get('eval_count', 0)
                print(f"    📊 [TOKEN PHASE 1] Prompt: {in_t} | Generate: {out_t} | Tot: {in_t + out_t}")

            raw_content = str(res.message.content).strip().lower()
            print(f"    🔍 DEBUG raw model output: '{raw_content}'")

            # Takes only valid words
            extracted_words = re.findall(r'\b[a-z0-9_]+\b', raw_content)

            llm_fluff = {"here", "is", "are", "the", "list", "keywords", "output", "of", "sure", "certainly", "based", "on", "and", "or"}
            brute_keywords = [w for w in extracted_words if w not in llm_fluff]

            enriched_keywords = default_keywords.copy()
            for w in brute_keywords:
                if w not in enriched_keywords:
                    enriched_keywords.append(w)

            # Truncate excess words (base keywords + 5 new ones from LLM)
            enriched_keywords = enriched_keywords[:len(default_keywords) + 5]

            print(f"    📝 Final elaborated keywords: {enriched_keywords}")
            user_input = input("    Press ENTER to confirm, or write a suggestion: ").strip()

            if not user_input or user_input.lower() in ['ok', 'y', 'si', 'yes']:
                print("    [✓] Keywords approved!")
                break
            
            keyword_hint = user_input
            print(f"    [!] Recalculating...")

        # PHASE 2: TABLE SELECTION WITH LOOP
        print(f"\nPHASE 2: TABLE SELECTION (Agent is inspecting files...)")

        table_scores = {}
        for kw in enriched_keywords:
            for index_kw, files in self.inverted_index.items():
                score = fuzz.ratio(kw, index_kw)
                if score >= 80:
                    for f in files:
                        table_scores[f] = table_scores.get(f, 0.0) + (score / 100.0)

        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        top_10 = [t[0] for t in sorted_tables if t[1] > 0.0][:10] or self.all_available_files[:5]

        explorer_agent = ReActAgent(
            name="data_explorer",
            tools=agent_tools,
            llm=self.llm_versatile,
            verbose=True,
            max_iterations=15,
            system_prompt="You are a Data Architect. Inspect CSVs. AS SOON as you know which files are needed, call the 'confirm_table_selection' tool. You MUST provide both the filenames and a brief reasoning of your choice for the Data Scientist."
        )

        token_counter = next((h for h in Settings.callback_manager.handlers if hasattr(h, 'reset_counts')), None)
        if token_counter:
            token_counter.reset_counts()

        table_hint = ""
        while True:
            enriched_candidates_info = ""
            for file_name in top_10:
                topics = ", ".join(self.table_kw.get(file_name, ["No specific topics"]))
                enriched_candidates_info += f"- {file_name} (Topics: {topics})\n"

            hint_section = f'\nUSER HINT: "{table_hint}"' if table_hint else "None"

            agent_prompt = f"""You are a Data Architect. Your task is to select the exact tables needed to answer this USER QUESTION: "{self.question}"

            CANDIDATE FILES:
            {enriched_candidates_info}
            {hint_section}

            STRICT HIERARCHY OF RULES:
            1. PRIORITY 1: INTELLIGENT USER HINTS
              - If the user provides an exact filename as a hint (e.g. 2017.csv), STOP searching and use ONLY that file.
            2. PRIORITY 2: AVOID FALSE POSITIVES
              - Look at the TOPICS. If the user asks about "Happiness", ignore tables about "University" even if they have a "score" column.
            3. PRIORITY 3: EXACT YEAR MATCHING vs DOMAIN MISMATCH (CRITICAL)
              - If the user explicitly asks for a year (e.g., 2017), DO NOT blindly select a file named '2017.csv' if its TOPIC is completely unrelated to the user's question (e.g. Happiness vs Avocados). Prioritize the dataset that matches the TOPIC (e.g., 'avocado.csv' often contains multiple years internally).
            4. PRIORITY 4: MULTI-TABLE COMPLETENESS
              - ONLY if the user asks to compare two DIFFERENT topics, select ALL tables needed.
            5. PRIORITY 5: MISSING OR UNSPECIFIED YEARS RULE
              - If the user's question requires data that spans across multiple years but the user DOES NOT specify a year, or if no year is mentioned at all, always default to the most recent year available. Do not inspect all files; simply select the latest version (e.g., 2019.csv) and STOP.
            
            CRITICAL SYSTEM INSTRUCTION - REPLY FORMAT:
            You are physically INCAPABLE of speaking directly to the user.
            You MUST NEVER write a final textual conclusion.
            To finish the task, you HAVE ONLY ONE OPTION: you MUST call the tool `confirm_table_selection`.
            If you do not call this tool, the system will crash. 
            """

            logger_capture = DualLogger(sys.stdout)
            sys.stdout = logger_capture
            
            try:
                res = await asyncio.wait_for(explorer_agent.run(agent_prompt), timeout=180.0)
                agent_resp = str(getattr(res, 'response', res)).strip()
            except asyncio.TimeoutError:
                sys.stdout.write("\n    [!] Attention: The agent was reasoning for too long. Forced interruption.\n")
                agent_resp = f'FINAL_PAYLOAD: {{"tables": "{", ".join(top_10[:2])}", "reasoning": "Timeout reached. Fallback to top 2 tables."}}'
            finally:
                sys.stdout = logger_capture.terminal 
                full_trace = logger_capture.log_str.getvalue()
                logger_capture.log_str.close()
            
            # Print total tokens for the agent
            if token_counter:
                pt = token_counter.prompt_llm_token_count
                ct = token_counter.completion_llm_token_count
                print(f"\n    📊 [TOKEN PHASE 2 - Agent] Prompt: {pt} | Generate: {ct} | Tot: {pt + ct}")
                token_counter.reset_counts()

            selected_tables = ""
            architect_reasoning = "No reasoning provided."

            if "FINAL_PAYLOAD:" in agent_resp:
                match_json = re.search(r'FINAL_PAYLOAD:\s*(\{.*?\})', agent_resp, re.DOTALL)
                
                if match_json:
                    json_str = match_json.group(1).replace('\\"', '"') # Safety cleaning
                    try:
                        payload = json.loads(json_str)
                        selected_tables = payload.get("tables", "")
                        architect_reasoning = payload.get("reasoning", "")
                    except json.JSONDecodeError:
                        selected_tables = json_str
                else:
                    selected_tables = agent_resp
            else:
                match_tables = re.search(r'(?i)TABLES:\s*(.*)', agent_resp)
                if match_tables:
                    selected_tables = match_tables.group(1).replace("tables:", "").strip()

            selected_tables = selected_tables.replace("'", "").replace('"', "")
            
            selected = [f.strip() for f in selected_tables.split(',') if f.strip() in self.all_available_files]
            if not selected:
                selected = top_10[:2]

            print(f"\n    [💡] Architect Reasoning: {architect_reasoning}")

            user_input_tables = input(f"    Tables ({selected}) - Press ENTER to confirm, or write a suggestion: ").strip()
            
            if not user_input_tables or user_input_tables.lower() in ['ok', 'y', 'si', 'yes']:
                print("    [✓] Table selection approved!")
                break

            table_hint = user_input_tables
            print(f"    [!] Recalculating based on: '{table_hint}'...")

        # Save keywords for logging
        self.raw_model_keywords = raw_content
        self.final_keywords = enriched_keywords

        # Turning off the callback to not interfere with Phases 3 and 5
        Settings.callback_manager = CallbackManager([])

        return TableSelectionEvent(selected_files=selected, reasoning=architect_reasoning, full_trace=full_trace)

    @step
    async def generate_or_correct_code(self, ev: TableSelectionEvent | CodeErrorEvent) -> ExecutionEvent | FinalResultEvent:
        """PHASE 3: Generate or correct code based on the chosen tables."""

        # Saving state at the first pass
        if isinstance(ev, TableSelectionEvent):
            self.arch_reasoning = ev.reasoning
            self.agent_full_trace = ev.full_trace
            self.tables_info = get_filtered_tables_info(ev.selected_files)
            self.selected_files_list = ev.selected_files
            retries = 0
            print(f"\nPHASE 3: Generating initial code...")
        else:
            retries = ev.retries
            print(f"\n[Corrector] Execution failed. Attempting correction (Attempt {retries}/{self.max_retries})...")
            if retries >= self.max_retries:
                print(f"    [!] Maximum number of attempts ({self.max_retries}) reached.")
                messaggio_di_resa = f"I tried my best, but I encountered a technical error: {getattr(ev, 'error_message', 'Unknown error')}"
                return FinalResultEvent(raw_result=messaggio_di_resa)

        if retries == 0:
            prompt = f"""You are an expert Data Scientist. Your ONLY task is to write a standalone Python script using Pandas to answer the user's question.

            <USER_QUESTION>
            {self.question}
            </USER_QUESTION>

            <DATA_ARCHITECT_INSTRUCTIONS>
            {self.arch_reasoning}
            </DATA_ARCHITECT_INSTRUCTIONS>

            <AVAILABLE_TABLES>
            {self.tables_info}
            </AVAILABLE_TABLES>

            <RULES>
            1. EXACT PATHS: You must use the exact file paths provided in <AVAILABLE_TABLES> inside `pd.read_csv()`. Never invent file paths.
            2. NO PROXIES: Use the exact metrics requested. If a specific column doesn't exist, do not substitute it with a similar one.
            3. JOINING: Use `pd.merge()` with robust `left_on` and `right_on` handling if the question requires crossing multiple tables.
            4. DEFENSIVE FILTERING: When filtering strings, ALWAYS use `.str.lower()` on the DataFrame column and lowercase your search term (e.g., `df[df['city'].str.lower() == 'rome']`).
            5. ARCHITECT FIRST: The Data Architect's instructions take precedence over your own assumptions.
            6. AVOID .apply() FOR DATA CLEANING: When extracting numbers from text columns (e.g., "90 min" -> 90), DO NOT write custom functions with .apply(), as they will crash on NaN float values. You MUST use vectorized Pandas string methods like df['column'].str.extract(r'(\\d+)').astype(float) to ensure missing values are handled gracefully.
            </RULES>

            <OUTPUT_FORMAT>
            Return ONLY valid Python code enclosed in a single Markdown code block (```python ... ```). 
            Do NOT add any conversational text before or after the code block.

            Your code MUST follow this exact structure:

            ```python
            import pandas as pd

            # 1. Load data
            # 2. Process data (apply Architect's logic)
            # 3. Check for empty results and Print

            # Example of required error handling at the end:
            # if final_df.empty or pd.isna(result):
            #     print("ERROR_EMPTY: No matching records found for those filters")
            # else:
            #     if isinstance(result, (pd.Series, pd.DataFrame)):
            #         print(result.to_string(index=False))
            #     else:
            #         print(result)"""
        else:
            prompt = f"""The Python code you previously generated for "{self.question}" resulted in a fatal error:

            ERROR TRACEBACK:
            {ev.error_message}

            DATA ARCHITECT'S INSTRUCTIONS (CRITICAL FOR JOINING):
            "{self.arch_reasoning}"

            AVAILABLE TABLES (YOU MUST USE EXACTLY THESE PATHS):
            {self.tables_info}

            CRITICAL FIX REQUIRED:
            If the error is a FileNotFoundError, it means you hallucinated the file path. Look at the AVAILABLE TABLES list above and USE EXACTLY THOSE PATHS.

            Fix the error and rewrite the complete, working Python script.
            Remember to use `print()` to output the final result.
            Reply EXCLUSIVELY with the raw Python code block. Do not apologize or explain.
            """

        res = await self.llm_versatile.achat([ChatMessage(role="user", content=prompt)])

        if res.raw:
            in_t = res.raw.get('prompt_eval_count', 0)
            out_t = res.raw.get('eval_count', 0)
            print(f"    📊 [TOKEN PHASE 3 (Attempt {retries+2})] Prompt: {in_t} | Generate: {out_t} | Tot: {in_t + out_t}")

        return ExecutionEvent(code=str(res.message.content), retries=retries)

    @step
    async def execute_code(self, ev: ExecutionEvent) -> CodeErrorEvent | FinalResultEvent:
        """PHASE 4: Execute the Python script in a separate process."""
        print(f"\nPHASE 4: Executing Python script...")
        code = ev.code

        # Extracting clean code
        match = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = code.replace("```python", "").replace("```", "").strip()

        # Security check
        forbidden = ["import os", "import sys", "import shutil", "subprocess", "eval(", "exec("]
        if any(f in code for f in forbidden):
            return CodeErrorEvent(error_message="Security Error: Forbidden libraries used.", retries=ev.retries + 1)

        # Creating folder and script file
        coding_dir = BASE_DIR / "coding"
        coding_dir.mkdir(exist_ok=True)
        filepath = coding_dir / "script.py"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        raw_kw = getattr(self, 'raw_model_keywords', '')
        trace_to_log = raw_kw
        reasoning_to_log = getattr(self, 'arch_reasoning', 'Reasoning not available')
        tabelle_log = getattr(self, 'selected_files_list', None)
        full_trace_to_log = getattr(self, 'agent_full_trace', '')

        try:
            # Isolated execution with 15 second timeout
            result = subprocess.run([sys.executable, str(filepath)], capture_output=True, text=True, timeout=15)
            
            final_kw = getattr(self, 'final_keywords', None)

            if result.returncode == 0:
                print("    [✓] Execution completed successfully!")
                # Store state for phase 5 logging
                self.log_payload = dict(code=code, raw_result=result.stdout.strip(), retries=ev.retries,
                                        reasoning=reasoning_to_log, tables=tabelle_log, raw_kw=raw_kw, final_kw=final_kw, debug_raw=trace_to_log, full_trace=full_trace_to_log)
                return FinalResultEvent(raw_result=result.stdout.strip())
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                print(f"    [!] Error during code execution.")
                save_experiment_log(self.question, code, f"[EXECUTION ERROR]:\n{error_msg}", ev.retries, reasoning=reasoning_to_log, tables=tabelle_log, raw_keywords=raw_kw, final_keywords=final_kw, debug_raw=trace_to_log, full_trace=full_trace_to_log)
                return CodeErrorEvent(error_message=error_msg, retries=ev.retries + 1)
                
        except Exception as e:
            error_msg = str(e)
            print(f"    [!] Critical system error: {error_msg}")
            save_experiment_log(self.question, code, f"[CRITICAL ERROR]:\n{error_msg}", ev.retries, reasoning=reasoning_to_log, tables=tabelle_log, debug_raw=trace_to_log, full_trace=full_trace_to_log)
            return CodeErrorEvent(error_message=error_msg, retries=ev.retries + 1)

    @step
    async def synthesize_response(self, ev: FinalResultEvent) -> StopEvent:
        """PHASE 5: Generate response."""
        print("\nPHASE 5: Generating response...")

        prompt = f"""You are a helpful and conversational data assistant.
        Your task is to read raw data extracted from a database and use it to answer the user's question clearly and naturally.

        ### INSTRUCTIONS:
        1. Direct Answer: Answer the user's question using ONLY the provided data.
        2. Natural Language: Speak like a human. Write 1 or 2 clear sentences.
        3. No Technical Jargon: Do NOT mention "pandas", "dataframes", "Python", "scripts", or "raw data". 
        4. No Formatting: Do NOT copy-paste the table format or row numbers (indexes). Extract the meaning.
        5. Empty Data: If the data shows "Empty DataFrame", "NaN", or has no actual results, politely say that no data was found for this specific request.
        6. Use the user question for create the final answer.

        ### USER QUESTION:
        {self.question}

        ### DATA RESULTS:
        {ev.raw_result}
        """

        try:
            res = await self.llm_instant.achat([
                ChatMessage(role="user", content=prompt)
            ])

            if res.raw:
                in_t = res.raw.get('prompt_eval_count', 0)
                out_t = res.raw.get('eval_count', 0)
                print(f"    📊 [TOKEN PHASE 5] Prompt: {in_t} | Generate: {out_t} | Tot: {in_t + out_t}")
            
            final_answer = str(res.message.content).strip()
                
            # Robust fallback
            if not final_answer:
                final_answer = f"The raw result is:\n{ev.raw_result}"
                
        except Exception as e:
            final_answer = f"An error occurred during response generation. Raw output: {ev.raw_result}"
            print(f"Error Phase 5: {e}")
    
        # Log experiments here
        payload = getattr(self, 'log_payload', None)
        if payload:
            save_experiment_log(
                self.question, payload['code'], payload['raw_result'], payload['retries'],
                reasoning=payload['reasoning'], tables=payload['tables'],
                raw_keywords=payload['raw_kw'], final_keywords=payload['final_kw'],
                debug_raw=payload.get('debug_raw', ''), final_result=final_answer,
                full_trace=payload.get('full_trace', '')
            )
            self.log_payload = None  # Reset to avoid duplicate logging on retries

        return StopEvent(result=final_answer)

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    # Loading pre-calculated indices
    try:
        with open(INDICI_DIR / "table_keywords.json", "r") as f: table_kw = json.load(f)
        with open(INDICI_DIR / "inverted_index.json", "r") as f: inv_index = json.load(f)
    except FileNotFoundError:
        print("❌ Indices not found! Run 'python index.py' first")
        return

    print("🔄 Initializing Token Tracking System...")
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])

    # LLM Configuration with Ollama
    # Available models: gpt-oss:20b (context: 128K), qwen3.5:latest (context: 256K), llama3.1:8b (context: 128k), gemma4:26b
    MODEL_NAME = "gemma4:26b" 
    URL_SERVER = "http://127.0.0.1:11434"

    print(f"🔄 Initializing local Ollama model: '{MODEL_NAME}'...")
    
    llm_versatile = Ollama(
        model=MODEL_NAME,
        base_url=URL_SERVER,
        request_timeout=300.0, 
        temperature=0.1,
        additional_kwargs={
            "num_ctx": 8192,
            # qwen3.5:latest
            #"presence_penalty": 0.1,    # Encourages the model to search for new paths if it gets stuck
            #"frequency_penalty": 0.1,   # Strongly penalizes the repetition of the same "ThinkingBlock" and tool calls
        }
    )

    llm_instant = Ollama(
        model=MODEL_NAME,
        base_url=URL_SERVER,
        request_timeout=300.0, 
        temperature=0.6,
        additional_kwargs={
            "num_ctx": 8192,
            "presence_penalty": 0.1,
            "top_p": 0.7,                # Limits the choice to the best words, avoiding total delirium
            "top_k": 30
        }
    )

    wf = RobustLakeGenWorkflow(
        timeout=900.0,
        llm_instant=llm_instant,
        llm_versatile=llm_versatile,
        inverted_index=inv_index, 
        table_kw=table_kw
    )
    
    query = input("\n🤖 Hi! What do you want to search for in the Data Lake? \n> ")
    
    try:
        result = await wf.run(question=query)

        print("\n" + "="*20 + " FINAL RESULT " + "="*20)
        output = result if isinstance(result, str) else getattr(result, 'result', str(result))
        print(f"\n{output}\n")
        print("="*54)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n[!] A fatal error occurred during the workflow: {error_msg}")
        
        # Recover partial state from workflow
        raw_kw = getattr(wf, 'raw_model_keywords', '')
        final_kw = getattr(wf, 'final_keywords', None)
        reasoning = getattr(wf, 'arch_reasoning', '')
        tables = getattr(wf, 'selected_files_list', None)
        full_trace = getattr(wf, 'agent_full_trace', '')
        
        payload = getattr(wf, 'log_payload', None)
        if payload:
            code = payload.get('code', '')
            raw_result = payload.get('raw_result', '')
            retries = payload.get('retries', 0)
            debug_raw = payload.get('debug_raw', raw_kw)
        else:
            code = ""
            raw_result = ""
            retries = 0
            debug_raw = raw_kw

        save_experiment_log(
            question=query, code=code, result=raw_result, retries=retries,
            reasoning=reasoning, tables=tables, raw_keywords=raw_kw,
            final_keywords=final_kw, debug_raw=debug_raw, full_trace=full_trace,
            error=error_msg
        )

if __name__ == "__main__":
    asyncio.run(main())