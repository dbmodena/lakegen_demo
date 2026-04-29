# Download wordnet and omw-1.4 before running the program (only the first time)
# uv run python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
import os
import sys
import io
import re
import uuid
import json
import math
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
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.workflow import Context

from utils import (
    BASE_DIR,
    DATA_DIR,
    CSV_DIR,
    INDEXES_DIR,
    DB_PATH,
    LOG_DIR,
    extract_query_keywords,
    save_experiment_log,
    DualLogger,
    ThinkingCapture
)

from tools import make_agent_tools
from build_indexes.blend_indexer import BlendIndexer
from prompts.prompt_manager import PromptManager

from client_solr import LocalSolrClient

# ==========================================
# WORKFLOW DEFINITION (Events and Class)
# ==========================================
class KeywordEvent(Event):
    enriched_keywords: list
    raw_content: str
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
    def __init__(self, llm_instant, llm_versatile, solr_client, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_instant = llm_instant
        self.llm_versatile = llm_versatile
        self.solr_client = solr_client
        self.max_retries = 3
        try:
            self.all_available_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
        except FileNotFoundError:
            self.all_available_files = []
        self.prompt_manager = PromptManager()

    @step
    async def generate_keywords(self, ev: StartEvent) -> KeywordEvent:
        """PHASE 1: Generate keywords from the user question."""
        self.question = ev.question

        raw_keywords_str = extract_query_keywords(self.question)
        default_keywords = [k.strip() for k in raw_keywords_str.split(',') if k.strip()]
        
        print(f"\nPHASE 1: KEYWORD GENERATION")
        print(f"    Question: '{self.question}'")
        print(f"    Base keywords: {default_keywords}")

        self.tokens_phase1 = 0
        keyword_hint = ""
        while True:
            system_prompt = self.prompt_manager.render("keyword_generator", "system_prompt")
            user_prompt = self.prompt_manager.render(
                "keyword_generator", "user_prompt",
                question=self.question,
                raw_keywords_str=raw_keywords_str,
                keyword_hint=keyword_hint
            )

            res = await self.llm_instant.achat([
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt)
            ])

            if res.raw:
                in_t = res.raw.get('prompt_eval_count', 0)
                out_t = res.raw.get('eval_count', 0)
                self.tokens_phase1 += in_t + out_t
                print(f"    📊 [TOKEN PHASE 1] Prompt: {in_t} | Generate: {out_t} | Tot: {in_t + out_t}")

            raw_content = str(res.message.content).strip().lower()
            print(f"    🔍 DEBUG raw model output: '{raw_content}'")

            # Takes only valid words
            extracted_words = re.findall(r'\b[a-z0-9_]+\b', raw_content)
            llm_fluff = {"here", "is", "are", "the", "list", "keywords", "output", "of", "sure", "certainly", "based", "on", "and", "or",
                         "voici", "la", "les", "des", "une", "liste", "mots", "cles", "bien", "sur", "certainement", "base", "et", "ou", "de", "pour", "ces"}
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

        # Save keywords for logging
        self.raw_model_keywords = raw_content
        self.final_keywords = enriched_keywords

        return KeywordEvent(enriched_keywords=enriched_keywords, raw_content=raw_content)

    @step
    async def select_tables(self, ev: KeywordEvent) -> TableSelectionEvent:
        """PHASE 2: Select the most relevant tables using the generated keywords."""
        enriched_keywords = ev.enriched_keywords

        print(f"\nPHASE 2: TABLE SELECTION (Agent is inspecting files...)")

        self.tokens_phase2 = 0

        try:
            # Query Solr using the enriched keywords.
            response = self.solr_client.select(
                tokens=enriched_keywords,
                q_op = "OR",
                rows=10
            )
            
            docs = response.get("response", {}).get("docs", [])
            
            top_10 = []
            self.solr_metadata_map = {}
            for doc in docs:
                dataset_id = doc.get("dataset_id")
                resource_id = doc.get("resource_id")
                if dataset_id or resource_id:
                    matched_file = None
                    for f in self.all_available_files:
                        if (dataset_id and dataset_id in f) or (resource_id and resource_id in f):
                            matched_file = f
                            break
                    
                    if matched_file and matched_file not in top_10:
                        top_10.append(matched_file)
                        # Extract tags directly from Solr doc
                        tags = doc.get("tags", [])
                        if not isinstance(tags, list):
                            tags = [str(tags)]
                        else:
                            tags = [str(t) for t in tags]
                            
                        # Solr client parses columns into a list of dicts
                        columns = doc.get("columns", [])
                        cols_name = [c.get("name") for c in columns if c.get("name")]
                        cols_type = [c.get("type") for c in columns if c.get("type")]
                        
                        title = doc.get("title", "")
                        desc = doc.get("description", "")

                        self.solr_metadata_map[matched_file] = {
                            "title": title,
                            "description": desc,
                            "tags": tags,
                            "columns.name": cols_name,
                            "columns.type": cols_type
                        }
            
            # Fallback if Solr returns nothing
            if not top_10:
                print("    [!] No result from Solr or files not found. Fallback on the first 5 available tables.")
                top_10 = self.all_available_files[:5]
                
        except Exception as e:
            print(f"    [!] Error querying Solr: {e}")
            top_10 = self.all_available_files[:5]

        print("    🎯 Top candidate tables selected by Solr indexing:")
        for idx, tbl in enumerate(top_10, 1):
            print(f"       {idx}. {tbl}")

        # Build the BLEND index
        blend_db_name = f"temp_blend_{uuid.uuid4().hex}.db"
        self.blend_db_path = DB_PATH.parent / blend_db_name
        
        try:
            print(f"    🔨 Building BLEND index for {len(top_10)} candidate tables...")
            indexer = BlendIndexer(csv_dir=CSV_DIR, db_path=self.blend_db_path)
            indexer.build_index(specific_files=top_10, silent=True)
            print(f"    [✓] BLEND index ready: {self.blend_db_path.name}\n")
            print("    Start reasoning...")
    
            system_prompt = self.prompt_manager.render("data_architect", "system_prompt")
    
            agent_tools = make_agent_tools(self.blend_db_path)
            explorer_agent = ReActAgent(
                name="data_explorer",
                tools=agent_tools,
                llm=self.llm_versatile,
                verbose=True,
                max_iterations=20,
                system_prompt=system_prompt,
                early_stopping_method='generate'
            )
    
            token_counter = next((h for h in Settings.callback_manager.handlers if hasattr(h, 'reset_counts')), None)
            if token_counter:
                token_counter.reset_counts()
    
            table_hint = ""
            while True:
                enriched_candidates_info = ""
                for file_name in top_10:
                    meta = getattr(self, "solr_metadata_map", {}).get(file_name, {})
                    title = meta.get("title", "Unknown")
                    desc = meta.get("description", "")
                    
                    all_kws = meta.get("tags", [])
                    if not all_kws:
                        all_kws = ["No specific topics"]
                    limited_topics = ", ".join(all_kws[:15]) 
                    
                    enriched_candidates_info += f"- File: {file_name}\n  Title: {title}\n  Topics: {limited_topics}\n\n"
    
                agent_prompt = self.prompt_manager.render(
                    "data_architect", "user_prompt",
                    question=self.question,
                    enriched_candidates_info=enriched_candidates_info,
                    table_hint=table_hint
                )
    
                logger_capture = DualLogger(sys.stdout)
                sys.stdout = logger_capture
    
                thinking_capture = ThinkingCapture()
                dispatcher = get_dispatcher()
                dispatcher.add_event_handler(thinking_capture)
    
                try:
                    res = await asyncio.wait_for(explorer_agent.run(agent_prompt), timeout=240.0)
                    agent_resp = str(getattr(res, 'response', res)).strip()
                except asyncio.TimeoutError:
                    sys.stdout.write("\n    [!] Attention: The agent was reasoning for too long. Forced interruption.\n")
                    agent_resp = f'FINAL_PAYLOAD: {{"tables": "{", ".join(top_10[:2])}", "reasoning": "Timeout reached. Fallback to top 2 tables."}}'
                finally:
                    sys.stdout = logger_capture.terminal
                    full_trace = logger_capture.log_str.getvalue()
                    logger_capture.log_str.close()
                    dispatcher.event_handlers.remove(thinking_capture)
    
                agent_thinking = "\n\n--- [next thinking block] ---\n".join(thinking_capture.parts)
                
                # Print total tokens for the agent
                if token_counter:
                    pt = token_counter.prompt_llm_token_count
                    ct = token_counter.completion_llm_token_count
                    self.tokens_phase2 += pt + ct
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
    
            # Turning off the callback to not interfere with Phases 3 and 5
            Settings.callback_manager = CallbackManager([])
    
            self.agent_thinking = agent_thinking
            return TableSelectionEvent(selected_files=selected, reasoning=architect_reasoning, full_trace=full_trace)
    
        finally:
            # Clean up the temporary BLEND index
            if hasattr(self, 'blend_db_path') and self.blend_db_path.exists():
                try:
                    os.remove(self.blend_db_path)
                    print(f"    🗑️  Temporary BLEND index removed.")
                except Exception as e:
                    print(f"    [!] Could not remove temp BLEND index: {e}")

    @step
    async def generate_or_correct_code(self, ev: TableSelectionEvent | CodeErrorEvent) -> ExecutionEvent | FinalResultEvent:
        """PHASE 3: Generate or correct code based on the chosen tables."""

        # Saving state at the first pass
        if isinstance(ev, TableSelectionEvent):
            self.arch_reasoning = ev.reasoning
            self.agent_full_trace = ev.full_trace
            
            info_lines = [f"AVAILABLE SELECTED TABLES IN '{CSV_DIR}/':"]
            for idx, filename in enumerate(ev.selected_files, 1):
                filepath = os.path.join(CSV_DIR, filename.strip())
                meta = getattr(self, "solr_metadata_map", {}).get(filename, {})
                cols_name = meta.get("columns.name", [])
                cols_type = meta.get("columns.type", [])
                
                if cols_name and len(cols_name) == len(cols_type):
                    cols_with_types = [f"'{n}' ({t})" for n, t in zip(cols_name, cols_type)]
                elif cols_name:
                    cols_with_types = [f"'{n}'" for n in cols_name]
                else:
                    try:
                        df_preview = pd.read_csv(filepath, nrows=5)
                        cols_name = df_preview.columns.tolist()
                        cols_type = [str(dt) for dt in df_preview.dtypes]
                        cols_with_types = [f"'{n}' ({t})" for n, t in zip(cols_name, cols_type)]
                    except Exception:
                        cols_with_types = ["Unknown columns"]
                    
                info_lines.append(f"{idx}. '{filepath}'")
                info_lines.append(f"   Columns: [" + ", ".join(cols_with_types) + "]")
                
            self.tables_info = "\n".join(info_lines)
            self.selected_files_list = ev.selected_files
            self.tokens_phase3 = 0
            retries = 0
            print(f"\nPHASE 3: Generating initial code...")
        else:
            retries = ev.retries
            print(f"\n[Corrector] Execution failed. Attempting correction (Attempt {retries}/{self.max_retries})...")
            if retries >= self.max_retries:
                print(f"    [!] Maximum number of attempts ({self.max_retries}) reached.")
                messaggio_di_resa = f"I tried my best, but I encountered a technical error: {getattr(ev, 'error_message', 'Unknown error')}"
                return FinalResultEvent(raw_result=messaggio_di_resa)

        system_prompt = self.prompt_manager.render("code_generator", "system_prompt")

        if retries == 0:
            user_prompt = self.prompt_manager.render(
                "code_generator", "initial_prompt",
                question=self.question,
                arch_reasoning=self.arch_reasoning,
                tables_info=self.tables_info
            )
        else:
            user_prompt = self.prompt_manager.render(
                "code_generator", "correction_prompt",
                question=self.question,
                error_message=ev.error_message,
                arch_reasoning=self.arch_reasoning,
                tables_info=self.tables_info
            )

        res = await self.llm_versatile.achat([
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt)
        ])

        if res.raw:
            in_t = res.raw.get('prompt_eval_count') or 0
            out_t = res.raw.get('eval_count') or 0
            self.tokens_phase3 += in_t + out_t
            print(f"    📊 [TOKEN PHASE 3 (Attempt {retries+1})] Prompt: {in_t} | Generate: {out_t} | Tot: {in_t + out_t}")

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
        agent_thinking_to_log = getattr(self, 'agent_thinking', '')
        llm_thinking = getattr(self, 'llm_thinking_phase3', '')

        try:
            # Isolated execution with 15 second timeout
            result = subprocess.run([sys.executable, str(filepath)], capture_output=True, text=True, timeout=15)
            
            final_kw = getattr(self, 'final_keywords', None)

            if result.returncode == 0:
                print("    [✓] Execution completed successfully!")
                # Store state for phase 5 logging
                self.log_payload = dict(code=code, raw_result=result.stdout.strip(), retries=ev.retries,
                                        reasoning=reasoning_to_log, tables=tabelle_log, raw_kw=raw_kw, final_kw=final_kw,
                                        debug_raw=trace_to_log, full_trace=full_trace_to_log,
                                        tokens_phase3=getattr(self, 'tokens_phase3', 0),
                                        llm_thinking=llm_thinking,
                                        agent_thinking=agent_thinking_to_log)
                return FinalResultEvent(raw_result=result.stdout.strip())
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                print(f"    [!] Error during code execution.")
                save_experiment_log(self.question, code, f"[EXECUTION ERROR]:\n{error_msg}", ev.retries, reasoning=reasoning_to_log, tables=tabelle_log, raw_keywords=raw_kw, final_keywords=final_kw, debug_raw=trace_to_log, full_trace=full_trace_to_log, tokens_phase3=getattr(self, 'tokens_phase3', 0), llm_thinking=llm_thinking, agent_thinking=agent_thinking_to_log)
                return CodeErrorEvent(error_message=error_msg, retries=ev.retries + 1)
                
        except Exception as e:
            error_msg = str(e)
            print(f"    [!] Critical system error: {error_msg}")
            save_experiment_log(self.question, code, f"[CRITICAL ERROR]:\n{error_msg}", ev.retries, reasoning=reasoning_to_log, tables=tabelle_log, debug_raw=trace_to_log, full_trace=full_trace_to_log, tokens_phase3=getattr(self, 'tokens_phase3', 0), llm_thinking=llm_thinking, agent_thinking=agent_thinking_to_log)
            return CodeErrorEvent(error_message=error_msg, retries=ev.retries + 1)

    @step
    async def synthesize_response(self, ev: FinalResultEvent) -> StopEvent:
        """PHASE 5: Generate response."""
        print("\nPHASE 5: Generating response...")

        prompt = self.prompt_manager.render(
            "synthesizer", "prompt",
            question=self.question,
            raw_result=ev.raw_result
        )

        try:
            res = await self.llm_instant.achat([
                ChatMessage(role="user", content=prompt)
            ])

            if res.raw:
                in_t = res.raw.get('prompt_eval_count', 0)
                out_t = res.raw.get('eval_count', 0)
                tokens_phase5 = in_t + out_t
                print(f"    📊 [TOKEN PHASE 5] Prompt: {in_t} | Generate: {out_t} | Tot: {tokens_phase5}")
            else:
                tokens_phase5 = 0
            
            final_answer = str(res.message.content).strip()
                
            if not final_answer:
                final_answer = f"The raw result is:\n{ev.raw_result}"
                
        except Exception as e:
            final_answer = f"An error occurred during response generation. Raw output: {ev.raw_result}"
            tokens_phase5 = 0
            print(f"Error Phase 5: {e}")
    
        # Log experiments here
        payload = getattr(self, 'log_payload', None)
        if payload:
            save_experiment_log(
                self.question, payload['code'], payload['raw_result'], payload['retries'],
                reasoning=payload['reasoning'], tables=payload['tables'],
                raw_keywords=payload['raw_kw'], final_keywords=payload['final_kw'],
                debug_raw=payload.get('debug_raw', ''), final_result=final_answer,
                full_trace=payload.get('full_trace', ''),
                tokens_phase1=getattr(self, 'tokens_phase1', 0),
                tokens_phase2=getattr(self, 'tokens_phase2', 0),
                tokens_phase3=payload.get('tokens_phase3', 0),
                tokens_phase5=tokens_phase5,
                llm_thinking=payload.get('llm_thinking', ''),
                agent_thinking=payload.get('agent_thinking', '')
            )
            self.log_payload = None  # Reset to avoid duplicate logging on retries

        return StopEvent(result=final_answer)

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    # Cleanup orphan temp db files
    for db_file in DB_PATH.parent.glob("temp_blend_*.db"):
        try:
            os.remove(db_file)
        except Exception:
            pass

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
            "num_ctx": 12288
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

    # Define the solr client
    # Available cores: bologna, valencia, paris
    solr_client = LocalSolrClient(core="paris")

    wf = RobustLakeGenWorkflow(
        timeout=900.0,
        llm_instant=llm_instant,
        llm_versatile=llm_versatile,
        solr_client=solr_client
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