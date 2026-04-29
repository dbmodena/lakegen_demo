"""
LakeGen Interactive – Streamlit Web Application
Run with:  uv run streamlit run app.py
"""

import os
import sys
import re
import io
import json
import uuid
import anyio
import asyncio
import subprocess
import concurrent.futures
from pathlib import Path

import nest_asyncio

import streamlit as st
import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Path bootstrap (same logic as utils.py so imports work from project root)
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_ROOT_DIR))

from src.utils import (
    BASE_DIR, DATA_DIR, INDEXES_DIR, LOG_DIR,
    extract_query_keywords, save_experiment_log, DualLogger, ThinkingCapture
)
from src.tools import make_agent_tools
from src.build_indexes.blend_indexer import BlendIndexer
from prompts.prompt_manager import PromptManager
from src.client_solr import LocalSolrClient

# LlamaIndex
import tiktoken
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.agent import ReActAgent
from llama_index.core.instrumentation import get_dispatcher

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="LakeGen Interactive", page_icon="🌊", layout="wide")

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header bar */
.header-bar {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0ea5e9 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(14,165,233,.25);
}
.header-bar h1 {
    color: #fff; margin: 0; font-size: 2rem; font-weight: 700;
    letter-spacing: -0.5px;
}
.header-bar p { color: #94a3b8; margin: .4rem 0 0; font-size: .95rem; }

/* Phase badges */
.phase-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: .8rem;
    font-weight: 600;
    letter-spacing: .5px;
}
.phase-1 { background: #1e3a5f; color: #38bdf8; }
.phase-2 { background: #1a2e1a; color: #4ade80; }
.phase-3 { background: #2e1a2e; color: #c084fc; }
.phase-5 { background: #2e2a1a; color: #fbbf24; }

/* Keyword chips */
.kw-chip {
    display: inline-block;
    background: rgba(56,189,248,.15);
    color: #38bdf8;
    border: 1px solid rgba(56,189,248,.3);
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px;
    font-size: .85rem;
    font-weight: 500;
}

/* Table card */
.table-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 10px;
    padding: 10px 16px;
    margin: 6px 0;
    transition: background .2s;
}
.table-card:hover { background: rgba(255,255,255,.08); }

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(14,165,233,.08), rgba(74,222,128,.08));
    border: 1px solid rgba(74,222,128,.2);
    border-radius: 14px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Sidebar tweaks */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="header-bar">
    <h1>🌊 LakeGen — Data Assistant</h1>
    <p>Ask natural-language questions over your Data Lake</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SESSION STATE INIT
# ==========================================
_defaults = {
    "phase": "idle",
    "query": "",
    "keywords": [],
    "tables": [],
    "candidates": [],
    "chat_history": [],
    "architect_reasoning": "",
    "full_trace": "",
    "solr_metadata_map": {},
    "tokens": {"p1": 0, "p2": 0, "p3": 0, "p5": 0},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# SIDEBAR CONFIG
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")

    ollama_url = st.text_input("Ollama Server URL", value="http://127.0.0.1:11434")
    model_name = st.selectbox("Model", [
        "gemma4:26b", "qwen3.5:latest", "llama3.1:8b", "gpt-oss:20b"
    ])
    solr_core = st.selectbox("Solr Core (dataset)", ["valencia", "bologna", "paris"])
    num_ctx = st.slider("Context window (num_ctx)", 4096, 32768, 12288, step=1024)

    # dynamically adjust paths based on solr_core
    st.session_state.csv_dir = BASE_DIR / f"data/{solr_core}/datasets/csv"
    st.session_state.db_path = BASE_DIR / f"data/blend_{solr_core}.db"

    st.divider()
    if st.button("🔄 Reset Conversation", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

# ==========================================
# LLM + SOLR HELPERS
# ==========================================
@st.cache_resource
def get_llms(model, url, ctx):
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
    )
    Settings.callback_manager = CallbackManager([token_counter])
    llm_versatile = Ollama(model=model, base_url=url, request_timeout=300.0,
                           temperature=0.1, additional_kwargs={"num_ctx": ctx})
    llm_instant = Ollama(model=model, base_url=url, request_timeout=300.0,
                         temperature=0.6,
                         additional_kwargs={"num_ctx": min(ctx, 8192),
                                            "presence_penalty": 0.1,
                                            "top_p": 0.7, "top_k": 30})
    return llm_versatile, llm_instant, token_counter

@st.cache_resource
def get_solr(core):
    return LocalSolrClient(core=core)

@st.cache_resource
def get_prompt_manager():
    return PromptManager()

def get_all_csv_files(csv_dir):
    try:
        return [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    except FileNotFoundError:
        return []

# ==========================================
# PHASE 1 – KEYWORD GENERATION
# ==========================================
def phase1_generate_keywords(query, llm_instant, pm, hint=""):
    raw_keywords_str = extract_query_keywords(query)
    default_keywords = [k.strip() for k in raw_keywords_str.split(",") if k.strip()]

    system_prompt = pm.render("keyword_generator", "system_prompt")
    user_prompt = pm.render("keyword_generator", "user_prompt",
                            question=query, raw_keywords_str=raw_keywords_str,
                            keyword_hint=hint)

    res = llm_instant.chat([
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ])

    tokens = 0
    if res.raw:
        tokens = (res.raw.get("prompt_eval_count", 0) +
                  res.raw.get("eval_count", 0))

    raw_content = str(res.message.content).strip().lower()
    extracted = re.findall(r"\b[a-z0-9_]+\b", raw_content)
    fluff = {"here","is","are","the","list","keywords","output","of",
             "sure","certainly","based","on","and","or",
             "voici","la","les","des","une","liste","mots","cles","bien",
             "sur","certainement","base","et","ou","de","pour","ces"}
    brute = [w for w in extracted if w not in fluff]

    enriched = default_keywords.copy()
    for w in brute:
        if w not in enriched:
            enriched.append(w)
    enriched = enriched[:len(default_keywords) + 5]

    return enriched, raw_content, tokens

# ==========================================
# PHASE 2 – TABLE SELECTION (Solr + Agent)
# ==========================================
def phase2_select_tables(query, keywords, llm_versatile, solr_client,
                         pm, all_files, csv_dir, db_path, hint=""):
    # --- Solr retrieval ---
    top_10, solr_meta = [], {}
    try:
        response = solr_client.select(tokens=keywords, q_op="OR", rows=10)
        docs = response.get("response", {}).get("docs", [])
        for doc in docs:
            did = doc.get("dataset_id")
            rid = doc.get("resource_id")
            matched = None
            for f in all_files:
                if (did and did in f) or (rid and rid in f):
                    matched = f
                    break
            if matched and matched not in top_10:
                top_10.append(matched)
                tags = doc.get("tags", [])
                if not isinstance(tags, list):
                    tags = [str(tags)]
                columns = doc.get("columns", [])
                cols_name = [c.get("name") for c in columns if c.get("name")]
                cols_type = [c.get("type") for c in columns if c.get("type")]
                solr_meta[matched] = {
                    "title": doc.get("title", ""),
                    "description": doc.get("description", ""),
                    "tags": [str(t) for t in tags],
                    "columns.name": cols_name,
                    "columns.type": cols_type,
                }
        if not top_10:
            top_10 = all_files[:5]
    except Exception:
        top_10 = all_files[:5]

    # --- BLEND index + ReAct Agent ---
    blend_db = db_path.parent / f"temp_blend_{uuid.uuid4().hex}.db"
    try:
        indexer = BlendIndexer(csv_dir=csv_dir, db_path=blend_db)
        indexer.build_index(specific_files=top_10, silent=True)

        architect_system_prompt = pm.render("data_architect", "system_prompt")
        agent_tools = make_agent_tools(blend_db)

        token_counter = next(
            (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
            None,
        )
        if token_counter:
            token_counter.reset_counts()

        enriched_info = ""
        for fn in top_10:
            meta = solr_meta.get(fn, {})
            title = meta.get("title", "Unknown")
            topics = ", ".join(meta.get("tags", ["No specific topics"])[:15])
            enriched_info += f"- File: {fn}\n  Title: {title}\n  Topics: {topics}\n\n"

        agent_prompt = pm.render("data_architect", "user_prompt",
                                 question=query,
                                 enriched_candidates_info=enriched_info,
                                 table_hint=hint)

        # Capture stdout
        old_stdout = sys.stdout
        capture = io.StringIO()
        sys.stdout = capture

        thinking_capture = ThinkingCapture()
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(thinking_capture)

        try:
            async def _run_agent():
                explorer = ReActAgent(
                    name="data_explorer", tools=agent_tools, llm=llm_versatile,
                    verbose=False, max_iterations=20,
                    system_prompt=architect_system_prompt,
                    early_stopping_method="generate",
                )
                return await explorer.run(agent_prompt)

            res = anyio.run(_run_agent)
            agent_resp = str(getattr(res, "response", res)).strip()
        except Exception as agent_err:
            agent_resp = (
                f'FINAL_PAYLOAD: {{"tables": "{", ".join(top_10[:2])}",'
                f' "reasoning": "Agent error: {str(agent_err)[:80]}. Fallback to top 2."}}'
            )
        finally:
            sys.stdout = old_stdout
            full_trace = capture.getvalue()
            capture.close()
            dispatcher.event_handlers.remove(thinking_capture)

        tokens_p2 = 0
        if token_counter:
            tokens_p2 = (token_counter.prompt_llm_token_count +
                         token_counter.completion_llm_token_count)
            token_counter.reset_counts()

        # Parse agent response
        reasoning = "No reasoning provided."
        selected_str = ""
        if "FINAL_PAYLOAD:" in agent_resp:
            m = re.search(r'FINAL_PAYLOAD:\s*(\{.*?\})', agent_resp, re.DOTALL)
            if m:
                try:
                    payload = json.loads(m.group(1).replace('\\"', '"'))
                    selected_str = payload.get("tables", "")
                    reasoning = payload.get("reasoning", "")
                except json.JSONDecodeError:
                    selected_str = m.group(1)
            else:
                selected_str = agent_resp
        else:
            mt = re.search(r"(?i)TABLES:\s*(.*)", agent_resp)
            if mt:
                selected_str = mt.group(1).strip()

        selected_str = selected_str.replace("'", "").replace('"', "")
        selected = [f.strip() for f in selected_str.split(",")
                     if f.strip() in all_files]
        if not selected:
            selected = top_10[:2]

        # Reset callback so it doesn't interfere with Phase 3
        Settings.callback_manager = CallbackManager([])

        return selected, top_10, solr_meta, reasoning, full_trace, tokens_p2

    finally:
        if blend_db.exists():
            try:
                os.remove(blend_db)
            except Exception:
                pass

# ==========================================
# PHASE 3 – CODE GENERATION
# ==========================================
def phase3_generate_code(query, tables, solr_meta, reasoning,
                         llm_versatile, pm, csv_dir, retries=0, error_msg=""):
    info_lines = [f"AVAILABLE SELECTED TABLES IN '{csv_dir}/':"]
    for idx, fn in enumerate(tables, 1):
        filepath = os.path.join(csv_dir, fn.strip())
        meta = solr_meta.get(fn, {})
        cn = meta.get("columns.name", [])
        ct = meta.get("columns.type", [])
        if cn and len(cn) == len(ct):
            cols = [f"'{n}' ({t})" for n, t in zip(cn, ct)]
        elif cn:
            cols = [f"'{n}'" for n in cn]
        else:
            try:
                df = pd.read_csv(filepath, nrows=5)
                cols = [f"'{n}' ({t})" for n, t in
                        zip(df.columns, [str(d) for d in df.dtypes])]
            except Exception:
                cols = ["Unknown columns"]
        info_lines.append(f"{idx}. '{filepath}'")
        info_lines.append(f"   Columns: [" + ", ".join(cols) + "]")
    tables_info = "\n".join(info_lines)

    system_prompt = pm.render("code_generator", "system_prompt")
    if retries == 0:
        user_prompt = pm.render("code_generator", "initial_prompt",
                                question=query, arch_reasoning=reasoning,
                                tables_info=tables_info)
    else:
        user_prompt = pm.render("code_generator", "correction_prompt",
                                question=query, error_message=error_msg,
                                arch_reasoning=reasoning,
                                tables_info=tables_info)

    res = llm_versatile.chat([
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ])

    tokens = 0
    if res.raw:
        tokens = (res.raw.get("prompt_eval_count", 0) +
                  res.raw.get("eval_count", 0))

    return str(res.message.content), tokens

# ==========================================
# PHASE 4 – CODE EXECUTION
# ==========================================
def phase4_execute(code_raw):
    match = re.search(r"```python\n(.*?)\n```", code_raw, re.DOTALL)
    code = match.group(1).strip() if match else code_raw.replace("```python","").replace("```","").strip()

    forbidden = ["import os","import sys","import shutil","subprocess","eval(","exec("]
    if any(f in code for f in forbidden):
        return None, "Security Error: Forbidden libraries used.", code

    coding_dir = BASE_DIR / "coding"
    coding_dir.mkdir(exist_ok=True)
    fp = coding_dir / "script.py"
    fp.write_text(code, encoding="utf-8")

    try:
        result = subprocess.run([sys.executable, str(fp)],
                                capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return result.stdout.strip(), None, code
        return None, (result.stderr.strip() or result.stdout.strip()), code
    except Exception as e:
        return None, str(e), code

# ==========================================
# PHASE 5 – SYNTHESIS
# ==========================================
def phase5_synthesize(query, raw_result, llm_instant, pm):
    prompt = pm.render("synthesizer", "prompt",
                       question=query, raw_result=raw_result)
    res = llm_instant.chat([ChatMessage(role="user", content=prompt)])
    tokens = 0
    if res.raw:
        tokens = (res.raw.get("prompt_eval_count", 0) +
                  res.raw.get("eval_count", 0))
    return str(res.message.content).strip(), tokens

# ==========================================
# RENDER CHAT HISTORY
# ==========================================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "code" in msg:
            with st.expander("👀 View generated Python code"):
                st.code(msg["code"], language="python")

# ==========================================
# PHASE ROUTER
# ==========================================
llm_v, llm_i, _tc = get_llms(model_name, ollama_url, num_ctx)
solr = get_solr(solr_core)
pm = get_prompt_manager()
all_csv = get_all_csv_files(st.session_state.csv_dir)

# --- IDLE: query input ---
if st.session_state.phase == "idle":
    query = st.chat_input("Ask a question about your data...")
    if query:
        st.session_state.query = query
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("🔍 Extracting keywords…"):
                kws, raw, tok = phase1_generate_keywords(query, llm_i, pm)
                st.session_state.keywords = kws
                st.session_state.tokens["p1"] = tok
        st.session_state.phase = "keyword_approval"
        st.rerun()

# --- KEYWORD APPROVAL ---
elif st.session_state.phase == "keyword_approval":
    with st.chat_message("assistant"):
        st.markdown('<span class="phase-badge phase-1">PHASE 1</span> **Keyword Generation**',
                    unsafe_allow_html=True)
        chips = " ".join(f'<span class="kw-chip">{k}</span>'
                         for k in st.session_state.keywords)
        st.markdown(f"Extracted keywords: {chips}", unsafe_allow_html=True)

        container = st.empty()
        with container.container():
            hint = st.text_input("Modify keywords? (e.g. 'add inflation')",
                                 key="hint_kw")
            c1, c2 = st.columns(2)
            approve = c1.button("✅ Approve & Proceed", use_container_width=True)
            recalc = c2.button("🔄 Recalculate", use_container_width=True)

        if approve:
            container.empty()
            kw_list = ", ".join(st.session_state.keywords)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"✅ **Keywords confirmed:** `{kw_list}`"
                           + (f"\n*(hint: {hint})*" if hint else "")
            })
            with st.spinner("🗂️ Searching & selecting tables…"):
                sel, cands, smeta, reasoning, trace, tok2 = phase2_select_tables(
                    st.session_state.query, st.session_state.keywords,
                    llm_v, solr, pm, all_csv, 
                    st.session_state.csv_dir, st.session_state.db_path, hint=hint)
                st.session_state.tables = sel
                st.session_state.candidates = cands
                st.session_state.solr_metadata_map = smeta
                st.session_state.architect_reasoning = reasoning
                st.session_state.full_trace = trace
                st.session_state.tokens["p2"] = tok2
            st.session_state.phase = "table_approval"
            st.rerun()

        elif recalc:
            container.empty()
            with st.spinner("Recalculating keywords…"):
                kws, _, tok = phase1_generate_keywords(
                    st.session_state.query, llm_i, pm, hint=hint)
                st.session_state.keywords = kws
                st.session_state.tokens["p1"] += tok
            st.rerun()

# --- TABLE APPROVAL ---
elif st.session_state.phase == "table_approval":
    with st.chat_message("assistant"):
        st.markdown('<span class="phase-badge phase-2">PHASE 2</span> **Table Selection**',
                    unsafe_allow_html=True)
        st.markdown("The AI selected these tables for analysis:")
        for t in st.session_state.tables:
            st.markdown(f'<div class="table-card">📄 <code>{t}</code></div>',
                        unsafe_allow_html=True)

        if st.session_state.architect_reasoning:
            with st.expander("💡 Architect Reasoning"):
                st.markdown(st.session_state.architect_reasoning)

        with st.expander("🔎 All candidate tables (Solr top-10)"):
            if st.session_state.candidates:
                for c in st.session_state.candidates:
                    st.markdown(f"- `{c}`")
            else:
                st.info("No candidates found.")

        container = st.empty()
        with container.container():
            hint_tb = st.text_input("Suggestions? (e.g. 'Add 2019.csv')",
                                    key="hint_tb")
            c1, c2 = st.columns(2)
            approve = c1.button("🚀 Approve & Run Code",
                                use_container_width=True)
            recalc = c2.button("🔄 Recalculate Tables",
                               use_container_width=True)

        if approve:
            container.empty()
            tbl_md = "\n".join(f"- `{t}`" for t in st.session_state.tables)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"✅ **Tables confirmed:**\n{tbl_md}"
                           + (f"\n*(hint: {hint_tb})*" if hint_tb else "")
            })
            st.session_state.phase = "execution"
            st.rerun()

        elif recalc:
            container.empty()
            with st.spinner("Re-selecting tables…"):
                sel, cands, smeta, reasoning, trace, tok2 = phase2_select_tables(
                    st.session_state.query, st.session_state.keywords,
                    llm_v, solr, pm, all_csv, 
                    st.session_state.csv_dir, st.session_state.db_path, hint=hint_tb)
                st.session_state.tables = sel
                st.session_state.candidates = cands
                st.session_state.solr_metadata_map = smeta
                st.session_state.architect_reasoning = reasoning
                st.session_state.full_trace = trace
                st.session_state.tokens["p2"] += tok2
            st.rerun()

# --- EXECUTION (Phase 3 → 4 → 5) ---
elif st.session_state.phase == "execution":
    with st.chat_message("assistant"):
        max_retries = 3
        retries = 0
        error_msg = ""
        final_code = ""

        progress = st.empty()

        # Phase 3+4: generate code & execute, with retry loop
        while retries < max_retries:
            progress.markdown(
                f'<span class="phase-badge phase-3">PHASE 3</span> '
                f'Generating code… (attempt {retries+1}/{max_retries})',
                unsafe_allow_html=True)

            with st.spinner("✍️ Writing code…"):
                code_raw, tok3 = phase3_generate_code(
                    st.session_state.query, st.session_state.tables,
                    st.session_state.solr_metadata_map,
                    st.session_state.architect_reasoning,
                    llm_v, pm, st.session_state.csv_dir, 
                    retries=retries, error_msg=error_msg)
                st.session_state.tokens["p3"] += tok3

            with st.spinner("⚡ Executing script…"):
                raw_result, err, clean_code = phase4_execute(code_raw)
                final_code = clean_code

            if err is None:
                break
            error_msg = err
            retries += 1

        progress.empty()

        if raw_result is None:
            raw_result = f"Execution failed after {max_retries} attempts. Last error: {error_msg}"

        # Phase 5: synthesis
        with st.spinner("📝 Synthesizing final answer…"):
            answer, tok5 = phase5_synthesize(
                st.session_state.query, raw_result, llm_i, pm)
            st.session_state.tokens["p5"] = tok5

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"### 📊 Final Result\n{answer}",
            "code": final_code,
        })
        st.session_state.phase = "idle"
        st.rerun()
