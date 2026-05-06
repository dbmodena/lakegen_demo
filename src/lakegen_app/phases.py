import io
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import anyio
import pandas as pd
from llama_index.core import Settings
# from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import AgentStream, ToolCall, ToolCallResult, FunctionAgent
from llama_index.core.callbacks import CallbackManager
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import ChatMessage, LLM

from lakegen_app.phase2_logging import (
    Phase2AgentStall,
    detect_phase2_agent_stall,
    format_cli_log_value,
    format_phase2_solr_results,
    format_phase2_tool_args,
    format_phase2_tool_call,
    format_phase2_tool_result,
)
from lakegen_app.types import Phase2SelectionResult, SolrMetadata, StreamCallback
from prompts.prompt_manager import PromptManager
from src.build_indexes.blend_indexer import BlendIndexer
from src.client_solr import LocalSolrClient
from src.tools import make_agent_tools
from src.utils import BASE_DIR, ThinkingCapture, extract_query_keywords


def phase1_generate_keywords(
    query: str,
    llm_instant: LLM,
    pm: PromptManager,
    hint="",
    stream_placeholder=None,
):
    raw_keywords_str = extract_query_keywords(query)
    default_keywords = [k.strip() for k in raw_keywords_str.split(",") if k.strip()]

    system_prompt = pm.render(
        "keyword_generator",
        "system_prompt"
    )

    user_prompt = pm.render(
        "keyword_generator",
        "user_prompt",
        question=query,
        raw_keywords_str=raw_keywords_str,
        keyword_hint=hint
    )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    raw_stream = ""
    tokens = 0
    print("[phase1 keyword stream] ", end="", flush=True)
    for chunk in llm_instant.stream_chat(messages):
        delta = chunk.delta or ""
        if delta:
            raw_stream += delta
            print(delta, end="", flush=True)
            if stream_placeholder is not None:
                stream_placeholder.markdown(raw_stream)

        if chunk.raw:
            prompt_tokens = chunk.raw.get("prompt_eval_count") or 0
            completion_tokens = chunk.raw.get("eval_count") or 0
            if prompt_tokens or completion_tokens:
                tokens = prompt_tokens + completion_tokens
    print("", flush=True)

    raw_content = raw_stream.strip().lower()
    extracted = re.findall(r"\b[a-z0-9_]+\b", raw_content)
    fluff = {"here","is","are","the","list","keywords","output","of",
             "sure","certainly","based","on","and","or",
             "voici","la","les","des","une","liste","mots","cles","bien",
             "sur","certainement","base","et","ou","de","pour","ces"}
    brute = [w for w in extracted if w not in fluff]

    enriched = []
    for w in brute:
        if w not in enriched:
            enriched.append(w)

    if not enriched:
        enriched = default_keywords.copy()

    enriched = enriched[:15]

    return enriched, raw_content, tokens


def _emit_phase2_activity(
    activity_log_parts: list[str],
    stream_callback: StreamCallback | None,
    delta: str,
) -> None:
    if not delta:
        return
    activity_log_parts.append(delta)
    if stream_callback is not None:
        stream_callback(delta)


def _match_local_csv(doc: dict, all_files: list[str]) -> str | None:
    dataset_id = doc.get("dataset_id")
    resource_id = doc.get("resource_id")
    return next(
        (
            filename
            for filename in all_files
            if (dataset_id and dataset_id in filename)
            or (resource_id and resource_id in filename)
        ),
        None,
    )


def _solr_metadata_from_doc(doc: dict) -> dict[str, object]:
    tags = doc.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]

    columns = doc.get("columns", [])
    return {
        "title": doc.get("title", ""),
        "description": doc.get("description", ""),
        "tags": [str(tag) for tag in tags],
        "columns.name": [col.get("name") for col in columns if col.get("name")],
        "columns.type": [col.get("type") for col in columns if col.get("type")],
    }


def phase2_retrieve_candidates(
    keywords: list[str],
    solr_client: LocalSolrClient,
    all_files: list[str],
    csv_dir: Path,
    db_path: Path,
    skip_solr: bool = False,
    top_10: list[str] | None = None,
    solr_meta: SolrMetadata | None = None,
    stream_callback: StreamCallback | None = None,
) -> tuple[list[str], SolrMetadata, list[str], Path]:
    """Part 1 of Phase 2: retrieve candidates and prepare their local index."""
    activity_log_parts: list[str] = []

    if skip_solr:
        candidates = top_10 or []
        metadata = solr_meta or {}
        heading = "Reusing Solr candidate tables"
        print(f"[phase2 candidates] reusing {len(candidates)} candidates", flush=True)
    else:
        candidates: list[str] = []
        metadata: SolrMetadata = {}
        query_text = " ".join(keywords)
        heading = "Solr candidate tables"

        try:
            print(
                "\n[phase2 candidates] Solr search "
                f"q={format_cli_log_value(query_text)} "
                f"csv_dir={csv_dir} csv_count={len(all_files)}",
                flush=True,
            )
            solr_response = solr_client.select(tokens=keywords, q_op="OR", rows=30)
            response_body = solr_response.get("response", {})
            docs = response_body.get("docs", [])
            print(
                "[phase2 candidates] Solr response "
                f"numFound={response_body.get('numFound', 'unknown')} "
                f"docs_returned={len(docs)}",
                flush=True,
            )

            for doc in docs:
                matched = _match_local_csv(doc, all_files)
                if matched is None or matched in candidates:
                    continue

                candidates.append(matched)
                metadata[matched] = _solr_metadata_from_doc(doc)
                if len(candidates) >= 10:
                    break

            if not candidates:
                candidates = all_files[:5]
                print(
                    "[phase2 candidates] no local Solr matches; "
                    f"fallback={candidates}",
                    flush=True,
                )
            else:
                print(f"[phase2 candidates] matched={candidates}", flush=True)
        except Exception as solr_err:
            candidates = all_files[:5]
            print(
                "[phase2 candidates] Solr error "
                f"{type(solr_err).__name__}: {solr_err}; fallback={candidates}",
                flush=True,
            )

    _emit_phase2_activity(
        activity_log_parts,
        stream_callback,
        format_phase2_solr_results(candidates, metadata, heading)
        + "\n\n---\n\n**Phase 2 activity log**\n",
    )

    blend_db = db_path.parent / f"temp_blend_{uuid.uuid4().hex}.db"
    try:
        print(
            f"[phase2 candidates] building BLEND index db={blend_db.name} "
            f"files={candidates}",
            flush=True,
        )
        _emit_phase2_activity(
            activity_log_parts,
            stream_callback,
            "\n**Preparing BLEND index**\n"
            f"- Candidate files: `{len(candidates)}`\n"
            f"- Temporary DB: `{blend_db.name}`\n",
        )

        indexer = BlendIndexer(csv_dir=csv_dir, db_path=blend_db)
        indexer.build_index(specific_files=candidates, silent=True)
        print(f"[phase2 candidates] BLEND ready db={blend_db.name}", flush=True)
        _emit_phase2_activity(activity_log_parts, stream_callback, "- Status: `ready`\n")
    except Exception:
        if blend_db.exists():
            try:
                os.remove(blend_db)
            except Exception:
                pass
        raise

    return candidates, metadata, activity_log_parts, blend_db


def _format_candidate_context(candidates: list[str], solr_meta: SolrMetadata) -> str:
    lines = []
    for filename in candidates:
        meta = solr_meta.get(filename, {})
        title = meta.get("title", "Unknown")
        topics = ", ".join(meta.get("tags", ["No specific topics"])[:15])
        lines.append(f"- File: {filename}\n  Title: {title}\n  Topics: {topics}")
    return "\n\n".join(lines) + ("\n" if lines else "")


def _parse_table_selector_response(
    agent_resp: str,
    all_files: list[str],
    candidates: list[str],
) -> tuple[list[str], str]:
    reasoning = "No reasoning provided."
    selected_str = ""

    if "REJECT_KEYWORDS" in agent_resp:
        reason = agent_resp.replace("REJECT_KEYWORDS:", "", 1)
        reason = reason.replace("REJECT_KEYWORDS", "", 1).strip()
        return [], f"REJECT_KEYWORDS: {reason}"

    if "FINAL_PAYLOAD:" in agent_resp:
        match = re.search(r"FINAL_PAYLOAD:\s*(\{.*?\})", agent_resp, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(1).replace('\\"', '"'))
                selected_str = payload.get("tables", "")
                reasoning = payload.get("reasoning", "")
            except json.JSONDecodeError:
                selected_str = match.group(1)
        else:
            selected_str = agent_resp
    else:
        tables_match = re.search(r"(?i)TABLES:\s*(.*)", agent_resp)
        if tables_match:
            selected_str = tables_match.group(1).strip()

    selected_str = selected_str.replace("'", "").replace('"', "")
    selected = [name.strip() for name in selected_str.split(",")
                if name.strip() in all_files]
    if not selected:
        selected = candidates[:2]

    return selected, reasoning


def phase2_table_selector_agent(
    query: str,
    llm_versatile: LLM,
    pm: PromptManager,
    all_files: list[str],
    candidates: list[str],
    solr_meta: SolrMetadata,
    csv_dir: Path,
    blend_db: Path,
    activity_log_parts: list[str],
    hint: str = "",
    stream_callback: StreamCallback | None = None,
) -> tuple[list[str], str, str, int]:
    """Part 2 of Phase 2: inspect candidates and choose or reject tables."""
    try:
        architect_system_prompt = pm.render("data_architect", "system_prompt")
        agent_tools = make_agent_tools(blend_db, csv_dir=csv_dir)

        token_counter = next(
            (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
            None,
        )
        if token_counter:
            token_counter.reset_counts()

        agent_prompt = pm.render("data_architect", "user_prompt",
                                 question=query,
                                 enriched_candidates_info=_format_candidate_context(
                                     candidates, solr_meta
                                 ),
                                 table_hint=hint)

        old_stdout = sys.stdout
        capture = io.StringIO()
        stream_trace = io.StringIO()
        sys.stdout = capture

        def emit_stream(delta: str) -> None:
            if not delta:
                return
            stream_trace.write(delta)
            print(delta, end="", flush=True, file=old_stdout)
            if stream_callback is not None:
                stream_callback(delta)

        thinking_capture = ThinkingCapture()
        dispatcher = get_dispatcher()
        dispatcher.add_event_handler(thinking_capture)
        emit_stream(
            "\n**Data Architect agent started**\n"
            "- Streaming model output and tool inspections below.\n"
        )

        try:
            async def _run_agent():
                explorer = FunctionAgent(
                    name="data_explorer", 
                    tools=agent_tools, 
                    llm=llm_versatile,
                    system_prompt=architect_system_prompt,
                    # verbose=False, 
                    # streaming=True,
                    # early_stopping_method="generate",
                )

                handler = explorer.run(
                    user_msg=agent_prompt,
                    max_iterations=10,
                    # early_stopping_method="generate",
                )

                tool_call_count = 0
                tool_result_count = 0
                tool_call_signatures: dict[str, int] = {}
                async for event in handler.stream_events():
                    if isinstance(event, AgentStream):
                        emit_stream(event.delta or "")
                        stall_reason = detect_phase2_agent_stall(
                            stream_trace.getvalue()
                        )
                        if stall_reason:
                            raise Phase2AgentStall(stall_reason)
                    elif isinstance(event, ToolCall):
                        tool_call_count += 1
                        tool_signature = (
                            f"{getattr(event, 'tool_name', 'unknown_tool')}:"
                            f"{format_phase2_tool_args(event)}"
                        )
                        tool_call_signatures[tool_signature] = (
                            tool_call_signatures.get(tool_signature, 0) + 1
                        )
                        if tool_call_signatures[tool_signature] >= 2:
                            continue
                            raise Phase2AgentStall(
                                "repeated identical tool call: "
                                f"{getattr(event, 'tool_name', 'unknown_tool')}"
                            )
                        emit_stream(format_phase2_tool_call(event, tool_call_count))
                    elif isinstance(event, ToolCallResult):
                        tool_result_count += 1
                        emit_stream(format_phase2_tool_result(event, tool_result_count))

                return await handler

            res = anyio.run(_run_agent)
            agent_resp = str(getattr(res, "response", res)).strip()
        except Phase2AgentStall as stall_err:
            emit_stream(
                "\n\n**Phase 2 loop guard triggered**\n"
                f"- Reason: `{str(stall_err)}`\n"
                "- Action: using the top Solr candidates as a fallback.\n"
            )
            agent_resp = (
                f'FINAL_PAYLOAD: {{"tables": "{", ".join(candidates[:2])}",'
                ' "reasoning": "Stopped a repeated Data Architect thought loop. '
                'Fallback to top Solr candidates."}}'
            )
        except Exception as agent_err:
            emit_stream(f"\n[phase2 agent error] {str(agent_err)[:160]}\n")
            agent_resp = (
                f'FINAL_PAYLOAD: {{"tables": "{", ".join(candidates[:2])}",'
                f' "reasoning": "Agent error: {str(agent_err)[:80]}. Fallback to top 2."}}'
            )
        finally:
            sys.stdout = old_stdout
            stdout_trace = capture.getvalue()
            agent_stream_trace = stream_trace.getvalue()
            full_trace = stdout_trace
            phase2_activity_trace = "".join(activity_log_parts) + agent_stream_trace
            if phase2_activity_trace:
                full_trace += (
                    "\n\n--- Phase 2 Activity Log ---\n"
                    f"{phase2_activity_trace}"
                )
            capture.close()
            stream_trace.close()
            dispatcher.event_handlers.remove(thinking_capture)

        tokens_p2 = 0
        if token_counter:
            tokens_p2 = (token_counter.prompt_llm_token_count +
                         token_counter.completion_llm_token_count)
            token_counter.reset_counts()

        Settings.callback_manager = CallbackManager([])
        selected, reasoning = _parse_table_selector_response(
            agent_resp,
            all_files,
            candidates,
        )
        return selected, reasoning, full_trace, tokens_p2

    finally:
        if blend_db.exists():
            try:
                os.remove(blend_db)
            except Exception:
                pass


def phase2_select_tables(
    query: str,
    keywords: list[str],
    llm_versatile: LLM,
    solr_client: LocalSolrClient,
    pm: PromptManager,
    all_files: list[str],
    csv_dir: Path,
    db_path: Path,
    hint: str = "",
    skip_solr: bool = False,
    top_10: list[str] | None = None,
    solr_meta: SolrMetadata | None = None,
    stream_callback: StreamCallback | None = None,
) -> Phase2SelectionResult:
    candidates, metadata, activity_log_parts, blend_db = phase2_retrieve_candidates(
        keywords=keywords,
        solr_client=solr_client,
        all_files=all_files,
        csv_dir=csv_dir,
        db_path=db_path,
        skip_solr=skip_solr,
        top_10=top_10,
        solr_meta=solr_meta,
        stream_callback=stream_callback,
    )
    selected, reasoning, full_trace, tokens_p2 = phase2_table_selector_agent(
        query=query,
        llm_versatile=llm_versatile,
        pm=pm,
        all_files=all_files,
        candidates=candidates,
        solr_meta=metadata,
        csv_dir=csv_dir,
        blend_db=blend_db,
        activity_log_parts=activity_log_parts,
        hint=hint,
        stream_callback=stream_callback,
    )
    return selected, candidates, metadata, reasoning, full_trace, tokens_p2


def phase3_generate_code(query, tables, candidates, solr_meta, reasoning,
                         llm_versatile: LLM, pm: PromptManager, csv_dir, retries=0, error_msg=""):
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

    print(res.message.content)
    print("-" * 100)
    print(res.raw)
    print("-" * 100)
    tokens = 0
    if res.raw is not None:
        tokens = (
            res.raw.get("prompt_eval_count", 0) +
            res.raw.get("eval_count", 0)
        )

    return str(res.message.content), tokens


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
            stdout_lower = result.stdout.lower()
            if "error:" in stdout_lower or "exception:" in stdout_lower:
                return None, result.stdout.strip(), code
            return result.stdout.strip(), None, code
        return None, (result.stderr.strip() or result.stdout.strip()), code
    except Exception as e:
        return None, str(e), code


def phase5_synthesize(query, raw_result, llm_instant, pm):
    prompt = pm.render("synthesizer", "prompt",
                       question=query, raw_result=raw_result)
    res = llm_instant.chat([ChatMessage(role="user", content=prompt)])
    tokens = 0
    if res.raw:
        tokens = (res.raw.get("prompt_eval_count", 0) +
                  res.raw.get("eval_count", 0))
    return str(res.message.content).strip(), tokens
