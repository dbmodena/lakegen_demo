import os

import pandas as pd
from llama_index.core.llms import ChatMessage, LLM

from lakegen.types import SolrMetadata
from prompts.prompt_manager import PromptManager

from .phase1 import split_thinking_blocks


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
    force_execution: bool = False,
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
):
    info_lines = [f"AVAILABLE SELECTED TABLES IN '{csv_dir}/':"]
    for idx, fn in enumerate(tables, 1):
        filepath = os.path.join(csv_dir, fn.strip())
        meta = solr_meta.get(fn, {})
        cn = meta.get("columns.name", [])
        ct = meta.get("columns.type", [])
        
        df = pd.read_csv(filepath, nrows=5)
        sample_rows = df.head(3).to_string(index=False)

        if cn and len(cn) == len(ct):
            cols = [f"'{n}' ({t})" for n, t in zip(cn, ct)]
        elif cn:
            cols = [f"'{n}'" for n in cn]
        else:
            try:
                cols = [f"'{n}' ({t})" for n, t in
                        zip(df.columns, [str(d) for d in df.dtypes])]
                
            except Exception:
                cols = ["Unknown columns"]
        info_lines.append(f"{idx}. '{filepath}'")
        info_lines.append(f"   Columns: [" + ", ".join(cols) + "]")
        info_lines.append(f"   Sample rows:\n{sample_rows}")

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
    try:
        chunk_stream = llm.stream_chat(messages, **stream_kwargs)
        for chunk in chunk_stream:
            thinking_delta = chunk.additional_kwargs.get("thinking_delta")
            if thinking_delta:
                structured_reasoning += thinking_delta
                print(thinking_delta, end="", flush=True)
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
    except Exception:
        if raw_stream or structured_reasoning or not stream_reasoning:
            raise
        chunk_stream = llm.stream_chat(messages)
        for chunk in chunk_stream:
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

    return visible_content.strip(), tokens
