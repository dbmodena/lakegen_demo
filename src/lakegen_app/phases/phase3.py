import os

import pandas as pd
from llama_index.core.llms import ChatMessage, LLM

from lakegen_app.types import SolrMetadata
from prompts.prompt_manager import PromptManager


def phase3_generate_code(query, tables, candidates, solr_meta: SolrMetadata, reasoning,
                         llm: LLM, pm: PromptManager, csv_dir, retries=0, error_msg=""):
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

    res = llm.chat([
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
