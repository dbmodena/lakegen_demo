from llama_index.core.llms import ChatMessage


def phase5_synthesize(query, raw_result, llm, pm):
    prompt = pm.render("synthesizer", "prompt",
                       question=query, raw_result=raw_result)
    res = llm.chat([ChatMessage(role="user", content=prompt)])
    tokens = 0
    if res.raw:
        tokens = (res.raw.get("prompt_eval_count", 0) +
                  res.raw.get("eval_count", 0))
    return str(res.message.content).strip(), tokens
