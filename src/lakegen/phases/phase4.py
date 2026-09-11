from llama_index.core.llms import ChatMessage, LLM
from prompts.prompt_manager import PromptManager
from lakegen.core.token_usage import (
    extract_total_tokens,
    get_llm_token_usage,
    reset_llm_token_usage,
)


def phase4_synthesize(query: str, raw_result: str, llm: LLM, pm: PromptManager) -> tuple[str, int]:
    prompt = pm.render("synthesizer", "prompt",
                       question=query, raw_result=raw_result)
    reset_llm_token_usage(llm)
    res = llm.chat([ChatMessage(role="user", content=prompt)])
    tokens = max(
        extract_total_tokens(res.raw),
        extract_total_tokens(res.message.additional_kwargs),
        get_llm_token_usage(llm),
    )
    return str(res.message.content).strip(), tokens
