"""Phase 4: the final answer, written from the coder's raw result."""

from llama_index.core.llms import LLM, ChatMessage

from lakegen.llm.prompts import PromptManager
from lakegen.llm.token_usage import (
    extract_total_tokens,
    get_llm_token_usage,
    reset_llm_token_usage,
)


def phase4_synthesize(query: str, raw_result: str, llm: LLM, pm: PromptManager) -> tuple[str, int]:
    """Write the answer to the question from the coder's raw result.

    The token count is the largest of what the response reports and what the
    LLM's usage counter recorded for this one call.

    Args:
        query: The user's question.
        raw_result: What the analysis code returned.
        llm: The model that writes the answer.
        pm: The prompt manager holding the ``synthesizer`` prompt.

    Returns:
        The answer text, stripped, and the tokens the call used.
    """
    prompt = pm.render("synthesizer", "prompt", question=query, raw_result=raw_result)
    reset_llm_token_usage(llm)
    response = llm.chat([ChatMessage(role="user", content=prompt)])
    tokens = max(
        extract_total_tokens(response.raw),
        extract_total_tokens(response.message.additional_kwargs),
        get_llm_token_usage(llm),
    )
    return str(response.message.content).strip(), tokens
