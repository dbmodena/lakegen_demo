import re
from collections.abc import Callable

from llama_index.core import Settings
from llama_index.core.llms import LLM, ChatMessage

from lakegen.integrations.llm.token_usage import (
    extract_total_tokens,
    get_llm_token_usage,
    reset_llm_token_usage,
)
from lakegen.prompts.prompt_manager import PromptManager
from lakegen.retrieval.intent import parse_retrieval_intent


def split_thinking_blocks(text: str) -> tuple[str, str]:
    """Separate content emitted in <think> blocks from the visible answer."""
    thinking_parts: list[str] = []

    def collect_closed(match: re.Match[str]) -> str:
        thinking_parts.append(match.group(1))
        return ""

    visible_text = re.sub(
        r"<(?:think|reasoning)>(.*?)</(?:think|reasoning)>",
        collect_closed,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    open_match = re.search(r"<(?:think|reasoning)>(.*)$", visible_text, flags=re.IGNORECASE | re.DOTALL)
    if open_match:
        thinking_parts.append(open_match.group(1))
        visible_text = visible_text[:open_match.start()]

    visible_text = re.sub(r"</(?:think|reasoning)>", "", visible_text, flags=re.IGNORECASE)
    thinking_text = "\n".join(part.strip() for part in thinking_parts if part.strip())
    return visible_text.strip(), thinking_text.strip()


REPEAT_WINDOW = 200
"""The tail of the reasoning the loop guard looks for, in characters."""
REPEAT_THRESHOLD = 5
"""How many times the tail must occur in the reasoning for it to count as a loop."""
LOOP_WARNING = "\n\n⚠️ **[Phase 1] The model looped; retrieval intent is unresolved.**"


class _IntentStream:
    """The model's streamed answer: its visible text, its reasoning, its tokens, and the loop guard.

    The reasoning arrives as thinking deltas (when the backend supports the
    ``think`` option) and as tagged blocks in the text; both reach the
    placeholders as they stream. A reasoning that repeats its own tail is a
    loop: the stream stops, and the intent is left unresolved.
    """

    def __init__(self, stream_placeholder, reasoning_placeholder, cancel_check: Callable[[], None] | None) -> None:
        self.stream_placeholder = stream_placeholder
        self.reasoning_placeholder = reasoning_placeholder
        self.cancel_check = cancel_check
        self.raw = ""
        self.structured_reasoning = ""
        self.tokens = 0
        self.loop_detected = False

    def _show(self) -> None:
        visible, tagged = split_thinking_blocks(self.raw)
        reasoning = "\n\n".join(part.strip() for part in (self.structured_reasoning, tagged) if part.strip())
        if self.stream_placeholder is not None:
            self.stream_placeholder.markdown(visible or self.raw)
        if self.reasoning_placeholder is not None and reasoning:
            self.reasoning_placeholder.markdown(reasoning)

    def _looping(self) -> bool:
        """Whether the reasoning repeats its own tail; the placeholder says so when it does."""
        cleaned = self.structured_reasoning.strip()
        if len(cleaned) <= REPEAT_WINDOW or cleaned.count(cleaned[-REPEAT_WINDOW:]) < REPEAT_THRESHOLD:
            return False
        print("\n[phase1] Repetition loop detected in reasoning – breaking stream.")
        self.loop_detected = True
        if self.reasoning_placeholder is not None:
            self.reasoning_placeholder.markdown(
                self.structured_reasoning
                + "\n\n⚠️ **[Phase 1] Warning: A repetition loop was detected in the model reasoning. "
                "The stream was stopped to prevent it from hanging.**"
            )
        return True

    def consume(self, chunks, *, with_thinking: bool) -> None:
        for chunk in chunks:
            if self.cancel_check is not None:
                self.cancel_check()
            thinking = chunk.additional_kwargs.get("thinking_delta") if with_thinking else None
            if thinking:
                self.structured_reasoning += thinking
                print(thinking, end="", flush=True)
                if self._looping():
                    break
                self._show()
            delta = chunk.delta or ""
            if delta:
                self.raw += delta
                print(delta, end="", flush=True)
                self._show()
            self.tokens = max(
                self.tokens, extract_total_tokens(chunk.raw), extract_total_tokens(chunk.additional_kwargs),
            )

    def read(self, llm: LLM, messages: list[ChatMessage], *, stream_reasoning: bool) -> None:
        """Stream the answer, asking for the reasoning; a backend that refuses the option gets a plain request."""
        try:
            self.consume(llm.stream_chat(messages, **({"think": True} if stream_reasoning else {})), with_thinking=True)
        except Exception:
            if self.raw or self.structured_reasoning or not stream_reasoning:
                raise
            self.consume(llm.stream_chat(messages), with_thinking=False)

    def reasoning(self) -> str:
        """The thinking deltas, the tagged blocks and any ``<reasoning>`` the text holds, together."""
        _visible, tagged = split_thinking_blocks(self.raw)
        for block in re.findall(r"<reasoning>(.*?)</reasoning>", self.raw, re.IGNORECASE | re.DOTALL):
            if block.strip():
                tagged += "\n" + block.strip()
        return "\n\n".join(part.strip() for part in (self.structured_reasoning, tagged) if part.strip())

    def visible(self) -> str:
        return split_thinking_blocks(self.raw)[0].strip()


def _token_count(stream: _IntentStream, token_counter, llm: LLM, system_prompt: str, user_prompt: str) -> int:
    """The stream's usage, else the callback counter's, else the model's own, else an estimate."""
    tokens = stream.tokens
    if token_counter and tokens == 0:
        tokens = token_counter.prompt_llm_token_count + token_counter.completion_llm_token_count
    tokens = max(tokens, get_llm_token_usage(llm))
    if tokens == 0:
        # Fallback estimation if stream skipped token tracking completely
        tokens = int((len(system_prompt.split()) + len(user_prompt.split()) + len(stream.raw.split())) * 1.3)
    return tokens


def _intent_terms(raw_content: str, value_search: bool) -> list[str]:
    """The search terms of a resolved intent; none for an unresolved or unparsable one."""
    try:
        intent = parse_retrieval_intent(raw_content)
    except ValueError:
        return []
    return intent.search_terms(value_search) if intent.status == "resolved" else []


def phase1_generate_keywords(
    query: str,
    llm: LLM,
    pm: PromptManager,
    hint: str = "",
    portal_name: str = "",
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
    cancel_check: Callable[[], None] | None = None,
    avoid_keywords: list[str] | None = None,
    value_search: bool = False,
    verbatim_entities: bool = False,
) -> tuple[list[str], str, int, str]:
    """The keyword writer's turn: its search terms, its raw intent, its tokens and its reasoning."""
    system_prompt = pm.render(
        "retrieval_intent", "system_prompt",
        # Cell-value retrieval needs terms the rows store, not dataset topics.
        value_search=value_search,
        # Content search scans for entities, so they must be named as written.
        verbatim_entities=verbatim_entities,
    )
    user_prompt = pm.render(
        "retrieval_intent", "user_prompt", question=query, catalog=portal_name, schema="not supplied",
        avoid_keywords_str=", ".join(avoid_keywords) if avoid_keywords else "", hint=hint,
    )
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    token_counter = next(
        (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")), None,
    )
    if token_counter:
        token_counter.reset_counts()
    reset_llm_token_usage(llm)
    print("[phase1 keyword stream] ", end="", flush=True)
    stream = _IntentStream(stream_placeholder, reasoning_placeholder, cancel_check)
    stream.read(llm, messages, stream_reasoning=stream_reasoning)
    print("", flush=True)
    tokens = _token_count(stream, token_counter, llm, system_prompt, user_prompt)
    reasoning = stream.reasoning()
    raw_content = stream.visible()
    extracted = _intent_terms(raw_content, value_search)
    if stream.loop_detected:
        reasoning += LOOP_WARNING
    return extracted, raw_content, tokens, reasoning
