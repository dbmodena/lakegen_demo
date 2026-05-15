import re
from collections.abc import Callable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from llama_index.core.llms import ChatMessage, LLM

from prompts.prompt_manager import PromptManager


def extract_wordnet_query_keywords(query: str) -> str:
    lemmatizer = WordNetLemmatizer()
    words = re.findall(r'\b\w+\b', query.lower())
    
    try:
        ita_stops = set(stopwords.words('italian'))
        spa_stops = set(stopwords.words('spanish'))
        fra_stops = set(stopwords.words('french'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        ita_stops = set(stopwords.words('italian'))
        spa_stops = set(stopwords.words('spanish'))
        fra_stops = set(stopwords.words('french'))
        
    combined_stops = ita_stops.union(spa_stops).union(fra_stops).union(ENGLISH_STOP_WORDS)
    
    extracted_keywords = [lemmatizer.lemmatize(w) for w in words if w not in combined_stops]
    return ", ".join(list(dict.fromkeys(extracted_keywords)))


def split_thinking_blocks(text: str) -> tuple[str, str]:
    """Separate content emitted in <think> blocks from the visible answer."""
    thinking_parts: list[str] = []

    def collect_closed(match: re.Match[str]) -> str:
        thinking_parts.append(match.group(1))
        return ""

    visible_text = re.sub(
        r"<think>(.*?)</think>",
        collect_closed,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    open_match = re.search(r"<think>(.*)$", visible_text, flags=re.IGNORECASE | re.DOTALL)
    if open_match:
        thinking_parts.append(open_match.group(1))
        visible_text = visible_text[:open_match.start()]

    visible_text = re.sub(r"</think>", "", visible_text, flags=re.IGNORECASE)
    thinking_text = "\n".join(part.strip() for part in thinking_parts if part.strip())
    return visible_text.strip(), thinking_text.strip()


def phase1_generate_keywords(
    query: str,
    llm: LLM,
    pm: PromptManager,
    hint="",
    portal_name: str = "",
    stream_placeholder=None,
    reasoning_placeholder=None,
    stream_reasoning: bool = True,
    cancel_check: Callable[[], None] | None = None,
):
    wordnet_keywords_str = extract_wordnet_query_keywords(query)
    wordnet_keywords = [k.strip() for k in wordnet_keywords_str.split(",") if k.strip()]

    system_prompt = pm.render(
        "keyword_generator",
        "system_prompt"
    )

    user_prompt = pm.render(
        "keyword_generator",
        "user_prompt",
        question=query,
        portal_name=portal_name,
        raw_keywords_str=wordnet_keywords_str,
        keyword_hint=hint
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

    print("[phase1 keyword stream] ", end="", flush=True)

    # Phase 1 is a simple keyword selection task — reasoning should never
    # exceed a few hundred tokens.  These guards break the stream early
    # when the model enters an infinite thinking loop.
    _MAX_REASONING_LEN = 8000   # chars – generous ceiling
    _REPEAT_WINDOW = 150        # chars – tail window for repetition check
    _REPEAT_THRESHOLD = 4       # how many times the tail must repeat

    stream_kwargs = {"think": True} if stream_reasoning else {}
    loop_detected = False
    try:
        chunk_stream = llm.stream_chat(messages, **stream_kwargs)
        for chunk in chunk_stream:
            if cancel_check is not None:
                cancel_check()
            thinking_delta = chunk.additional_kwargs.get("thinking_delta")
            if thinking_delta:
                structured_reasoning += thinking_delta
                print(thinking_delta, end="", flush=True)

                # ── Loop detection ────────────────────────────────
                cleaned = structured_reasoning.strip()
                if len(cleaned) > _MAX_REASONING_LEN:
                    print("\n[phase1] Reasoning exceeded max length – breaking stream.")
                    loop_detected = True
                    break
                if len(cleaned) > _REPEAT_WINDOW:
                    tail = cleaned[-_REPEAT_WINDOW:]
                    if cleaned.count(tail) >= _REPEAT_THRESHOLD:
                        print("\n[phase1] Repetition loop detected in reasoning – breaking stream.")
                        loop_detected = True
                        break
                # ──────────────────────────────────────────────────

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
            if cancel_check is not None:
                cancel_check()
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

    visible_content, tagged_reasoning = split_thinking_blocks(raw_stream)
    reasoning_content = "\n\n".join(
        part.strip()
        for part in (structured_reasoning, tagged_reasoning)
        if part.strip()
    )
    raw_content = visible_content.strip().lower()
    model_keywords = re.findall(r"(?u)\b[\w-]+\b", raw_content)
    query_numbers = re.findall(r"\b\d+\b", query)
    extracted = list(dict.fromkeys(model_keywords + query_numbers))#[:3]

    # Fallback: if the model looped or produced no keywords, use WordNet
    if not extracted or loop_detected:
        if loop_detected:
            print(f"[phase1] Loop fallback → using WordNet keywords: {wordnet_keywords[:3]}")
        extracted = wordnet_keywords[:3]
    return extracted, raw_content, tokens, reasoning_content
