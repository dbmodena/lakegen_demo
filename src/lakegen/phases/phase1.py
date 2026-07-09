import re
from collections.abc import Callable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core import Settings

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
) -> tuple[list[str], str, int, str]:
    wordnet_keywords_str = extract_wordnet_query_keywords(query)
    wordnet_keywords = [k.strip() for k in wordnet_keywords_str.split(",") if k.strip()]

    system_prompt = pm.render(
        "keyword_generator",
        "system_prompt"
    )

    avoid_keywords_str = ", ".join(avoid_keywords) if avoid_keywords else ""

    user_prompt = pm.render(
        "keyword_generator",
        "user_prompt",
        question=query,
        portal_name=portal_name,
        raw_keywords_str=wordnet_keywords_str,
        keyword_hint=hint,
        avoid_keywords_str=avoid_keywords_str,
    )

    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    raw_stream = ""
    structured_reasoning = ""
    tokens = 0

    token_counter = next(
        (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
        None,
    )
    if token_counter:
        token_counter.reset_counts()

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

    _REPEAT_WINDOW = 200        # chars – tail window for repetition check
    _REPEAT_THRESHOLD = 5       # how many times the tail must repeat
    loop_detected = False

    stream_kwargs = {"think": True} if stream_reasoning else {}
    try:
        chunk_stream = llm.stream_chat(messages, **stream_kwargs)
        for chunk in chunk_stream:
            if cancel_check is not None:
                cancel_check()
            thinking_delta = chunk.additional_kwargs.get("thinking_delta")
            if thinking_delta:
                structured_reasoning += thinking_delta
                print(thinking_delta, end="", flush=True)

                # ── Repetition loop detection (No max length constraint) ──
                cleaned = structured_reasoning.strip()
                if len(cleaned) > _REPEAT_WINDOW:
                    tail = cleaned[-_REPEAT_WINDOW:]
                    if cleaned.count(tail) >= _REPEAT_THRESHOLD:
                        print("\n[phase1] Repetition loop detected in reasoning – breaking stream.")
                        loop_detected = True
                        if reasoning_placeholder is not None:
                            reasoning_placeholder.markdown(
                                structured_reasoning + "\n\n⚠️ **[Phase 1] Attenzione: Rilevato loop di ripetizione nel ragionamento del modello! Lo stream è stato interrotto per evitare blocchi.**"
                            )
                        break
                # ──────────────────────────────────────────────────────────

                update_placeholders()

            delta = chunk.delta or ""
            if delta:
                raw_stream += delta
                print(delta, end="", flush=True)
                update_placeholders()

            if chunk.raw and isinstance(chunk.raw, dict):
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

            if chunk.raw and isinstance(chunk.raw, dict):
                prompt_tokens = chunk.raw.get("prompt_eval_count") or 0
                completion_tokens = chunk.raw.get("eval_count") or 0
                if prompt_tokens or completion_tokens:
                    tokens = prompt_tokens + completion_tokens
    print("", flush=True)

    if token_counter and tokens == 0:
        tokens = token_counter.prompt_llm_token_count + token_counter.completion_llm_token_count
    if tokens == 0:
        # Fallback estimation if stream skipped token tracking completely
        tokens = int((len(system_prompt.split()) + len(user_prompt.split()) + len(raw_stream.split())) * 1.3)

    visible_content, tagged_reasoning = split_thinking_blocks(raw_stream)

    reasoning_blocks = re.findall(r"<reasoning>(.*?)</reasoning>", raw_stream, re.IGNORECASE | re.DOTALL)
    for block in reasoning_blocks:
        if block.strip():
            tagged_reasoning += "\n" + block.strip()

    reasoning_content = "\n\n".join(
        part.strip()
        for part in (structured_reasoning, tagged_reasoning)
        if part.strip()
    )

    keywords_match = re.search(r"<keywords>\s*(.*?)\s*</keywords>", raw_stream, re.IGNORECASE | re.DOTALL)
    if keywords_match:
        raw_content = keywords_match.group(1).strip().lower()
    else:
        raw_content = re.sub(r"<reasoning>.*?</reasoning>", "", visible_content, flags=re.IGNORECASE | re.DOTALL).strip().lower()
        if not raw_content:
            fallback_match = re.search(r"keywords?[:\s]+(.*)", raw_stream, flags=re.IGNORECASE | re.DOTALL)
            raw_content = fallback_match.group(1).strip().lower() if fallback_match else raw_stream.strip().lower()

    model_keywords = re.findall(r"(?u)\b[\w-]+\b", raw_content)
    query_numbers = re.findall(r"\b\d+\b", query)
    extracted = list(dict.fromkeys(model_keywords + query_numbers))#[:3]

    # Fallback: if the model looped or produced no keywords, use WordNet
    if not extracted or loop_detected:
        if loop_detected:
            print(f"[phase1] Loop fallback → using WordNet keywords: {wordnet_keywords[:3]}")
        extracted = wordnet_keywords[:3]

    if loop_detected:
        reasoning_content += "\n\n⚠️ **[Phase 1] Attenzione: Rilevato loop di ripetizione nel ragionamento del modello! Lo stream è stato interrotto ed è stato attivato il fallback sulle keyword WordNet.**"

    return extracted, raw_content, tokens, reasoning_content
