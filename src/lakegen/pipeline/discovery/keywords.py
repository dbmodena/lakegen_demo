"""Phase 1 of the divided architecture: the keyword writer and the strict-AND check of its keywords.

The writer streams a retrieval intent whose search terms become the keywords
of Phase 2's retrieval; ``KeywordSearch`` counts each proposal before
retrieval runs and refuses or bans the word sets that match no table.
"""

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from llama_index.core import Settings
from llama_index.core.llms import LLM, ChatMessage

from lakegen.config.retrieval import FusionMethod
from lakegen.llm.prompts import PromptManager
from lakegen.llm.token_usage import (
    extract_total_tokens,
    get_llm_token_usage,
    reset_llm_token_usage,
)
from lakegen.pipeline.discovery.terms import banned_subset, keyword_terms, shrink_and_ban
from lakegen.retrieval import RetrievalConfig, RetrievalMode
from lakegen.retrieval.intent import parse_retrieval_intent
from lakegen.retrieval.modes.keyword import KeywordRetriever
from lakegen.retrieval.solr.client import SolrClient

logger = logging.getLogger(__name__)


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

    open_match = re.search(
        r"<(?:think|reasoning)>(.*)$", visible_text, flags=re.IGNORECASE | re.DOTALL
    )
    if open_match:
        thinking_parts.append(open_match.group(1))
        visible_text = visible_text[: open_match.start()]

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

    def __init__(
        self, stream_placeholder, reasoning_placeholder, cancel_check: Callable[[], None] | None
    ) -> None:
        self.stream_placeholder = stream_placeholder
        self.reasoning_placeholder = reasoning_placeholder
        self.cancel_check = cancel_check
        self.raw = ""
        self.structured_reasoning = ""
        self.tokens = 0
        self.loop_detected = False

    def _show(self) -> None:
        visible, tagged = split_thinking_blocks(self.raw)
        reasoning = "\n\n".join(
            part.strip() for part in (self.structured_reasoning, tagged) if part.strip()
        )
        if self.stream_placeholder is not None:
            self.stream_placeholder.markdown(visible or self.raw)
        if self.reasoning_placeholder is not None and reasoning:
            self.reasoning_placeholder.markdown(reasoning)

    def _looping(self) -> bool:
        """Whether the reasoning repeats its own tail; the placeholder says so when it does."""
        cleaned = self.structured_reasoning.strip()
        if (
            len(cleaned) <= REPEAT_WINDOW
            or cleaned.count(cleaned[-REPEAT_WINDOW:]) < REPEAT_THRESHOLD
        ):
            return False
        logger.warning("[phase1] Repetition loop detected in reasoning – breaking stream.")
        self.loop_detected = True
        if self.reasoning_placeholder is not None:
            self.reasoning_placeholder.markdown(
                self.structured_reasoning
                + "\n\n⚠️ **[Phase 1] Warning: A repetition loop was detected in the model reasoning. "
                "The stream was stopped to prevent it from hanging.**"
            )
        return True

    def consume(self, chunks, *, with_thinking: bool) -> None:
        """Print and show each chunk as it arrives, stopping at a reasoning loop.

        Args:
            chunks: The model's streamed chat chunks.
            with_thinking: Whether to read the chunks' thinking deltas.
        """
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
                self.tokens,
                extract_total_tokens(chunk.raw),
                extract_total_tokens(chunk.additional_kwargs),
            )

    def read(self, llm: LLM, messages: list[ChatMessage], *, stream_reasoning: bool) -> None:
        """Stream the answer, asking for the reasoning; a backend that refuses the option gets a plain request."""
        try:
            self.consume(
                llm.stream_chat(messages, **({"think": True} if stream_reasoning else {})),
                with_thinking=True,
            )
        except Exception:
            if self.raw or self.structured_reasoning or not stream_reasoning:
                raise
            logger.debug(
                "[phase1] the backend refused the think option; asking without it", exc_info=True
            )
            self.consume(llm.stream_chat(messages), with_thinking=False)

    def reasoning(self) -> str:
        """The thinking deltas, the tagged blocks and any ``<reasoning>`` the text holds, together."""
        _visible, tagged = split_thinking_blocks(self.raw)
        for block in re.findall(
            r"<reasoning>(.*?)</reasoning>", self.raw, re.IGNORECASE | re.DOTALL
        ):
            if block.strip():
                tagged += "\n" + block.strip()
        return "\n\n".join(
            part.strip() for part in (self.structured_reasoning, tagged) if part.strip()
        )

    def visible(self) -> str:
        """The answer without its reasoning blocks."""
        return split_thinking_blocks(self.raw)[0].strip()


def _token_count(
    stream: _IntentStream, token_counter, llm: LLM, system_prompt: str, user_prompt: str
) -> int:
    """The stream's usage, else the callback counter's, else the model's own, else an estimate."""
    tokens = stream.tokens
    if token_counter and tokens == 0:
        tokens = token_counter.prompt_llm_token_count + token_counter.completion_llm_token_count
    tokens = max(tokens, get_llm_token_usage(llm))
    if tokens == 0:
        # Fallback estimation if stream skipped token tracking completely
        tokens = int(
            (len(system_prompt.split()) + len(user_prompt.split()) + len(stream.raw.split())) * 1.3
        )
    return tokens


def _intent_terms(raw_content: str, value_search: bool) -> list[str]:
    """The search terms of a resolved intent; none for an unresolved or unparsable one."""
    try:
        intent = parse_retrieval_intent(raw_content)
    except ValueError:
        return []
    return intent.search_terms(value_search) if intent.status == "resolved" else []


def write_keywords(
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
    zero_result_sets: Sequence[Sequence[str]] = (),
    follow_ups: Sequence[tuple[str, str]] = (),
) -> tuple[list[str], str, int, str]:
    """The keyword writer's turn: its search terms, its raw intent, its tokens and its reasoning.

    ``zero_result_sets`` are the word sets earlier rounds proved to match no
    table. ``follow_ups`` continue the round's conversation: each earlier
    answer of the writer, and what the search said about it.

    Args:
        query: The user's question.
        llm: The writer's LLM.
        pm: Prompt manager of the run.
        hint: The user's hint for this round.
        portal_name: Name of the open-data portal, shown as the catalogue.
        stream_placeholder: UI placeholder for the streamed answer, if any.
        reasoning_placeholder: UI placeholder for the streamed reasoning, if any.
        stream_reasoning: Whether to ask the backend for thinking deltas.
        cancel_check: Raises when the run is cancelled.
        avoid_keywords: Keywords the writer must not propose again.
        value_search: Whether retrieval searches cell values, so terms the rows store.
        verbatim_entities: Whether entities must be named as written.
        zero_result_sets: Word sets earlier rounds proved to match no table.
        follow_ups: Earlier answers of the writer, each with the search's feedback.

    Returns:
        The search terms (empty for an unresolved or unparsable intent), the
        raw intent, the tokens used and the reasoning.
    """
    system_prompt = pm.render(
        "retrieval_intent",
        "system_prompt",
        # Cell-value retrieval needs terms the rows store, not dataset topics.
        value_search=value_search,
        # Content search scans for entities, so they must be named as written.
        verbatim_entities=verbatim_entities,
    )
    user_prompt = pm.render(
        "retrieval_intent",
        "user_prompt",
        question=query,
        catalog=portal_name,
        schema="not supplied",
        avoid_keywords_str=", ".join(avoid_keywords) if avoid_keywords else "",
        hint=hint,
        zero_result_sets_str="; ".join(_word_set(words) for words in zero_result_sets),
    )
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]
    for answer, feedback in follow_ups:
        messages.append(ChatMessage(role="assistant", content=answer))
        messages.append(ChatMessage(role="user", content=feedback))
    token_counter = next(
        (h for h in Settings.callback_manager.handlers if hasattr(h, "reset_counts")),
        None,
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


# --- the strict-AND search of the writer's keywords ----------------------------

MAX_REFUSALS = 3
"""Proposals refused for a banned word set, per round, before the round goes on without a match."""

EMPTY_FEEDBACK = (
    "Your keywords matched no table: no catalogue entry contains every word of {words}. "
    "Retrieval is a strict AND search, so a table matches only when its title, description, "
    "publisher or tags contain all of your words. {learned}"
    "A search containing any of these word sets is refused from now on: {bans}. "
    "Write a new RETRIEVAL_INTENT whose concepts drop or replace the failing words and keep "
    "the words that match tables. Searches left: {left}."
)
REFUSED_FEEDBACK = (
    "Refused before searching: {words} contains {blocked}, a word set no table matches. "
    "Write a new RETRIEVAL_INTENT whose concepts leave out that combination."
)


def _word_set(words) -> str:
    return "{" + ", ".join(sorted(words)) + "}"


def strict_and_count(
    solr: SolrClient, config: RetrievalConfig
) -> Callable[[Sequence[str]], int] | None:
    """How many tables a strict-AND search of the keywords matches, where the mode searches them so; else None.

    Keyword mode searches the keywords alone, hybrid mode on its lexical
    side, both by the same Solr query this count runs. An OR fallback, or a
    weighted hybrid that gives the lexical side no weight, never comes back
    empty for want of a word, so there is nothing to count.

    Args:
        solr: Client of the lake's Solr core.
        config: Retrieval settings of the run.

    Returns:
        A function counting the tables that match every keyword, or None.
    """
    if (
        config.mode not in (RetrievalMode.KEYWORD, RetrievalMode.HYBRID)
        or config.keyword_or_fallback
    ):
        return None
    if (
        config.mode is RetrievalMode.HYBRID
        and config.fusion_method is FusionMethod.WEIGHTED
        and config.alpha == 0.0
    ):
        return None
    return KeywordRetriever(solr, query_fields=config.lexical_query_fields).count


@dataclass
class KeywordSearch:
    """The strict-AND searches of one keyword round: each proposal refused, counted or banned.

    Every proposal of the writer is checked here before retrieval runs. One
    that contains a word set known to match no table is refused unsearched;
    one whose count is zero is banned as its smallest failing word sets, and
    the writer is told what was learned. ``searches`` proposals may be
    counted and ``MAX_REFUSALS`` refused; past either the round goes on with
    the last proposal (``exhausted``), whose keywords match nothing: hybrid
    retrieval answers from its semantic side, keyword retrieval finds no table.

    Attributes:
        count: Counts the tables a strict-AND search of the keywords matches.
        bans: The run's zero-result word sets; the new ones are added in place.
        searches: How many proposals may be counted.
        log: One entry per proposal: its keywords and whether it matched, was
            empty, was refused or could not be checked.
        exhausted: Whether the round goes on with keywords that match nothing.
    """

    count: Callable[[Sequence[str]], int]
    bans: list[frozenset[str]]
    searches: int
    log: list[dict[str, Any]] = field(default_factory=list)
    exhausted: bool = False

    def _counted(self) -> int:
        return sum(entry["outcome"] in ("matched", "empty") for entry in self.log)

    def review(self, keywords: Sequence[str]) -> str | None:
        """What the writer is told before ``keywords`` may reach retrieval; None when they may.

        A count that fails (Solr unreachable) lets the keywords through:
        retrieval runs as it did before the check, and reports its own error.

        Args:
            keywords: The writer's proposal.

        Returns:
            The feedback for the writer, or None when the keywords may be searched.
        """
        if not keywords:
            return None
        words = _word_set(keyword_terms(keywords))
        blocked = banned_subset(self.bans, keywords)
        if blocked is not None:
            self.log.append(
                {"keywords": list(keywords), "outcome": "refused", "banned": sorted(blocked)}
            )
            if sum(entry["outcome"] == "refused" for entry in self.log) > MAX_REFUSALS:
                self.exhausted = True
                return None
            return REFUSED_FEEDBACK.format(words=words, blocked=_word_set(blocked))
        try:
            matched = self.count(keywords)
        except Exception as exc:
            logger.warning(
                "[phase1] keyword count failed, searching as written: %s: %s",
                type(exc).__name__,
                exc,
            )
            self.log.append({"keywords": list(keywords), "outcome": "unchecked"})
            return None
        if matched:
            self.log.append({"keywords": list(keywords), "outcome": "matched", "tables": matched})
            return None
        learned = shrink_and_ban(self.bans, keywords, self.count)
        self.log.append({"keywords": list(keywords), "outcome": "empty", "learned": learned or ""})
        left = self.searches - self._counted()
        if left <= 0:
            self.exhausted = True
            return None
        return EMPTY_FEEDBACK.format(
            words=words,
            learned=f"{learned} " if learned else "",
            bans="; ".join(_word_set(ban) for ban in self.bans),
            left=left,
        )
