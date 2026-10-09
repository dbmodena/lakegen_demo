"""Send small, tool-free gpt-oss-120b chat calls to Groq's free tier when it has room.

LakeGen's model calls go to OCI. With ``LAKEGEN_GROQ_OFFLOAD=1`` a plain chat
call of ``openai.gpt-oss-120b`` (no tools, no tool-call history: the semantic
code judge, the answer phase, the split-search decomposition) first tries
Groq's ``openai/gpt-oss-120b``, the same open weights, with one of the keys in
``GROQ_API_KEY*`` / ``GROP_API_KEY*``. The free tier allows 8,000 tokens per
minute per key, counting the prompt and the requested output, so a larger
request is refused outright, plus 1,000 requests and 200,000 tokens per day.
Each key is kept under those figures and Groq's retry-after is honoured.
Whenever Groq cannot take a call, ``chat`` returns None and the caller goes to
OCI as before, so the offload never fails a question.

Every attempt is one line of ``LAKEGEN_GROQ_LEDGER`` (default
``logs/groq_calls.jsonl``): which key, tokens, outcome and the worker thread.
The ledger also restores each key's daily usage after a restart.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

from lakegen.config.paths import LOG_DIR
from lakegen.config.registry import groq_models
from lakegen.config.settings import get_settings, groq_api_keys
from lakegen.integrations.llm.token_usage import estimate_tokens

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = groq_models()
TOKENS_PER_MINUTE = 8_000
TOKENS_PER_DAY = 200_000
REQUESTS_PER_DAY = 1_000
# Keep clear of the daily limits: Groq's own count runs slightly ahead of ours.
DAILY_HEADROOM = 0.95
# Output room a gpt-oss call needs for its reasoning plus the answer; a call
# whose prompt leaves less than this under the per-minute cap stays on OCI.
MIN_OUTPUT_TOKENS = 2_500
MAX_OUTPUT_TOKENS = 6_000
TIMEOUT_SECONDS = 120.0
ERROR_COOLDOWN_SECONDS = 60.0
_PLAIN_ROLES = {MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT}
# Request options that do not change a plain chat call (see the OCI client).
_IGNORED_KWARGS = {"think"}


def _post(url: str, headers: dict[str, str], payload: dict[str, Any]) -> requests.Response:
    return requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_SECONDS)


@dataclass
class _Key:
    number: int
    secret: str
    minute: deque = field(default_factory=deque)  # (time, tokens) of the last 60 s
    day: deque = field(default_factory=deque)  # (time, tokens) of the last 24 h
    requests: deque = field(default_factory=deque)  # request times of the last 24 h
    cooldown_until: float = 0.0

    def _trim(self, now: float) -> None:
        while self.minute and self.minute[0][0] <= now - 60:
            self.minute.popleft()
        while self.day and self.day[0][0] <= now - 86_400:
            self.day.popleft()
        while self.requests and self.requests[0] <= now - 86_400:
            self.requests.popleft()

    def fits(self, tokens: int, now: float) -> bool:
        self._trim(now)
        return (
            now >= self.cooldown_until
            and sum(t for _, t in self.minute) + tokens <= TOKENS_PER_MINUTE
            and sum(t for _, t in self.day) + tokens <= TOKENS_PER_DAY * DAILY_HEADROOM
            and len(self.requests) < REQUESTS_PER_DAY * DAILY_HEADROOM
        )


class GroqOffload:
    def __init__(self, secrets: Sequence[str], ledger: Path) -> None:
        self.keys = [_Key(number, secret) for number, secret in enumerate(secrets, 1)]
        self.ledger = ledger
        self._lock = threading.Lock()
        self._next = 0
        self._restore()

    def _restore(self) -> None:
        """Daily usage per key from the ledger, so a restart keeps counting."""
        if not self.ledger.exists():
            return
        now = time.time()
        by_number = {key.number: key for key in self.keys}
        for line in self.ledger.read_text().splitlines():
            try:
                entry = json.loads(line)
                at = datetime.fromisoformat(entry["at"]).timestamp()
            except (ValueError, KeyError, TypeError):
                continue
            key = by_number.get(entry.get("key"))
            if key is None or at <= now - 86_400 or not entry.get("sent"):
                continue
            key.requests.append(at)
            key.day.append((at, int(entry.get("total_tokens") or 0)))

    def _log(self, **entry: Any) -> None:
        entry = {"at": datetime.now(timezone.utc).isoformat(), **entry,
                 "thread": threading.current_thread().name}
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.ledger.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _reserve(self, tokens: int) -> _Key | None:
        now = time.time()
        with self._lock:
            for offset in range(len(self.keys)):
                key = self.keys[(self._next + offset) % len(self.keys)]
                if key.fits(tokens, now):
                    self._next = (self._next + offset + 1) % len(self.keys)
                    key.minute.append((now, tokens))
                    key.requests.append(now)
                    return key
        return None

    def chat(self, model: str, messages: Sequence[Any], kwargs: dict[str, Any],
             temperature: float) -> ChatResponse | None:
        """Groq's answer to a plain chat call, or None to send it to OCI."""
        if set(kwargs) - _IGNORED_KWARGS:
            return None  # tools, tool_choice or other options: an agent turn
        plain = []
        for message in messages:
            if (not isinstance(message, ChatMessage) or message.role not in _PLAIN_ROLES
                    or message.additional_kwargs.get("tool_calls")
                    or not isinstance(message.content, str)):
                return None
            plain.append({"role": message.role.value, "content": message.content})
        # A margin for the tokenizer difference and the chat template.
        prompt = int(estimate_tokens(*(m["content"] for m in plain)) * 1.15) + 40 * len(plain) + 100
        output = min(MAX_OUTPUT_TOKENS, TOKENS_PER_MINUTE - prompt - 200)
        if output < MIN_OUTPUT_TOKENS:
            self._log(outcome="skip_large", requested=prompt, sent=False)
            return None
        key = self._reserve(prompt + output)
        if key is None:
            self._log(outcome="skip_busy", requested=prompt + output, sent=False)
            return None
        payload = {"model": GROQ_MODELS[model], "messages": plain,
                   "temperature": temperature, "max_completion_tokens": output}
        headers = {"Authorization": f"Bearer {key.secret}", "Content-Type": "application/json",
                   "User-Agent": "lakegen-groq-offload"}
        started = time.time()
        try:
            response = _post(GROQ_URL, headers, payload)
        except requests.RequestException as exc:
            key.cooldown_until = time.time() + ERROR_COOLDOWN_SECONDS
            self._log(outcome="error", key=key.number, error=str(exc)[:200], sent=True,
                      seconds=round(time.time() - started, 2))
            return None
        seconds = round(time.time() - started, 2)
        if response.status_code != 200:
            wait = ERROR_COOLDOWN_SECONDS
            if response.status_code == 429:
                try:
                    wait = float(response.headers.get("retry-after") or wait)
                except ValueError:
                    pass
            if response.status_code != 413:  # too large is about the call, not the key
                key.cooldown_until = time.time() + wait
            self._log(outcome=f"http_{response.status_code}", key=key.number, sent=True,
                      error=response.text[:200], seconds=seconds)
            return None
        data = response.json()
        usage = data.get("usage") or {}
        total = int(usage.get("total_tokens") or 0)
        with self._lock:
            key.day.append((time.time(), total))
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason")
        outcome = "ok" if content.strip() and finish != "length" else f"unusable_{finish or 'empty'}"
        self._log(outcome=outcome, key=key.number, sent=True, seconds=seconds,
                  prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                  total_tokens=total)
        if outcome != "ok":
            return None
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=content), raw=data)


_offload: GroqOffload | None = None
_offload_lock = threading.Lock()


def _keys_from_env() -> list[str]:
    return list(groq_api_keys())


def enabled_for(model: str) -> bool:
    return get_settings().groq_offload and model in GROQ_MODELS and bool(_keys_from_env())


def chat(model: str, messages: Sequence[Any], kwargs: dict[str, Any],
         temperature: float = 0.1) -> ChatResponse | None:
    """Groq's answer when the offload is on and can take the call, else None."""
    global _offload
    if not enabled_for(model):
        return None
    with _offload_lock:
        if _offload is None:
            ledger = Path(get_settings().groq_ledger or LOG_DIR / "groq_calls.jsonl")
            _offload = GroqOffload(_keys_from_env(), ledger)
    return _offload.chat(model, messages, kwargs, temperature)
