"""Durable lexical-search memory shared across batch executions."""

from __future__ import annotations

import json
import hashlib
import re
import threading
from pathlib import Path
from typing import Any, Iterable

from lakegen.keyword_terms import keyword_terms


_LOCK = threading.Lock()
_VERSION = 1
_MAX_EVENTS_PER_QUESTION = 20
_MAX_QUESTIONS_PER_SCOPE = 200


def _prune(combinations: Iterable[Iterable[str]]) -> list[frozenset[str]]:
    # Re-normalising on load also upgrades bans stored under an older, coarser
    # term rule, so they keep matching what Solr actually ANDs.
    unique = sorted(
        {keyword_terms(item) for item in combinations} - {frozenset()},
        key=lambda item: (len(item), sorted(item)),
    )
    kept: list[frozenset[str]] = []
    for candidate in unique:
        if not any(known <= candidate for known in kept):
            kept.append(candidate)
    return kept


def load_keyword_memory(path: Path, scope: str) -> tuple[list[list[str]], list[frozenset[str]]]:
    """Load durable history and the minimal zero-result antichain for one index."""
    with _LOCK:
        if not path.is_file():
            return [], []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return [], []
        scoped = payload.get("scopes", {}).get(scope, {})
        history = [
            [str(term) for term in attempt if str(term).strip()]
            for attempt in scoped.get("keyword_history", [])
            if isinstance(attempt, list)
        ]
        failed = _prune(
            item for item in scoped.get("failed_keyword_combinations", [])
            if isinstance(item, list)
        )
        return history, failed


def persist_keyword_memory(
    path: Path,
    scope: str,
    keyword_history: Iterable[Iterable[str]],
    failed_keyword_combinations: Iterable[Iterable[str]],
) -> None:
    """Merge one run into durable memory using an atomic file replacement."""
    with _LOCK:
        payload: dict[str, Any] = {"version": _VERSION, "scopes": {}}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("scopes"), dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        scopes = payload.setdefault("scopes", {})
        current = scopes.get(scope, {})
        existing_history = current.get("keyword_history", [])
        merged_history = [
            [str(term) for term in attempt if str(term).strip()]
            for attempt in [*existing_history, *keyword_history]
            if isinstance(attempt, (list, tuple))
        ]
        # Keep an audit trail without allowing an unbounded state file.
        merged_history = merged_history[-1000:]
        merged_failed = _prune([
            *current.get("failed_keyword_combinations", []),
            *failed_keyword_combinations,
        ])
        scopes[scope] = {
            "keyword_history": merged_history,
            "failed_keyword_combinations": [sorted(item) for item in merged_failed],
        }
        payload["version"] = _VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


def question_memory_key(question: str) -> str:
    """Return a stable, privacy-preserving key for one normalized question."""
    normalized = " ".join(str(question).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _clean_memory_text(value: object, limit: int = 480) -> str:
    """Keep persisted model feedback bounded and safe to render as plain evidence."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value))
    return " ".join(text.split())[:limit]


def _clean_memory_event(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, dict):
        return None
    outcome = _clean_memory_text(raw.get("outcome", ""), 80)
    if outcome not in {"candidates_found", "zero_results", "insufficient_coverage"}:
        return None
    terms = [
        _clean_memory_text(item, 120)
        for item in raw.get("terms", [])
        if _clean_memory_text(item, 120)
    ] if isinstance(raw.get("terms"), list) else []
    event: dict[str, object] = {"outcome": outcome, "terms": terms[:8]}
    for key, limit in (("reason", 480), ("suggestion", 320)):
        value = _clean_memory_text(raw.get(key, ""), limit)
        if value:
            event[key] = value
    tables = [
        _clean_memory_text(item, 180)
        for item in raw.get("tables", [])
        if _clean_memory_text(item, 180)
    ] if isinstance(raw.get("tables"), list) else []
    if tables:
        event["tables"] = tables[:8]
    evidence = raw.get("evidence", {})
    if isinstance(evidence, dict):
        cleaned_evidence = {
            _clean_memory_text(table, 180): _clean_memory_text(detail, 480)
            for table, detail in evidence.items()
            if _clean_memory_text(table, 180) and _clean_memory_text(detail, 480)
        }
        if cleaned_evidence:
            event["evidence"] = dict(list(cleaned_evidence.items())[:5])
    return event


def load_question_retrieval_memory(
    path: Path, scope: str, question_key: str
) -> list[dict[str, object]]:
    """Load bounded soft retrieval evidence for exactly one question."""
    with _LOCK:
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_events = payload.get("scopes", {}).get(scope, {}).get(
                "question_memories", {}
            ).get(question_key, [])
        except (OSError, json.JSONDecodeError, TypeError, AttributeError):
            return []
        if not isinstance(raw_events, list):
            return []
        return [
            event for event in (_clean_memory_event(item) for item in raw_events)
            if event is not None
        ][-_MAX_EVENTS_PER_QUESTION:]


def persist_question_retrieval_memory(
    path: Path, scope: str, question_key: str, events: Iterable[object]
) -> None:
    """Append question-scoped, bounded retrieval evidence without touching bans."""
    clean_events = [
        event for event in (_clean_memory_event(item) for item in events)
        if event is not None
    ]
    if not clean_events:
        return
    with _LOCK:
        payload: dict[str, Any] = {"version": _VERSION, "scopes": {}}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and isinstance(loaded.get("scopes"), dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError, TypeError):
                pass
        scopes = payload.setdefault("scopes", {})
        current = scopes.setdefault(scope, {})
        memories = current.setdefault("question_memories", {})
        if not isinstance(memories, dict):
            memories = current["question_memories"] = {}
        existing = memories.get(question_key, [])
        if not isinstance(existing, list):
            existing = []
        merged = [
            event for event in (_clean_memory_event(item) for item in [*existing, *clean_events])
            if event is not None
        ]
        # De-duplicate retry round writes while preserving their temporal order.
        unique: list[dict[str, object]] = []
        seen: set[str] = set()
        for event in merged:
            signature = json.dumps(event, sort_keys=True, ensure_ascii=False)
            if signature not in seen:
                unique.append(event)
                seen.add(signature)
        memories[question_key] = unique[-_MAX_EVENTS_PER_QUESTION:]
        while len(memories) > _MAX_QUESTIONS_PER_SCOPE:
            memories.pop(next(iter(memories)))
        payload["version"] = _VERSION
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)


def format_question_retrieval_memory(events: Iterable[dict[str, object]]) -> str:
    """Render persisted facts compactly; never present them as instructions."""
    lines: list[str] = []
    for event in list(events)[-_MAX_EVENTS_PER_QUESTION:]:
        terms = ", ".join(map(str, event.get("terms", []))) or "(question-only)"
        outcome = event.get("outcome")
        if outcome == "zero_results":
            lines.append(f"- [{terms}] returned zero local candidates.")
        elif outcome == "insufficient_coverage":
            tables = ", ".join(map(str, event.get("tables", [])))
            reason = str(event.get("reason", ""))
            suffix = f" Inspected tables: {tables}." if tables else ""
            lines.append(f"- [{terms}] was insufficient: {reason}.{suffix}")
            evidence = event.get("evidence", {})
            if isinstance(evidence, dict):
                for table, detail in list(evidence.items())[:3]:
                    lines.append(f"  Evidence for {table}: {detail}")
            suggestion = str(event.get("suggestion", ""))
            if suggestion:
                lines.append(f"  Recorded next-search focus: {suggestion}")
    return "\n".join(lines[:24])
