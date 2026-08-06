"""Small, provider-neutral helpers for structured run traces."""

from __future__ import annotations

import re


_TOOL_PATTERNS = (
    re.compile(r"Tool:\s*`([^`]+)`"),
    re.compile(r"ToolCall:\s*([\w.-]+)"),
    re.compile(r"tool_name=['\"]([\w.-]+)['\"]"),
)


def summarize_tool_calls(trace: str) -> list[dict[str, object]]:
    counts: dict[str, int] = {}
    for pattern in _TOOL_PATTERNS:
        for name in pattern.findall(trace or ""):
            counts[name] = counts.get(name, 0) + 1
    return [
        {"phase": "discovery", "type": name, "count": count}
        for name, count in sorted(counts.items())
    ]


def llm_call_record(phase: str, call_count: int, total_tokens: int) -> dict[str, object]:
    return {
        "phase": phase,
        "call_count": call_count,
        # Current phase APIs expose only totals. Null is explicit and avoids
        # presenting an invented prompt/completion split as measured data.
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": total_tokens,
        "usage_breakdown_available": False,
    }
