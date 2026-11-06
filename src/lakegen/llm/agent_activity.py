"""How an agent's activity reads in the stream: candidate lists, tool calls and results, stalls."""

import json
import re
from typing import Any

from lakegen.datastructures.types import SolrMetadata


def format_phase2_solr_results(
    candidates: list[str],
    solr_meta: SolrMetadata,
    heading: str = "Solr candidate tables",
) -> str:
    """A markdown list of the candidates, numbered, each with its title.

    Args:
        candidates: The candidate table ids.
        solr_meta: Their metadata, by id.
        heading: The list's bold heading.

    Returns:
        The markdown; a "no candidates" line when the list is empty.
    """
    lines = [f"**{heading}**"]
    if not candidates:
        lines.append("_No candidates found._")
        return "\n".join(lines)

    for idx, candidate in enumerate(candidates, 1):
        title = solr_meta.get(candidate, {}).get("title") or "No title"
        lines.append(f"{idx}. `{candidate}` - {title}")
    return "\n".join(lines)


PHASE2_TOOL_LABELS = {
    "search_tables": "Search tables",
    "inspect_columns": "Inspect columns",
    "preview_data": "Preview rows",
    "check_join_union": "Check join/union",
    "confirm_table_selection": "Confirm table selection",
}


def compact_phase2_text(value: Any, max_len: int = 900) -> str:
    """The value as text, triple backticks defused, cut to ``max_len`` with "..."; "" for None."""
    if value is None:
        return ""

    text = str(value).replace("```", "'''").strip()
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def format_phase2_tool_args(event: Any) -> str:
    """A tool-call event's arguments as compact JSON; ``No arguments`` when it has none."""
    args = (
        getattr(event, "tool_kwargs", None)
        or getattr(event, "kwargs", None)
        or getattr(event, "tool_input", None)
        or getattr(event, "input", None)
    )
    if args is None or (isinstance(args, (dict, list, tuple, set)) and not args):
        return "No arguments"

    try:
        return compact_phase2_text(
            json.dumps(args, ensure_ascii=False, default=str),
            max_len=700,
        )
    except TypeError:
        return compact_phase2_text(args, max_len=700)


def format_phase2_tool_output(tool_output: Any) -> str:
    """A tool's output as compact text: its content, raw output or output, else the object itself."""
    for attr in ("content", "raw_output", "output"):
        value = getattr(tool_output, attr, None)
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        return compact_phase2_text(value)
    return compact_phase2_text(tool_output)


def format_phase2_tool_call(event: Any, call_number: int) -> str:
    """The markdown block announcing a tool call: its number, label, tool and arguments."""
    tool_name = getattr(event, "tool_name", "unknown_tool")
    label = PHASE2_TOOL_LABELS.get(tool_name, "Run tool")
    args = format_phase2_tool_args(event)
    output = f"\n\n**Phase 2 tool #{call_number}: {label}**\n- Tool: `{tool_name}`\n"
    if args == "No arguments":
        return output + "- Args: `No arguments`\n"
    return output + f"- Args:\n```json\n{args}\n```\n"


def format_phase2_tool_result(event: Any, call_number: int) -> str:
    """The markdown block of a tool's result: its number, tool, status and output."""
    tool_name = getattr(event, "tool_name", "unknown_tool")
    tool_output = getattr(event, "tool_output", None)
    is_error = bool(getattr(tool_output, "is_error", False))
    status = "error" if is_error else "done"
    output = format_phase2_tool_output(tool_output)

    result = (
        f"\n**Phase 2 tool #{call_number} result**\n- Tool: `{tool_name}`\n- Status: `{status}`\n"
    )
    if output:
        result += f"\n```text\n{output}\n```\n"
    return result


def extract_phase2_activity_log(full_trace: str) -> str:
    """The activity log of a trace: what follows its activity-log marker, without the candidate summary."""
    marker = "--- Phase 2 Activity Log ---"
    if marker in full_trace:
        activity_log = full_trace.split(marker, 1)[1].strip()
    else:
        activity_log = full_trace.strip()

    candidate_summary_end = "\n\n---\n\n**Agent activity log**\n"
    if activity_log.startswith("**Candidate tables**") and candidate_summary_end in activity_log:
        return activity_log.split(candidate_summary_end, 1)[1].strip()

    return activity_log


class AgentStall(RuntimeError):
    """An agent run stopped because it stalled: it repeats itself, or hit a call, size or time limit."""


PHASE2_STALL_PHRASES = (
    "i can answer without using any more tools",
    "i need the data to answer",
    "i can't answer without the data",
    "there is no data",
    "ask for the data",
    "need the data to answer the question",
)

PHASE2_MAX_STREAM_CHARS = 20000


def normalize_phase2_stall_text(text: str) -> str:
    """The text lower-cased, whitespace collapsed, for counting repeats."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_phase2_agent_stall(stream_text: str, min_repeats: int = 5) -> str | None:
    """Why the streamed text shows a stalled agent, or None.

    A stall is a stream longer than ``PHASE2_MAX_STREAM_CHARS``, or one that
    repeats a stall phrase, an observation block, a tool action with the same
    input, or a thought or bullet line of 35 characters or more.

    Args:
        stream_text: The text the agent streamed so far.
        min_repeats: How many repeats make a stall.

    Returns:
        The reason, or None.
    """
    if len(stream_text) > PHASE2_MAX_STREAM_CHARS:
        return f"stream exceeded {PHASE2_MAX_STREAM_CHARS} characters without finishing"

    normalized_stream = normalize_phase2_stall_text(stream_text)
    for phrase in PHASE2_STALL_PHRASES:
        if normalized_stream.count(phrase) >= min_repeats:
            return f"repeated phrase: {phrase}"

    if normalized_stream.count("observation:") >= min_repeats:
        return "repeated observation blocks"

    repeated_actions = re.findall(
        r"(?is)action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*action input:\s*(\{.*?\})",
        stream_text,
    )
    action_counts: dict[str, int] = {}
    for tool_name, raw_args in repeated_actions:
        key = f"{tool_name}:{normalize_phase2_stall_text(raw_args)}"
        action_counts[key] = action_counts.get(key, 0) + 1
        if action_counts[key] >= min_repeats:
            return f"repeated tool action: {tool_name}"

    thought_lines = re.findall(
        r"(?im)^\s*(?:thought:\s*|wait,\s*)(.+)$",
        stream_text,
    )
    counts: dict[str, int] = {}
    repeated_lines = thought_lines + re.findall(r"(?m)^\s*-\s+(.+)$", stream_text)
    for line in repeated_lines:
        normalized_line = normalize_phase2_stall_text(line)
        if len(normalized_line) < 35:
            continue
        counts[normalized_line] = counts.get(normalized_line, 0) + 1
        if counts[normalized_line] >= min_repeats:
            return f"repeated thought: {normalized_line[:120]}"

    return None
