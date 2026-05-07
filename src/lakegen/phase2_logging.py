import json
import re
from typing import Any

from lakegen.types import SolrMetadata


def format_phase2_solr_results(
    candidates: list[str],
    solr_meta: SolrMetadata,
    heading: str = "Solr candidate tables",
) -> str:
    lines = [f"**{heading}**"]
    if not candidates:
        lines.append("_No candidates found._")
        return "\n".join(lines)

    for idx, candidate in enumerate(candidates, 1):
        title = solr_meta.get(candidate, {}).get("title") or "No title"
        lines.append(f"{idx}. `{candidate}` - {title}")
    return "\n".join(lines)


def format_cli_log_value(value: Any, max_len: int = 160) -> str:
    if value is None:
        return "None"

    text = str(value).replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return repr(text)


PHASE2_TOOL_LABELS = {
    "inspect_columns": "Inspect columns",
    "preview_data": "Preview rows",
    "find_joinable_tables": "Search joinable tables",
    "find_exact_overlaps": "Check exact overlaps",
    "find_schema_matches": "Check schema matches",
    "confirm_table_selection": "Confirm table selection",
}


def compact_phase2_text(value: Any, max_len: int = 900) -> str:
    if value is None:
        return ""

    text = str(value).replace("```", "'''").strip()
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


def format_phase2_tool_args(event: Any) -> str:
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
    for attr in ("content", "raw_output", "output"):
        value = getattr(tool_output, attr, None)
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        return compact_phase2_text(value)
    return compact_phase2_text(tool_output)


def format_phase2_tool_call(event: Any, call_number: int) -> str:
    tool_name = getattr(event, "tool_name", "unknown_tool")
    label = PHASE2_TOOL_LABELS.get(tool_name, "Run tool")
    args = format_phase2_tool_args(event)
    output = (
        f"\n\n**Phase 2 tool #{call_number}: {label}**\n"
        f"- Tool: `{tool_name}`\n"
    )
    if args == "No arguments":
        return output + "- Args: `No arguments`\n"
    return output + f"- Args:\n```json\n{args}\n```\n"


def format_phase2_tool_result(event: Any, call_number: int) -> str:
    tool_name = getattr(event, "tool_name", "unknown_tool")
    tool_output = getattr(event, "tool_output", None)
    is_error = bool(getattr(tool_output, "is_error", False))
    status = "error" if is_error else "done"
    output = format_phase2_tool_output(tool_output)

    result = (
        f"\n**Phase 2 tool #{call_number} result**\n"
        f"- Tool: `{tool_name}`\n"
        f"- Status: `{status}`\n"
    )
    if output:
        result += f"\n```text\n{output}\n```\n"
    return result


def extract_phase2_activity_log(full_trace: str) -> str:
    marker = "--- Phase 2 Activity Log ---"
    if marker in full_trace:
        activity_log = full_trace.split(marker, 1)[1].strip()
    else:
        activity_log = full_trace.strip()

    candidate_summary_end = "\n\n---\n\n**Agent activity log**\n"
    if activity_log.startswith("**Candidate tables**") and candidate_summary_end in activity_log:
        return activity_log.split(candidate_summary_end, 1)[1].strip()

    return activity_log


class Phase2AgentStall(RuntimeError):
    pass


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
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_phase2_agent_stall(stream_text: str, min_repeats: int = 3) -> str | None:
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
