"""Tool calls as the providers return them, made runnable: stream fragments merged, a call leaked as text recovered, one call kept."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from llama_index.core.base.llms.types import TextBlock, ToolCallBlock

logger = logging.getLogger(__name__)


STREAM_END = object()


def next_stream_item(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return STREAM_END


def merge_generic_stream_tool_calls(
    accumulated: list[dict[str, Any]],
    event_data: dict[str, Any],
) -> None:
    fragments = event_data.get("message", {}).get("toolCalls", [])
    for index, fragment in enumerate(fragments):
        tool_id = fragment.get("id")
        current = next(
            (
                item
                for item in accumulated
                if tool_id is not None and item["toolUseId"] == tool_id
            ),
            None,
        )
        if current is None and index < len(accumulated):
            current = accumulated[index]
        if current is None:
            current = {
                "toolUseId": tool_id or uuid.uuid4().hex,
                "name": fragment.get("name", ""),
                "input": "",
            }
            accumulated.append(current)

        if tool_id:
            current["toolUseId"] = tool_id
        if fragment.get("name"):
            current["name"] = fragment["name"]
        current["input"] += fragment.get("arguments") or ""


def finalized_stream_tool_calls(
    accumulated: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    finalized = []
    for tool_call in accumulated:
        raw_input = (tool_call.get("input") or "").strip() or "{}"
        try:
            json.loads(raw_input)
        except json.JSONDecodeError:
            continue
        if not tool_call.get("name"):
            continue
        finalized.append({**tool_call, "input": raw_input})
    return finalized


def parse_tool_call_input(raw_input: Any) -> dict[str, Any] | None:
    if isinstance(raw_input, dict):
        return raw_input
    if raw_input is None or not str(raw_input).strip():
        return {}
    try:
        parsed = json.loads(str(raw_input))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


# gpt-oss on Together sometimes returns a whole harmony turn as plain text:
# "analysis<reasoning>assistantcommentary to=functions.<tool> json{<arguments>}".
LEAKED_TOOL_CALL = re.compile(r"(?:assistant)?commentary to=functions\.([A-Za-z_][\w-]*)[^{]{0,40}?\{", re.S)


def recover_leaked_tool_call(response) -> bool:
    """Turn a tool call that arrived as text into a real one; True when it did.

    Only a reply with no tool call of its own whose text holds the marker and a
    complete JSON object is changed: the call becomes a ToolCallBlock, the text
    before it is kept as the turn's reasoning and the visible content is emptied.
    """
    message = getattr(response, "message", None)
    if message is None or any(isinstance(block, ToolCallBlock) for block in message.blocks):
        return False
    if message.additional_kwargs.get("tool_calls"):
        return False
    text = message.content or ""
    match = LEAKED_TOOL_CALL.search(text)
    if not match:
        return False
    try:
        arguments, _ = json.JSONDecoder().raw_decode(text[match.end() - 1:])
    except json.JSONDecodeError:
        return False
    if not isinstance(arguments, dict):
        return False
    before = text[: match.start()].strip()
    before = before[len("analysis"):] if before.startswith("analysis") else before
    if before:
        message.additional_kwargs["reasoning"] = (
            message.additional_kwargs.get("reasoning", "") + before).strip()
    message.blocks = [TextBlock(text=""), ToolCallBlock(
        tool_call_id=f"call_{uuid.uuid4().hex[:24]}", tool_name=match.group(1), tool_kwargs=arguments)]
    message.additional_kwargs["recovered_tool_call"] = match.group(1)
    logger.warning("Together: recovered a %s call that arrived as text", match.group(1))
    return True


def keep_first_tool_call(response) -> None:
    """Drop all but the first tool call of a response, in both places a message keeps them."""
    message = response.message
    blocks = [block for block in message.blocks if isinstance(block, ToolCallBlock)]
    if len(blocks) > 1:
        message.blocks = [block for block in message.blocks if not isinstance(block, ToolCallBlock)] + blocks[:1]
    calls = message.additional_kwargs.get("tool_calls") or []
    if len(calls) > 1:
        message.additional_kwargs["tool_calls"] = calls[:1]
