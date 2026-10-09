import json
import re
from pathlib import Path

from lakegen.core.catalogue import clean_catalogue_text
from lakegen.domain.types import SolrMetadata
from lakegen.lakes.catalogue import match_local_csv, solr_metadata_from_doc, text

# The names the discovery agents and the tests read here.
_text = text


MAX_CANDIDATE_DESCRIPTION_CHARS = 320
MAX_CANDIDATE_TAGS = 8
MAX_CANDIDATE_COLUMNS = 12
MAX_COLUMN_DESCRIPTION_CHARS = 100

def _bounded_text(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _candidate_columns(meta: dict[str, object]) -> list[dict[str, str]]:
    structured = meta.get("columns", [])
    if isinstance(structured, list):
        valid = [column for column in structured if isinstance(column, dict)]
        if valid:
            return [
                {
                    "name": _text(column.get("name")),
                    "type": _text(column.get("type")),
                    "description": _text(column.get("description")),
                }
                for column in valid
                if _text(column.get("name"))
            ]

    # Compatibility with metadata recorded before structured columns existed.
    names = meta.get("columns.name", [])
    descriptions = meta.get("columns.description", [])
    types = meta.get("columns.type", [])
    if not isinstance(names, list):
        return []
    descriptions = descriptions if isinstance(descriptions, list) else []
    types = types if isinstance(types, list) else []
    return [
        {
            "name": _text(name),
            "type": _text(types[index]) if index < len(types) else "",
            "description": (
                _text(descriptions[index]) if index < len(descriptions) else ""
            ),
        }
        for index, name in enumerate(names)
        if _text(name)
    ]


def format_candidate_context(
    candidates: list[str], solr_meta: SolrMetadata, *, start_rank: int = 1
) -> str:
    """Render bounded metadata used to shortlist real table inspections.

    ``start_rank`` lets a later batch (e.g. from ``expand_candidates``)
    continue the same "Candidate N" numbering the agent already saw, rather
    than resetting to 1 and colliding with an earlier candidate's number --
    that numbering is also how ``inspect_columns(candidate_number=...)``
    resolves a candidate deterministically, without the agent retyping a
    long generated filename.
    """

    blocks: list[str] = []
    for display_rank, filename in enumerate(candidates, start=start_rank):
        meta = solr_meta.get(filename, {})
        title = _bounded_text(meta.get("title", "Unknown"), 200) or "Unknown"
        description = _bounded_text(
            meta.get("description", ""), MAX_CANDIDATE_DESCRIPTION_CHARS
        ) or "Not available"
        raw_tags = meta.get("tags", [])
        tags = raw_tags if isinstance(raw_tags, list) else [raw_tags]
        topics = ", ".join(
            _bounded_text(tag, 60) for tag in tags[:MAX_CANDIDATE_TAGS]
        ) or "No specific topics"

        retrieval = meta.get("retrieval", {})
        solr_rank = (
            retrieval.get("rank")
            if isinstance(retrieval, dict) and retrieval.get("rank") is not None
            else display_rank
        )
        columns = _candidate_columns(meta)
        raw_column_count = meta.get("column_count", len(columns))
        try:
            column_count = max(int(raw_column_count), len(columns))
        except (TypeError, ValueError):
            column_count = len(columns)

        lines = [
            f"Candidate {display_rank} (retrieval rank {solr_rank})",
            f"  File: {filename}",
            f"  Title: {title}",
            f"  Description: {description}",
            f"  Topics: {topics}",
            (
                "  Indexed schema preview: "
                f"{min(len(columns), MAX_CANDIDATE_COLUMNS)} of "
                f"{column_count} columns"
            ),
        ]
        for column in columns[:MAX_CANDIDATE_COLUMNS]:
            type_hint = f" [{column['type']}]" if column["type"] else ""
            description_hint = _bounded_text(
                column["description"], MAX_COLUMN_DESCRIPTION_CHARS
            )
            suffix = f" — {description_hint}" if description_hint else ""
            lines.append(f"    - {column['name']}{type_hint}{suffix}")
        omitted = column_count - min(len(columns), MAX_CANDIDATE_COLUMNS)
        if omitted > 0:
            lines.append(f"    ... {omitted} additional columns omitted")
        if not columns:
            lines.append("    - No indexed column metadata available")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + ("\n" if blocks else "")


def rejection_feedback_hint(
    reasoning: str, kept_tables: list[str], *, schemas_preloaded: bool = False
) -> str:
    """The next discovery round's hint after the architect rejected this one.

    Shared by the batch service and the chat UI so both hand the agent the
    same feedback. With tables kept, only the uncovered gap should move --
    pivoting every concept risks losing the coverage already found.
    ``schemas_preloaded`` marks kept tables whose inspection carries over.
    """
    if not kept_tables:
        return (
            "The previous keywords led to bad tables. Architect "
            f"feedback: {reasoning}. Generate completely different keywords."
        )
    hint = (
        "The previous attempt already found tables that "
        "satisfy part of the question; those are kept and "
        f"must not be searched for again. Architect feedback: {reasoning}. "
        "Keep the concepts, entities, and time constraints "
        "that led to the kept tables unchanged, and adjust "
        "only what is needed to cover the remaining gap "
        "described above."
    )
    if schemas_preloaded:
        hint += (
            f" {', '.join(kept_tables)} will already appear as "
            "an inspected candidate with its schema pre-loaded; "
            "do not spend a fresh inspect_columns call re-verifying "
            "it, spend that budget on the missing requirement instead."
        )
    return hint


def _final_payload(agent_resp: str) -> dict | None:
    """The JSON object after ``FINAL_PAYLOAD:``, whole, however deeply it nests.

    A judge's payload carries its selection plan, so the object has objects
    inside it; a non-greedy match to the first closing brace would cut it
    short and lose the reasoning. The decoder reads one complete object;
    quotes the model escaped as ``\\"`` are tried unescaped when the raw text
    does not decode.
    """
    match = re.search(r"FINAL_PAYLOAD:\s*", agent_resp)
    if match is None:
        return None
    decoder = json.JSONDecoder()
    for candidate in (agent_resp, agent_resp.replace('\\"', '"')):
        try:
            payload, _end = decoder.raw_decode(candidate, match.end())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def parse_table_selector_response(
    agent_resp: str,
    all_files: list[str],
    candidates: list[str],
) -> tuple[list[str], str]:
    reasoning = "No reasoning provided."
    selected_str = ""

    if "REJECT_KEYWORDS" in agent_resp:
        reason = agent_resp.replace("REJECT_KEYWORDS:", "", 1)
        reason = reason.replace("REJECT_KEYWORDS", "", 1).strip()
        return [], f"REJECT_KEYWORDS: {reason}"

    if "FINAL_PAYLOAD:" in agent_resp:
        payload = _final_payload(agent_resp)
        if payload is not None:
            tables = payload.get("tables", "")
            selected_str = ", ".join(map(str, tables)) if isinstance(tables, list) else str(tables)
            reasoning = str(payload.get("reasoning", ""))
        else:
            match = re.search(r"FINAL_PAYLOAD:\s*(\{.*?\})", agent_resp, re.DOTALL)
            selected_str = match.group(1) if match else agent_resp
    else:
        tables_match = re.search(r"(?i)TABLES:\s*(.*)", agent_resp)
        if tables_match:
            selected_str = tables_match.group(1).strip()

    selected_str = selected_str.replace("'", "").replace('"', "")
    
    selected = [
        name.strip() 
        for name in selected_str.split(",")
        if name.strip() in all_files
    ]

    if not selected:
        selected = candidates[:2]

    return selected, reasoning
