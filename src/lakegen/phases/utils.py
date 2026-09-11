import json
import os
import re
import uuid
from pathlib import Path

from lakegen.phases.logging import format_phase2_solr_results
from lakegen.core.types import SolrMetadata, StreamCallback
# from src.indexes.blend_indexer import BlendIndexer
import blend


MAX_CANDIDATE_DESCRIPTION_CHARS = 320
MAX_CANDIDATE_TAGS = 8
MAX_CANDIDATE_COLUMNS = 12
MAX_COLUMN_DESCRIPTION_CHARS = 100

def emit_agent_activity(
    activity_log_parts: list[str],
    stream_callback: StreamCallback | None,
    delta: str,
) -> None:
    if not delta:
        return
    activity_log_parts.append(delta)
    if stream_callback is not None:
        stream_callback(delta)


def match_local_csv(doc: dict, all_files: list[str]) -> str | None:
    dataset_id = doc.get("dataset_id")
    resource_id = doc.get("resource_id")
    return next(
        (
            filename
            for filename in all_files
            if (dataset_id and dataset_id in filename)
            or (resource_id and resource_id in filename)
        ),
        None,
    )


def solr_metadata_from_doc(doc: dict) -> dict[str, object]:
    tags = doc.get("tags", [])
    if not isinstance(tags, list):
        tags = [str(tags)]

    columns = doc.get("columns", [])
    if not isinstance(columns, list):
        columns = []
    structured_columns = [
        {
            "name": str(column.get("name", "")).strip(),
            "description": str(column.get("description", "")).strip(),
            "type": str(column.get("type", "")).strip(),
        }
        for column in columns
        if isinstance(column, dict) and str(column.get("name", "")).strip()
    ]
    return {
        "title": doc.get("title", ""),
        "description": doc.get("description", ""),
        "tags": [str(tag) for tag in tags],
        "columns.name": [column["name"] for column in structured_columns],
        "columns.description": [
            column["description"]
            for column in structured_columns
            if column["description"]
        ],
        "columns.type": [
            column["type"] for column in structured_columns if column["type"]
        ],
        # Retain per-column alignment for the bounded agent-facing preview.
        "columns": structured_columns,
        "column_count": len(columns),
    }


def emit_candidate_summary(
    candidates: list[str],
    metadata: SolrMetadata,
    activity_log_parts: list[str],
    stream_callback: StreamCallback | None,
) -> None:
    emit_agent_activity(
        activity_log_parts,
        stream_callback,
        format_phase2_solr_results(candidates, metadata, "Candidate tables")
        + "\n\n---\n\n**Agent activity log**\n",
    )


def prepare_candidate_index(
    candidates: list[str],
    csv_dir: Path,
    db_path: Path,
    activity_log_parts: list[str],
    stream_callback: StreamCallback | None = None,
) -> Path:
    blend_db = db_path.parent / f"temp_blend_{uuid.uuid4().hex}.db"
    try:
        print(
            f"[phase2 tables] building BLEND index db={blend_db.name} "
            f"files={candidates}",
            flush=True,
        )
        emit_agent_activity(
            activity_log_parts,
            stream_callback,
            "\n**Preparing BLEND index**\n"
            f"- Candidate files: `{len(candidates)}`\n"
            f"- Temporary DB: `{blend_db.name}`\n",
        )

        indexer = blend.BLEND(db_path=db_path)
        _blend_load_opts = {"ignore_errors": True, "infer_schema_length": 0, "n_rows": 10000}
        blend.index_tables_seq(indexer, csv_dir, load_opts=_blend_load_opts, log_stdout=True)

        # indexer = BlendIndexer(csv_dir=csv_dir, db_path=blend_db)
        # indexer.build_index(specific_files=candidates, silent=True)
        print(f"[phase2 tables] BLEND ready db={blend_db.name}", flush=True)
        emit_agent_activity(activity_log_parts, stream_callback, "- Status: `ready`\n")
    except Exception:
        if blend_db.exists():
            try:
                os.remove(blend_db)
            except Exception:
                pass
        raise

    return blend_db


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
                    "name": str(column.get("name", "")).strip(),
                    "type": str(column.get("type", "")).strip(),
                    "description": str(column.get("description", "")).strip(),
                }
                for column in valid
                if str(column.get("name", "")).strip()
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
            "name": str(name),
            "type": str(types[index]) if index < len(types) else "",
            "description": (
                str(descriptions[index]) if index < len(descriptions) else ""
            ),
        }
        for index, name in enumerate(names)
        if str(name).strip()
    ]


def format_candidate_context(candidates: list[str], solr_meta: SolrMetadata) -> str:
    """Render bounded metadata used to shortlist real table inspections."""

    blocks: list[str] = []
    for display_rank, filename in enumerate(candidates, start=1):
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
            f"Candidate {display_rank} (Solr rank {solr_rank})",
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
        match = re.search(r"FINAL_PAYLOAD:\s*(\{.*?\})", agent_resp, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(1).replace('\\"', '"'))
                selected_str = payload.get("tables", "")
                reasoning = payload.get("reasoning", "")
            except json.JSONDecodeError:
                selected_str = match.group(1)
        else:
            selected_str = agent_resp
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
