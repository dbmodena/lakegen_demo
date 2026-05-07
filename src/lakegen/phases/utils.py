import json
import os
import re
import uuid
from pathlib import Path

from lakegen.phase2_logging import format_phase2_solr_results
from lakegen.types import SolrMetadata, StreamCallback
from src.indexes.blend_indexer import BlendIndexer


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
    return {
        "title": doc.get("title", ""),
        "description": doc.get("description", ""),
        "tags": [str(tag) for tag in tags],
        "columns.name": [col.get("name") for col in columns if col.get("name")],
        "columns.type": [col.get("type") for col in columns if col.get("type")],
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

        indexer = BlendIndexer(csv_dir=csv_dir, db_path=blend_db)
        indexer.build_index(specific_files=candidates, silent=True)
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


def format_candidate_context(candidates: list[str], solr_meta: SolrMetadata) -> str:
    lines = []
    for filename in candidates:
        meta = solr_meta.get(filename, {})
        title = meta.get("title", "Unknown")
        topics = ", ".join(meta.get("tags", ["No specific topics"])[:15])
        lines.append(f"- File: {filename}\n  Title: {title}\n  Topics: {topics}")
    return "\n\n".join(lines) + ("\n" if lines else "")


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
    selected = [name.strip() for name in selected_str.split(",")
                if name.strip() in all_files]
    if not selected:
        selected = candidates[:2]

    return selected, reasoning
