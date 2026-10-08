"""Sidebar of reference questions drawn from the UK benchmark.

The source JSON is the lake's ``reference_questions`` file (config/lakes.yaml):
a curated benchmark (``cases`` with the question, the gold tables under their
aliases, the reference result and its description), or the raw
generated-queries file build_benchmark.py reads (engine -> query_kind ->
group -> record, with per-group ``_meta.tables`` resolving table aliases
to ``dataset___resource`` ids). We only read from it; nothing here executes
or regenerates questions.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.config import FILES_DIRECTORY

from lakegen.config.registry import get_lake

from lakegen.ui.i18n import t

UK_REFERENCE_FILE: Path | None = get_lake("uk").reference_questions_path
DESCRIPTION_LIMIT = 220


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def _iter_success_records(payload: Any):
    if not isinstance(payload, dict):
        return
    for families in payload.values():
        if not isinstance(families, dict):
            continue
        for query_kind, groups in families.items():
            if not isinstance(groups, dict):
                continue
            for group in groups.values():
                if not isinstance(group, dict):
                    continue
                metadata = group.get("_meta", {})
                table_map = metadata.get("tables", {}) if isinstance(metadata, dict) else {}
                for record_key, record in group.items():
                    if record_key == "_meta" or not isinstance(record, dict):
                        continue
                    if record.get("status") == "success":
                        yield str(query_kind), table_map, record


def _golden_tables(record: dict[str, Any], table_map: dict[str, Any]) -> list[dict[str, str]]:
    tables = record.get("tables")
    if not isinstance(tables, list):
        return []
    resolved: list[dict[str, str]] = []
    for table in tables:
        alias = table.get("name") if isinstance(table, dict) else table
        if not isinstance(alias, str) or not alias:
            continue
        description = table.get("description") if isinstance(table, dict) else ""
        resolved.append(
            {
                "alias": alias,
                "table_id": str(table_map.get(alias, "")),
                "description": _truncate(str(description or ""), DESCRIPTION_LIMIT),
            }
        )
    return resolved


def _entry_from_record(
    query_kind: str, table_map: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any] | None:
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    golden_tables = _golden_tables(record, table_map)
    if not golden_tables:
        return None
    return {
        "id": str(record.get("client_id") or record.get("id") or question),
        "question": " ".join(question.split()),
        "query_kind": query_kind,
        "golden_tables": golden_tables,
        "expected_result": record.get("query_result"),
        "reference_response": record.get("response"),
    }


SIDEBAR_ELEMENT_SESSION_KEY = "uk_benchmark_sidebar_element"


@lru_cache(maxsize=1)
def load_uk_reference_questions() -> tuple[dict[str, Any], ...]:
    if UK_REFERENCE_FILE is None or not UK_REFERENCE_FILE.exists():
        return ()
    try:
        payload = json.loads(UK_REFERENCE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    candidates = (
        (_entry_from_case(case) for case in payload.get("cases", []))
        if isinstance(payload, dict) and isinstance(payload.get("cases"), list)
        else (_entry_from_record(kind, tables, record) for kind, tables, record in _iter_success_records(payload))
    )
    for entry in candidates:
        if entry is None or entry["id"] in seen:
            continue
        seen.add(entry["id"])
        entries.append(entry)
    return tuple(entries)


def _entry_from_case(case: Any) -> dict[str, Any] | None:
    """A curated benchmark case as the sidebar shows it: the gold tables by alias, the reference result."""
    if not isinstance(case, dict):
        return None
    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    aliases = case.get("table_aliases") if isinstance(case.get("table_aliases"), dict) else {}
    gold = case.get("relevant_table_ids") if isinstance(case.get("relevant_table_ids"), list) else []
    golden_tables = [
        {"alias": str(alias), "table_id": str(table_id), "description": ""}
        for alias, table_id in aliases.items()
    ] or [{"alias": str(table_id), "table_id": str(table_id), "description": ""} for table_id in gold]
    if not golden_tables:
        return None
    return {
        "id": str(case.get("id") or question),
        "question": " ".join(question.split()),
        "query_kind": str(case.get("expected_result_type") or "curated"),
        "golden_tables": golden_tables,
        "expected_result": case.get("reference_result"),
        "reference_response": _truncate(str(case.get("expected_result_description") or ""), 600),
    }


async def show_uk_benchmark_sidebar() -> None:
    """Open the element sidebar with the UK reference benchmark questions."""
    # Chainlit persists every element under this directory before it can
    # render it (element.py `_create` -> `session.persist_file`), but its own
    # `mkdir(exist_ok=True)` has no `parents=True` and the directory has been
    # observed to disappear under session churn, so recreate it defensively.
    FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)
    questions = load_uk_reference_questions()
    element = cl.CustomElement(
        name="ReferenceBenchmark",
        props={
            "available": bool(questions),
            "unavailableMessage": t("benchmark.unavailable", default="Benchmark unavailable."),
            "questions": list(questions),
            "executing": False,
        },
    )
    cl.user_session.set(SIDEBAR_ELEMENT_SESSION_KEY, element)
    await cl.ElementSidebar.set_title(
        t("benchmark.sidebar_title", default="UK Reference Benchmark")
    )
    await cl.ElementSidebar.set_elements([element], key="uk-reference-benchmark")


async def set_uk_benchmark_executing(executing: bool) -> None:
    """Reflect whether a LakeGen workflow run is in progress in the sidebar.

    No-op when the sidebar was never opened for this session (e.g. non-UK
    cores, or the reference file was unavailable).
    """
    element = cl.user_session.get(SIDEBAR_ELEMENT_SESSION_KEY)
    if element is None:
        return
    element.props["executing"] = executing
    await element.update()
