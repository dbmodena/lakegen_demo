"""Shared, architecture-neutral requirement ledger construction."""

from __future__ import annotations

import re


_COMPUTATIONAL_CUES = re.compile(
    r"\b(?:aggregat|average|bin|bucket|calculat|comput|correlat|count|derive|"
    r"group|rank|ratio|sort|top|range categor|range bin)\w*\b",
    re.IGNORECASE,
)


def is_computational_requirement(value: object) -> bool:
    """Return whether a missing item describes work rather than source data."""
    text = " ".join(str(value or "").split())
    return bool(_COMPUTATIONAL_CUES.search(text))


def build_minimal_selection_fallback(
    selected: list[str], reasoning: str
) -> tuple[dict[str, object], list[str]]:
    """Mark an unvalidated discovery recovery identically for P2 and P12."""
    if not selected:
        return {}, []
    lowered = reasoning.casefold()
    if len(selected) == 1:
        strategy = "single_table"
    elif any(term in lowered for term in ("concat", "partition", "append", "union")):
        strategy = "concat_partitions"
    elif any(term in lowered for term in ("lookup", "mapping", "reference table")):
        strategy = "lookup"
    elif any(term in lowered for term in ("compare", "comparison", "versus", " vs ")):
        strategy = "compare"
    elif any(term in lowered for term in ("join", "merge", "shared key")):
        strategy = "join"
    else:
        strategy = "aggregate_separately"
    return {
        "requirement_coverage": {},
        "table_roles": {
            table: (
                "primary selected source" if index == 0
                else "supporting selected source"
            )
            for index, table in enumerate(selected)
        },
        "combination_strategy": strategy,
        "uncovered_requirements": [],
        "alternatives_rejected": {},
        "recovered_from_existing_discovery_context": True,
    }, [
        "The structured selection plan was recovered from the existing discovery "
        "decision. The coder may complete computational details from the question "
        "and runtime schema; unsupported data requirements must still be rejected."
    ]


def requirement_ledger_blockers(
    ledger: list[dict[str, object]], selected_tables: list[str]
) -> list[str]:
    """Report data-bound requirements that lack concrete selected-table evidence."""
    selected = {str(table).strip() for table in selected_tables}
    blockers: list[str] = []
    for item in ledger:
        if item.get("status") == "computational":
            continue
        request = str(item.get("request") or "").strip()
        if item.get("status") == "unresolved":
            blockers.append(request)
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, dict):
            blockers.append(request)
            continue
        table = str(evidence.get("table") or "").strip()
        columns = evidence.get("columns")
        if table == "join" and isinstance(columns, list) and columns:
            annotated_tables = {
                match.group(1).strip()
                for column in columns
                if (match := re.search(r"\(([^()]+)\)\s*$", str(column)))
            }
            if annotated_tables and annotated_tables <= selected:
                continue
        if table not in selected or not isinstance(columns, list) or not any(
            str(column or "").strip() for column in columns
        ):
            blockers.append(request)
    return list(dict.fromkeys(blockers))


def build_requirement_ledger(
    question: str,
    coverage: dict[str, dict[str, object]],
    requirements: dict[str, object],
    uncovered: list[str],
    semantic_plan: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Build the same compact coder contract for divided and unified discovery."""
    ledger: list[dict[str, object]] = []
    stopwords = {
        "a", "an", "and", "by", "for", "from", "in", "of", "per",
        "requested", "the", "to", "with", "measure", "dimension", "filter",
    }

    def text_for(value: object) -> str:
        if isinstance(value, dict):
            return " ".join(str(part).strip() for part in (
                value.get("output") or value.get("alias"),
                value.get("operation") or value.get("type"), value.get("column"),
            ) if part).strip()
        return " ".join(str(value or "").split())

    def tokens(value: object) -> set[str]:
        normalized = text_for(value).casefold().replace("boro", "borough")
        return {token for token in re.findall(r"[a-z0-9]+", normalized)
                if len(token) >= 3 and token not in stopwords}

    def add(kind: str, request: object, status: str, evidence: object = None):
        text = text_for(request)
        if not text:
            return None
        compatible = {
            "dimension": {"dimension", "geographic_filter"},
            "measure": {"measure"},
            "filter": {"filter", "temporal_filter", "geographic_filter"},
            "temporal_filter": {"temporal_filter"},
        }
        new_tokens = tokens(text)
        for item in ledger:
            if item["kind"] not in compatible.get(kind, {kind}):
                continue
            old_tokens = tokens(item["request"])
            if kind == "output" or (new_tokens and old_tokens and (
                new_tokens <= old_tokens or old_tokens <= new_tokens
                or len(new_tokens & old_tokens) >= 2
            )):
                if status == "bound":
                    item.update(status="bound", evidence=evidence or [])
                elif status == "computational":
                    item["computation"] = text
                return item
        item = {"kind": kind, "request": text, "status": status,
                "evidence": evidence if evidence is not None else []}
        ledger.append(item)
        return item

    def classify(text: str) -> str:
        lowered = text.casefold()
        if re.search(r"year|date|fy\b|school year|period", lowered):
            return "temporal_filter"
        if re.search(r"borough|city|district|location|geograph", lowered):
            return "geographic_filter"
        if re.search(r"join|key|relationship|match", lowered):
            return "join"
        if re.search(r"count|average|mean|sum|total|rate|length|code", lowered):
            return "measure"
        return "data_requirement"

    for request, raw_evidence in coverage.items():
        evidence = dict(raw_evidence) if isinstance(raw_evidence, dict) else raw_evidence
        add(classify(str(request)), request, "bound", evidence)
    for request in uncovered:
        if is_computational_requirement(request):
            add("derived_operation", request, "computational")
        else:
            add(classify(request), request, "unresolved")

    plan = semantic_plan or {}
    for group, kind in (("filters", "filter"),
                        ("temporal_filters", "temporal_filter"),
                        ("dimensions", "dimension"),
                        ("measures", "measure")):
        values = plan.get(group, [])
        for binding in values if isinstance(values, list) else []:
            if isinstance(binding, dict):
                request = binding.get("requirement") or binding.get("output") or binding.get("column")
                columns = binding.get("columns") or [binding.get("column")]
                if group == "measures" and not any(columns):
                    add("measure", request, "computational")
                else:
                    add(kind, request, "bound", {"table": binding.get("table"),
                        "columns": columns})

    for key, kind in (("grouping", "dimension"), ("measures", "measure")):
        values = requirements.get(key, [])
        for value in values if isinstance(values, list) else []:
            add(kind, value, "computational")
    for value in requirements.get("output_columns", []) if isinstance(
        requirements.get("output_columns"), list
    ) else []:
        add("output", value, "computational")
    for key, kind in (("ordering", "ordering"), ("limit", "limit"),
                      ("result_type", "output")):
        value = requirements.get(key)
        if value not in (None, "", [], "auto"):
            add(kind, value, "computational")

    periods = list(dict.fromkeys([
        *re.findall(r"\b(?:19|20)\d{2}\s*[-–‑/]\s*(?:\d{2}|(?:19|20)\d{2})\b", question),
        *re.findall(r"\b(?:19|20)\d{2}\b", question),
    ]))
    for period in periods:
        matching = next((evidence for request, evidence in coverage.items()
                         if period.casefold() in str(request).casefold()), None)
        add("temporal_scope", period, "bound" if matching else "unresolved",
            matching if matching else [])
    boroughs = [name for name in ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
                if re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE)]
    if boroughs:
        matching = next((evidence for request, evidence in coverage.items()
                         if any(name.casefold() in str(request).casefold()
                                or "borough" in str(request).casefold()
                                for name in boroughs)), None)
        add("geographic_scope", " and ".join(boroughs),
            "bound" if matching else "unresolved", matching if matching else [])
    for marker, label in ((r"correlat", "correlation"),
                          (r"geographic(?:al)? center", "geographic center"),
                          (r"\bratio\b", "ratio")):
        if re.search(marker, question, re.IGNORECASE):
            add("derived_operation", label, "computational")

    lowered = question.casefold()
    declared = str(requirements.get("result_type") or "auto").casefold()
    table_cues = bool(re.search(r"\b(?:for each|each borough|each district|which\s+(?:three|five|\d+)|top\s+\d+)\b", lowered))
    scalar_cues = bool(re.search(r"\b(?:correlat|ratio|how many|what (?:is|was) the (?:average|total|number))", lowered)) and not table_cues
    shape = ("scalar" if declared == "number" else "table" if declared == "table"
             else "scalar" if scalar_cues else "table" if table_cues else "unknown")
    derived = any(item["kind"] == "derived_operation" for item in ledger)
    for item in ledger:
        if item["status"] != "computational" and "computation" not in item:
            continue
        if item["kind"] == "dimension":
            item["role"] = "intermediate" if shape == "scalar" else "final"
        elif item["kind"] == "measure":
            item["role"] = "intermediate" if derived else "final"
        elif item["kind"] in {"derived_operation", "output"}:
            item["role"] = "final"
    if shape != "unknown":
        output = add("output", f"final {shape} answer", "computational")
        if output is not None:
            output.update(role="final", shape=shape)

    order = {"bound": 0, "unresolved": 1, "computational": 2}
    ordered = sorted(ledger, key=lambda item: order[str(item["status"])])
    if len(ordered) <= 10:
        return ordered
    final_items = [item for item in ordered if item.get("role") == "final"][:3]
    return [*[item for item in ordered if item not in final_items][:10-len(final_items)],
            *final_items]
