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


def _has_valid_table_and_columns(evidence: object) -> bool:
    """True for a well-formed {"table": "...", "columns": ["..."]} evidence
    dict -- mirrors requirement_ledger_blockers' own validity check, reused
    here to decide whether a ledger item's evidence needs repairing from a
    more reliable structured source."""
    if not isinstance(evidence, dict):
        return False
    table = str(evidence.get("table") or "").strip()
    columns = evidence.get("columns")
    return bool(table) and isinstance(columns, list) and any(
        str(column or "").strip() for column in columns
    )


def build_minimal_selection_fallback(
    selected: list[str], reasoning: str
) -> tuple[dict[str, object], list[str]]:
    """Mark an unvalidated discovery recovery identically for P2 and P12."""
    if not selected:
        return {}, []
    # A narrative mentioning a join is not evidence for a join. Leave the
    # relationship open until the coder inspects the missing source evidence.
    strategy = "single_table" if len(selected) == 1 else "unspecified"
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
            # Similar wording does not make two periods or source bindings
            # interchangeable (e.g. "year 2014 trips" and "year 2022 trips").
            if set(re.findall(r"\b\d+\b", text)) != set(re.findall(r"\b\d+\b", item["request"])):
                continue
            if status == "bound" and item["status"] == "bound" and item["evidence"] != evidence:
                continue
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
    # requirements.filters carries an explicit table/column/value binding for
    # exactly this kind of literal filter -- more reliable than the coverage
    # substring check below, which only matches when the architect happens to
    # repeat the literal year inside its free-text requirement_coverage key.
    # Confirmed missing on a live run: the architect correctly bound
    # "Financial Year" -> a fiscal-year filter in requirements.filters
    # (value=2016), yet selection was blocked anyway because only the
    # coverage keys were ever consulted for period binding.
    filter_entries = requirements.get("filters", [])
    if not isinstance(filter_entries, list):
        filter_entries = []

    def _filter_value_evidences_period(period: str, value: object) -> bool:
        """True if `value` (a requirements.filters entry's `value`) evidences
        `period` -- either a scalar equals-style match (value == "2016"), or
        a [start, end] range-style temporal filter (e.g.
        ["1990-01-01", "1990-12-31"] for the year 1990), or an isin-style
        list of candidates. A range's bounds are date strings that CONTAIN
        the bare year, not equal it, so this checks substring containment,
        not just exact equality -- confirmed missing on a second live run:
        the architect correctly bound StartDate to a
        {"operator": "range", "value": ["1990-01-01", "1990-12-31"]} filter
        for "the year 1990", but the original scalar-only equality check
        (`str(value) == period`) never matched a list value at all, so
        selection kept blocking across 8 repeated confirm attempts until the
        architect gave up and returned unparsed text instead of a tool call."""
        candidates = value if isinstance(value, (list, tuple)) else [value]
        return any(period in str(candidate).strip() for candidate in candidates)

    for period in periods:
        matching = next((evidence for request, evidence in coverage.items()
                         if period.casefold() in str(request).casefold()), None)
        if matching is None:
            matched_filter = next(
                (item for item in filter_entries
                 if isinstance(item, dict) and _filter_value_evidences_period(period, item.get("value"))),
                None,
            )
            if matched_filter is not None:
                table = matched_filter.get("table")
                column = matched_filter.get("column")
                if table and column:
                    matching = {"table": table, "columns": [column]}
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

    # A `requirement_coverage` measure entry ("observation count" -> {...})
    # can carry malformed evidence (columns as a bare string, an empty
    # list, or None; table missing) while `requirements.measures` has the
    # SAME measure correctly bound with a real table and columns --
    # confirmed live: the architect wrote a well-formed
    # {"table": ..., "columns": ["RecordKey"], "operation": "count_rows"}
    # measures entry, yet selection still blocked on "observation count"
    # because only the free-text coverage evidence was ever validated. Same
    # "prefer the structured requirements sub-block over unreliable
    # free-text coverage" principle as the period-binding fallback above,
    # applied to measures specifically since that's the shape observed.
    measure_entries = requirements.get("measures", [])
    if isinstance(measure_entries, list):
        for item in ledger:
            if item["kind"] != "measure" or item["status"] != "bound":
                continue
            if _has_valid_table_and_columns(item.get("evidence")):
                continue
            for m in measure_entries:
                if not isinstance(m, dict):
                    continue
                table = str(m.get("table") or "").strip()
                columns = m.get("columns")
                if not isinstance(columns, list):
                    columns = [m.get("column")] if m.get("column") else []
                columns = [str(c).strip() for c in columns if str(c or "").strip()]
                if table and columns:
                    item["evidence"] = {"table": table, "columns": columns}
                    break

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
