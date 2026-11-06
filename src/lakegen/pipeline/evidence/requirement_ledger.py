"""Shared, architecture-neutral requirement ledger construction."""

from __future__ import annotations

import re
from collections.abc import Mapping

_COMPUTATIONAL_CUES = re.compile(
    r"\b(?:aggregat|average|bin|bucket|calculat|comput|correlat|count|derive|"
    r"group|rank|ratio|sort|top|range categor|range bin)\w*\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def _period_years(period: str) -> list[int]:
    """Calendar years a question period spans.

    "2023" -> [2023], "2019-20" and "2019/2020" -> [2019, 2020].

    Args:
        period: A year or year range as the question writes it.

    Returns:
        The years, in order; empty when ``period`` names none.
    """
    years = [int(year) for year in _YEAR.findall(period)]
    short_end = re.search(r"[-–‑/]\s*(\d{2})$", period.strip())
    if len(years) == 1 and short_end:
        end = years[0] // 100 * 100 + int(short_end.group(1))
        years.append(end if end >= years[0] else end + 100)
    return years


def _inspection_shows_years(inspection: str, column: str, years: list[int]) -> str | None:
    """How inspect_columns output shows ``column`` holding every one of ``years``.

    "values" when they appear in its name or sampled values
    ("- col (Category sample): ['2023/05/16 ...']"), "range" when they fall
    inside its measured temporal coverage ("- col: 2007-07-10 to 2023-02-24
    (...)"), or None.

    Args:
        inspection: The inspect_columns output of one table.
        column: The column to look for.
        years: The years the column must hold.

    Returns:
        "values", "range" or None.
    """
    shown = None
    for raw_line in inspection.splitlines():
        line = raw_line.strip()
        if line.startswith(f"- {column} ("):
            if set(years) <= {int(year) for year in _YEAR.findall(line)}:
                return "values"
        elif line.startswith(f"- {column}: "):
            bounds = re.match(r"(.+?) to (.+?)(?:\s+\(|$)", line[len(column) + 4 :])
            start = _YEAR.search(bounds.group(1)) if bounds else None
            end = _YEAR.search(bounds.group(2)) if bounds else None
            if (
                start
                and end
                and all(int(start.group(1)) <= year <= int(end.group(1)) for year in years)
            ):
                shown = "range"
    return shown


def question_periods(question: str) -> list[str]:
    """Years and year ranges the question names, e.g. "2019-20" and "2023".

    Args:
        question: The user question.

    Returns:
        The periods in order of appearance, ranges before single years, without repeats.
    """
    return list(
        dict.fromkeys(
            [
                *re.findall(r"\b(?:19|20)\d{2}\s*[-–‑/]\s*(?:\d{2}|(?:19|20)\d{2})\b", question),
                *re.findall(r"\b(?:19|20)\d{2}\b", question),
            ]
        )
    )


def inspected_period_evidence(inspection: str, period: str) -> tuple[str, str] | None:
    """The column an inspect_columns profile shows holding ``period``, and how.

    ("Dataset_Da", "values") or ("date_licen", "range"). Values beat ranges.

    Args:
        inspection: The inspect_columns output of one table.
        period: A year or year range from the question.

    Returns:
        The column and "values" or "range", or None when no column shows the period.
    """
    years = _period_years(period)
    if not years:
        return None
    columns = dict.fromkeys(
        match.group(1) for match in re.finditer(r"^- (.+?)(?: \(|: )", inspection, re.MULTILINE)
    )
    found: dict[str, str] = {}
    for column in columns:
        strength = _inspection_shows_years(inspection, column, years)
        if strength:
            found.setdefault(strength, column)
    for strength in ("values", "range"):
        if strength in found:
            return found[strength], strength
    return None


def _inspected_period_binding(
    period: str,
    bindings: list[tuple[str, list[str]]],
    inspections: Mapping[str, str],
) -> dict[str, object] | None:
    """Bind ``period`` to a column the agent itself bound whose inspection shows it.

    Never to a column the agent did not choose. A column whose values name the
    period beats one whose date range merely spans it.

    Args:
        period: A year or year range from the question.
        bindings: Every (table, columns) the agent bound.
        inspections: inspect_columns output by casefolded table name.

    Returns:
        ``{"table": ..., "columns": [column]}``, or None when no bound column shows the period.
    """
    years = _period_years(period)
    if not years:
        return None
    by_strength: dict[str, dict[str, object]] = {}
    for table, columns in bindings:
        inspection = str(inspections.get(table.casefold(), ""))
        for column in columns:
            strength = _inspection_shows_years(inspection, column, years)
            if strength:
                by_strength.setdefault(strength, {"table": table, "columns": [column]})
    return by_strength.get("values") or by_strength.get("range")


def is_computational_requirement(value: object) -> bool:
    """Return whether a missing item describes work rather than source data."""
    text = " ".join(str(value or "").split())
    return bool(_COMPUTATIONAL_CUES.search(text))


def _has_valid_table_and_columns(evidence: object) -> bool:
    """True for a well-formed {"table": "...", "columns": ["..."]} evidence dict.

    Mirrors requirement_ledger_blockers' own validity check, reused here to
    decide whether a ledger item's evidence needs repairing from a more
    reliable structured source.

    Args:
        evidence: A ledger item's evidence.

    Returns:
        Whether it names a table and at least one non-empty column.
    """
    if not isinstance(evidence, dict):
        return False
    table = str(evidence.get("table") or "").strip()
    columns = evidence.get("columns")
    return (
        bool(table)
        and isinstance(columns, list)
        and any(str(column or "").strip() for column in columns)
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
            table: ("primary selected source" if index == 0 else "supporting selected source")
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


def _blocking_items(
    ledger: list[dict[str, object]], selected_tables: list[str]
) -> list[dict[str, object]]:
    """Ledger items that are data-bound but lack concrete selected-table evidence."""
    selected = {str(table).strip() for table in selected_tables}
    blocking: list[dict[str, object]] = []
    for item in ledger:
        if item.get("status") == "computational":
            continue
        if item.get("status") == "unresolved":
            blocking.append(item)
            continue
        evidence = item.get("evidence")
        if not isinstance(evidence, dict):
            blocking.append(item)
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
        if (
            table not in selected
            or not isinstance(columns, list)
            or not any(str(column or "").strip() for column in columns)
        ):
            blocking.append(item)
    return blocking


def requirement_ledger_blockers(
    ledger: list[dict[str, object]], selected_tables: list[str]
) -> list[str]:
    """Report data-bound requirements that lack concrete selected-table evidence."""
    return list(
        dict.fromkeys(
            str(item.get("request") or "").strip()
            for item in _blocking_items(ledger, selected_tables)
        )
    )


_LEDGER_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "of",
    "per",
    "requested",
    "the",
    "to",
    "with",
    "measure",
    "dimension",
    "filter",
}
# The kinds of existing ledger items a new request of a kind may merge into.
_COMPATIBLE_KINDS = {
    "dimension": {"dimension", "geographic_filter"},
    "measure": {"measure"},
    "filter": {"filter", "temporal_filter", "geographic_filter"},
    "temporal_filter": {"temporal_filter"},
}


def _ledger_text(value: object) -> str:
    """A requirement as one line: a binding's output, operation and column, or the value itself.

    Args:
        value: A requirement key, a plan or requirements entry, or a dict binding.

    Returns:
        The text, whitespace collapsed; empty when there is none.
    """
    if isinstance(value, dict):
        return " ".join(
            str(part).strip()
            for part in (
                value.get("output") or value.get("alias"),
                value.get("operation") or value.get("type"),
                value.get("column"),
            )
            if part
        ).strip()
    return " ".join(str(value or "").split())


def _ledger_tokens(value: object) -> set[str]:
    """The words of a requirement that tell it apart (three characters or more, no stopwords)."""
    normalized = _ledger_text(value).casefold().replace("boro", "borough")
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 3 and token not in _LEDGER_STOPWORDS
    }


def _add_ledger_item(
    ledger: list[dict[str, object]],
    kind: str,
    request: object,
    status: str,
    evidence: object = None,
) -> None:
    """Add a requirement to ``ledger``, or merge it into an item that already states it.

    An item of a compatible kind states the same requirement when its words
    contain the new ones, are contained in them, or share two of them (every
    output merges). A bound request then binds the item; a computational one
    records its wording as the item's computation.

    Args:
        ledger: The ledger being built, changed in place.
        kind: The kind of requirement ("measure", "temporal_scope", ...).
        request: The requirement as the agent or the question states it.
        status: "bound", "unresolved" or "computational".
        evidence: The table and columns a bound requirement is bound to.
    """
    text = _ledger_text(request)
    if not text:
        return
    new_tokens = _ledger_tokens(text)
    for item in ledger:
        if item["kind"] not in _COMPATIBLE_KINDS.get(kind, {kind}):
            continue
        old_tokens = _ledger_tokens(item["request"])
        # Similar wording does not make two periods or source bindings
        # interchangeable (e.g. "year 2014 trips" and "year 2022 trips").
        if set(re.findall(r"\b\d+\b", text)) != set(re.findall(r"\b\d+\b", item["request"])):
            continue
        if status == "bound" and item["status"] == "bound" and item["evidence"] != evidence:
            continue
        if kind == "output" or (
            new_tokens
            and old_tokens
            and (
                new_tokens <= old_tokens
                or old_tokens <= new_tokens
                or len(new_tokens & old_tokens) >= 2
            )
        ):
            if status == "bound":
                item.update(status="bound", evidence=evidence or [])
            elif status == "computational":
                item["computation"] = text
            return
    ledger.append(
        {
            "kind": kind,
            "request": text,
            "status": status,
            "evidence": evidence if evidence is not None else [],
        }
    )


def _classify_requirement(text: str) -> str:
    """The ledger kind a requirement's wording suggests ("temporal_filter", "measure", ...)."""
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


def _add_plan_bindings(ledger: list[dict[str, object]], plan: dict[str, object]) -> None:
    """Add the semantic plan's filters, dimensions and measures; a measure without columns is computed.

    Args:
        ledger: The ledger being built, changed in place.
        plan: The semantic plan, or an empty dict.
    """
    for group, kind in (
        ("filters", "filter"),
        ("temporal_filters", "temporal_filter"),
        ("dimensions", "dimension"),
        ("measures", "measure"),
    ):
        values = plan.get(group, [])
        for binding in values if isinstance(values, list) else []:
            if isinstance(binding, dict):
                request = (
                    binding.get("requirement") or binding.get("output") or binding.get("column")
                )
                columns = binding.get("columns") or [binding.get("column")]
                if group == "measures" and not any(columns):
                    _add_ledger_item(ledger, "measure", request, "computational")
                else:
                    _add_ledger_item(
                        ledger,
                        kind,
                        request,
                        "bound",
                        {"table": binding.get("table"), "columns": columns},
                    )


def _add_requested_computations(
    ledger: list[dict[str, object]], requirements: dict[str, object]
) -> None:
    """Add the grouping, measures, output columns, ordering and limit the requirements ask for.

    Args:
        ledger: The ledger being built, changed in place.
        requirements: The agent's structured requirements.
    """
    for key, kind in (("grouping", "dimension"), ("measures", "measure")):
        values = requirements.get(key, [])
        for value in values if isinstance(values, list) else []:
            _add_ledger_item(ledger, kind, value, "computational")
    for value in (
        requirements.get("output_columns", [])
        if isinstance(requirements.get("output_columns"), list)
        else []
    ):
        _add_ledger_item(ledger, "output", value, "computational")
    for key, kind in (("ordering", "ordering"), ("limit", "limit")):
        value = requirements.get(key)
        if value not in (None, "", [], "auto"):
            _add_ledger_item(ledger, kind, value, "computational")


def _filter_value_evidences_period(period: str, value: object) -> bool:
    """True if ``value`` (a requirements.filters entry's ``value``) evidences ``period``.

    Either a scalar equals-style match (value == "2016"), or a [start, end]
    range-style temporal filter (e.g. ["1990-01-01", "1990-12-31"] for the
    year 1990), or an isin-style list of candidates. A range's bounds are date
    strings that CONTAIN the bare year, not equal it, so this checks substring
    containment, not just exact equality -- confirmed missing on a second live
    run: the architect correctly bound StartDate to a
    {"operator": "range", "value": ["1990-01-01", "1990-12-31"]} filter for
    "the year 1990", but the original scalar-only equality check
    (`str(value) == period`) never matched a list value at all, so selection
    kept blocking across 8 repeated confirm attempts until the architect gave
    up and returned unparsed text instead of a tool call.

    Args:
        period: A year or year range from the question.
        value: The filter's value: a scalar, or a list or tuple of them.

    Returns:
        Whether any of the values contains ``period``.
    """
    candidates = value if isinstance(value, (list, tuple)) else [value]
    return any(period in str(candidate).strip() for candidate in candidates)


def _bound_columns(
    coverage: dict[str, dict[str, object]],
    filter_entries: list[object],
    plan: dict[str, object],
) -> list[tuple[str, list[str]]]:
    """Every (table, columns) the agent bound, coverage first.

    Args:
        coverage: The agent's requirement_coverage.
        filter_entries: The agent's requirements.filters.
        plan: The semantic plan, or an empty dict.

    Returns:
        The bindings that name a table and at least one column.
    """
    sources: list[object] = [*coverage.values(), *filter_entries]
    for group in ("temporal_filters", "filters", "dimensions", "measures"):
        values = plan.get(group, [])
        sources.extend(values if isinstance(values, list) else [])
    found: list[tuple[str, list[str]]] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        table = str(source.get("table") or "").strip()
        columns = source.get("columns") or [source.get("column")]
        if isinstance(columns, str):
            columns = [columns]
        if not isinstance(columns, list):
            continue
        names = [str(column).strip() for column in columns if str(column or "").strip()]
        if table and names:
            found.append((table, names))
    return found


def _period_evidence(
    period: str,
    coverage: dict[str, dict[str, object]],
    filter_entries: list[object],
    plan: dict[str, object],
    inspections: Mapping[str, str] | None,
) -> object:
    """The evidence that binds a question period: a coverage key, a filter value, or an inspection.

    Args:
        period: A year or year range from the question.
        coverage: The agent's requirement_coverage.
        filter_entries: The agent's requirements.filters.
        plan: The semantic plan, or an empty dict.
        inspections: inspect_columns output by casefolded table name, or None.

    Returns:
        The evidence of the first source that binds the period, or None.
    """
    matching = next(
        (
            evidence
            for request, evidence in coverage.items()
            if period.casefold() in str(request).casefold()
        ),
        None,
    )
    if matching is None:
        matched_filter = next(
            (
                item
                for item in filter_entries
                if isinstance(item, dict)
                and _filter_value_evidences_period(period, item.get("value"))
            ),
            None,
        )
        if matched_filter is not None:
            table = matched_filter.get("table")
            column = matched_filter.get("column")
            if table and column:
                matching = {"table": table, "columns": [column]}
    if matching is None and inspections:
        # Neither wording matched, but the evidence may still be there: a
        # live architect bound Dataset_Da (sampled '2023/05/16 ...') under
        # the key "snapshot date" for "as of 16 May 2023" and was blocked
        # three times over the missing digits.
        matching = _inspected_period_binding(
            period, _bound_columns(coverage, filter_entries, plan), inspections
        )
    return matching


def _add_borough_scope(
    ledger: list[dict[str, object]], question: str, coverage: dict[str, dict[str, object]]
) -> None:
    """Add the NYC boroughs the question names, bound by a coverage key that names one or "borough".

    Args:
        ledger: The ledger being built, changed in place.
        question: The user question.
        coverage: The agent's requirement_coverage.
    """
    boroughs = [
        name
        for name in ("Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island")
        if re.search(rf"\b{re.escape(name)}\b", question, re.IGNORECASE)
    ]
    if boroughs:
        matching = next(
            (
                evidence
                for request, evidence in coverage.items()
                if any(
                    name.casefold() in str(request).casefold()
                    or "borough" in str(request).casefold()
                    for name in boroughs
                )
            ),
            None,
        )
        _add_ledger_item(
            ledger,
            "geographic_scope",
            " and ".join(boroughs),
            "bound" if matching else "unresolved",
            matching if matching else [],
        )


def _repair_measure_evidence(
    ledger: list[dict[str, object]], requirements: dict[str, object]
) -> None:
    """Give a bound measure with malformed evidence the first well-formed requirements.measures entry.

    Args:
        ledger: The ledger being built, changed in place.
        requirements: The agent's structured requirements.
    """
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
    if not isinstance(measure_entries, list):
        return
    for item in ledger:
        if item["kind"] != "measure" or item["status"] != "bound":
            continue
        if _has_valid_table_and_columns(item.get("evidence")):
            continue
        for entry in measure_entries:
            if not isinstance(entry, dict):
                continue
            table = str(entry.get("table") or "").strip()
            columns = entry.get("columns")
            if not isinstance(columns, list):
                columns = [entry.get("column")] if entry.get("column") else []
            columns = [str(c).strip() for c in columns if str(c or "").strip()]
            if table and columns:
                item["evidence"] = {"table": table, "columns": columns}
                break


def _assign_roles(ledger: list[dict[str, object]]) -> None:
    """Mark computed items final, or intermediate when a derived operation consumes them."""
    derived = any(item["kind"] == "derived_operation" for item in ledger)
    for item in ledger:
        if item["status"] != "computational" and "computation" not in item:
            continue
        if item["kind"] in {"dimension", "measure"}:
            item["role"] = "intermediate" if derived else "final"
        elif item["kind"] in {"derived_operation", "output"}:
            item["role"] = "final"


def _ordered_ledger(ledger: list[dict[str, object]]) -> list[dict[str, object]]:
    """Bound items first, then unresolved, then computational; at most ten, three final ones kept.

    Args:
        ledger: The built ledger.

    Returns:
        The ledger in that order, cut to ten items.
    """
    order = {"bound": 0, "unresolved": 1, "computational": 2}
    ordered = sorted(ledger, key=lambda item: order[str(item["status"])])
    if len(ordered) <= 10:
        return ordered
    final_items = [item for item in ordered if item.get("role") == "final"][:3]
    return [
        *[item for item in ordered if item not in final_items][: 10 - len(final_items)],
        *final_items,
    ]


def build_requirement_ledger(
    question: str,
    coverage: dict[str, dict[str, object]],
    requirements: dict[str, object],
    uncovered: list[str],
    semantic_plan: dict[str, object] | None = None,
    inspections: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    """Build the same compact coder contract for divided and unified discovery.

    ``inspections`` maps casefolded table names to their inspect_columns
    output, so a question period can be proven by a bound column's inspected
    values rather than only by how the agent worded its requirement keys.

    Args:
        question: The user question.
        coverage: The agent's requirement_coverage: requirement -> evidence.
        requirements: The agent's structured requirements (measures, filters, grouping, ...).
        uncovered: The requirements the agent listed as uncovered.
        semantic_plan: The semantic plan's filters, dimensions and measures, if any.
        inspections: inspect_columns output by casefolded table name, if any.

    Returns:
        The ledger items, bound first, at most ten.
    """
    ledger: list[dict[str, object]] = []
    for request, raw_evidence in coverage.items():
        evidence = dict(raw_evidence) if isinstance(raw_evidence, dict) else raw_evidence
        _add_ledger_item(ledger, _classify_requirement(str(request)), request, "bound", evidence)
    for request in uncovered:
        if is_computational_requirement(request):
            _add_ledger_item(ledger, "derived_operation", request, "computational")
        else:
            _add_ledger_item(ledger, _classify_requirement(request), request, "unresolved")

    plan = semantic_plan or {}
    _add_plan_bindings(ledger, plan)
    _add_requested_computations(ledger, requirements)

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
    for period in question_periods(question):
        matching = _period_evidence(period, coverage, filter_entries, plan, inspections)
        _add_ledger_item(
            ledger,
            "temporal_scope",
            period,
            "bound" if matching else "unresolved",
            matching if matching else [],
        )
    _add_borough_scope(ledger, question, coverage)
    for marker, label in (
        (r"correlat", "correlation"),
        (r"geographic(?:al)? center", "geographic center"),
        (r"\bratio\b", "ratio"),
    ):
        if re.search(marker, question, re.IGNORECASE):
            _add_ledger_item(ledger, "derived_operation", label, "computational")

    _repair_measure_evidence(ledger, requirements)
    _assign_roles(ledger)
    return _ordered_ledger(ledger)
