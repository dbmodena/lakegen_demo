"""Direct port of a real production failure: `confirm_unified_selection` kept
raising "Selection blocked: fundamental data requirements lack concrete
selected-table/column evidence: 2016" even though the architect correctly
bound the year filter to a real column (`requirements.filters` carried
{"table": ..., "column": "Financial Year", "value": 2016}) -- the ledger's
period-binding heuristic only ever checked `requirement_coverage`'s free-text
keys for a literal year substring, never the structured `requirements.filters`
block, so a `requirement_coverage` key that phrased the same binding without
repeating the year digits (e.g. "financial year filter" instead of
"financial year filter (2016)") was wrongly reported as unresolved."""
import pytest

from lakegen.discovery.ledger import build_requirement_ledger, requirement_ledger_blockers


def _temporal_scope_item(ledger):
    return next(item for item in ledger if item["kind"] == "temporal_scope")


def test_period_binds_via_requirement_coverage_substring_match():
    ledger = build_requirement_ledger(
        question="Total amount written off for LS8 5NA in financial year 2016?",
        coverage={
            "financial year filter (2016)": {
                "table": "a.parquet", "columns": ["Financial Year"],
            },
        },
        requirements={},
        uncovered=[],
    )
    item = _temporal_scope_item(ledger)
    assert item["status"] == "bound"
    assert requirement_ledger_blockers(ledger, ["a.parquet"]) == []


def test_period_binds_via_structured_requirements_filters_when_coverage_key_omits_the_year():
    """The exact production failure: requirement_coverage's key phrases the
    binding without repeating the literal year, but requirements.filters
    carries an explicit table/column/value for it -- this must still count
    as bound, not block the selection."""
    ledger = build_requirement_ledger(
        question="Total amount written off for LS8 5NA in financial year 2016?",
        coverage={
            "financial year filter": {  # no "2016" in the free-text key
                "table": "a.parquet", "columns": ["Financial Year"],
            },
        },
        requirements={
            "filters": [
                {"requirement": "financial year = 2016", "table": "a.parquet",
                 "column": "Financial Year", "operator": "equals", "value": 2016},
            ],
        },
        uncovered=[],
    )
    item = _temporal_scope_item(ledger)
    assert item["status"] == "bound"
    assert item["evidence"] == {"table": "a.parquet", "columns": ["Financial Year"]}
    assert requirement_ledger_blockers(ledger, ["a.parquet"]) == []


def test_period_with_no_binding_anywhere_is_still_correctly_unresolved():
    """The heuristic must not become a blanket pass -- a year genuinely
    absent from both coverage and requirements.filters stays unresolved."""
    ledger = build_requirement_ledger(
        question="Total amount written off in 2016?",
        coverage={},
        requirements={"filters": [
            {"requirement": "postcode = LS8 5NA", "table": "a.parquet",
             "column": "Postcode", "operator": "equals", "value": "LS8 5NA"},
        ]},
        uncovered=[],
    )
    item = _temporal_scope_item(ledger)
    assert item["status"] == "unresolved"
    assert requirement_ledger_blockers(ledger, ["a.parquet"]) == ["2016"]


def test_requirements_filters_value_is_compared_as_string_regardless_of_type():
    """requirements.filters' value may be an int (2016) or a str ("2016")
    depending on how the architect emitted it; both must match."""
    ledger = build_requirement_ledger(
        question="Total for 2016?",
        coverage={},
        requirements={"filters": [
            {"table": "a.parquet", "column": "Year", "value": "2016"},
        ]},
        uncovered=[],
    )
    item = _temporal_scope_item(ledger)
    assert item["status"] == "bound"


def test_malformed_filters_entries_are_ignored_not_fatal():
    ledger = build_requirement_ledger(
        question="Total for 2016?",
        coverage={},
        requirements={"filters": ["not a dict", {"value": 2016}]},  # missing table/column
        uncovered=[],
    )
    item = _temporal_scope_item(ledger)
    assert item["status"] == "unresolved"


def test_period_binds_via_a_range_style_temporal_filter():
    """A second real production failure, on a different question: the
    architect correctly bound StartDate to a range filter for "the year
    1990" ({"operator": "range", "value": ["1990-01-01", "1990-12-31"]}),
    but the range's bounds are date strings that CONTAIN the bare year, not
    equal it -- the original scalar-only equality check never matched a
    list value at all, so `confirm_unified_selection` blocked 8 times in a
    row (each attempt re-supplying the same correct, well-evidenced filter)
    until the architect gave up and returned raw text instead of calling
    the tool."""
    ledger = build_requirement_ledger(
        question="How many sediment habitat observations in 1990?",
        coverage={
            "year filter": {  # no "1990" in the free-text key
                "table": "a.parquet", "columns": ["StartDate"],
            },
        },
        requirements={
            "filters": [
                {"requirement": "year filter", "table": "a.parquet",
                 "column": "StartDate", "operator": "range",
                 "value": ["1990-01-01", "1990-12-31"]},
            ],
        },
        uncovered=[],
    )
    item = _temporal_scope_item(ledger)
    assert item["status"] == "bound"
    assert item["evidence"] == {"table": "a.parquet", "columns": ["StartDate"]}
    assert requirement_ledger_blockers(ledger, ["a.parquet"]) == []


def test_range_filter_for_a_different_year_does_not_falsely_bind():
    """The substring-containment check must stay period-specific -- a range
    for 1991 must not be mistaken as evidence for 1990."""
    ledger = build_requirement_ledger(
        question="How many observations in 1990?",
        coverage={},
        requirements={"filters": [
            {"table": "a.parquet", "column": "StartDate", "operator": "range",
             "value": ["1991-01-01", "1991-12-31"]},
        ]},
        uncovered=[],
    )
    item = _temporal_scope_item(ledger)
    assert item["status"] == "unresolved"


def _measure_item(ledger):
    return next(item for item in ledger if item["kind"] == "measure")


_BASE_COVERAGE = {
    "survey identification": {"table": "a.parquet", "columns": ["SurveyName"]},
}
_BASE_FILTERS = [
    {"requirement": "survey name", "table": "a.parquet", "column": "SurveyName",
     "operator": "contains", "value": "1990 Wimpey Environmental"},
]
_VALID_MEASURE = [{
    "output": "sediment_observation_count", "operation": "count_rows",
    "table": "a.parquet", "columns": ["RecordKey"],
}]


@pytest.mark.parametrize("malformed_columns", ["RecordKey", [], None])
def test_measure_repairs_from_requirements_measures_when_coverage_columns_malformed(malformed_columns):
    """A third real production failure, on the same question: the
    architect's requirement_coverage["observation count"] evidence had a
    malformed `columns` (a bare string, an empty list, or None -- all
    confirmed-plausible LLM formatting slips), while requirements.measures
    carried the SAME measure correctly bound with real table/columns.
    Selection blocked on "observation count" even though a reliable,
    structured binding existed right there in requirements.measures."""
    ledger = build_requirement_ledger(
        question="How many sediment habitat observations in the 1990 Wimpey Environmental Survey?",
        coverage={**_BASE_COVERAGE, "observation count": {"table": "a.parquet", "columns": malformed_columns}},
        requirements={"filters": _BASE_FILTERS, "measures": _VALID_MEASURE},
        uncovered=[],
    )
    item = _measure_item(ledger)
    assert item["status"] == "bound"
    assert item["evidence"] == {"table": "a.parquet", "columns": ["RecordKey"]}
    assert requirement_ledger_blockers(ledger, ["a.parquet"]) == []


def test_measure_repaired_when_coverage_evidence_has_no_table():
    """The malformed coverage evidence has NO table at all -- still repaired
    from requirements.measures, which carries the real table."""
    ledger = build_requirement_ledger(
        question="How many sediment habitat observations in the 1990 Wimpey Environmental Survey?",
        coverage={**_BASE_COVERAGE, "observation count": {"table": "", "columns": ["RecordKey"]}},
        requirements={"filters": _BASE_FILTERS, "measures": _VALID_MEASURE},
        uncovered=[],
    )
    item = _measure_item(ledger)
    assert item["status"] == "bound"
    assert requirement_ledger_blockers(ledger, ["a.parquet"]) == []


def test_measure_stays_blocked_when_no_repairable_measures_entry_exists():
    """The repair must not become a blanket pass -- malformed coverage
    evidence with no matching requirements.measures entry to repair from
    stays correctly blocked."""
    ledger = build_requirement_ledger(
        question="How many sediment habitat observations in the 1990 Wimpey Environmental Survey?",
        coverage={**_BASE_COVERAGE, "observation count": {"table": "a.parquet", "columns": []}},
        requirements={"filters": _BASE_FILTERS, "measures": []},
        uncovered=[],
    )
    assert requirement_ledger_blockers(ledger, ["a.parquet"]) == ["observation count"]


def test_measure_repair_does_not_touch_an_already_validly_bound_measure():
    """A measure whose coverage evidence is already well-formed must be
    left untouched -- the repair only fires for malformed evidence."""
    ledger = build_requirement_ledger(
        question="How many sediment habitat observations in the 1990 Wimpey Environmental Survey?",
        coverage={**_BASE_COVERAGE, "observation count": {"table": "a.parquet", "columns": ["RecordKey"]}},
        requirements={"filters": _BASE_FILTERS, "measures": [{
            "output": "other", "operation": "count_rows", "table": "b.parquet", "columns": ["OtherKey"],
        }]},
        uncovered=[],
    )
    item = _measure_item(ledger)
    assert item["evidence"] == {"table": "a.parquet", "columns": ["RecordKey"]}
