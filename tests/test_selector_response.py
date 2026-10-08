"""The judge's answer, parsed: a payload however it nests, a rejection, prose, and the fallbacks."""

from __future__ import annotations

import json

from lakegen.discovery.candidates import parse_table_selector_response

FILES = ["a.parquet", "b.parquet", "c.parquet"]


def test_a_payload_with_a_nested_plan_keeps_its_tables_and_reasoning():
    payload = {"tables": "a.parquet, c.parquet", "reasoning": "a and c cover it", "selection_plan": {"table_roles": {"a.parquet": "primary"}, "requirements": {}}}
    selected, reasoning = parse_table_selector_response("FINAL_PAYLOAD: " + json.dumps(payload) + "\nDone.", FILES, FILES)
    assert selected == ["a.parquet", "c.parquet"] and reasoning == "a and c cover it"


def test_a_payload_naming_no_known_table_falls_back_to_the_top_two():
    selected, reasoning = parse_table_selector_response(_payload(tables="", reasoning="nothing inspected"), FILES, FILES)
    assert selected == ["a.parquet", "b.parquet"] and reasoning == "nothing inspected"
    selected, _ = parse_table_selector_response(_payload(tables=["b.parquet", "zzz"], reasoning="list form"), FILES, FILES)
    assert selected == ["b.parquet"]


def test_escaped_quotes_and_a_broken_payload_are_tolerated():
    escaped = 'FINAL_PAYLOAD: {\\"tables\\": \\"b.parquet\\", \\"reasoning\\": \\"escaped\\"}'
    assert parse_table_selector_response(escaped, FILES, FILES) == (["b.parquet"], "escaped")
    broken = 'FINAL_PAYLOAD: {"tables": "b.parquet", "reasoning": '
    selected, reasoning = parse_table_selector_response(broken, FILES, FILES)
    assert selected == ["a.parquet", "b.parquet"] and reasoning == "No reasoning provided."


def test_a_rejection_and_prose_keep_their_old_readings():
    assert parse_table_selector_response("REJECT_KEYWORDS: too broad", FILES, FILES) == ([], "REJECT_KEYWORDS: too broad")
    assert parse_table_selector_response("TABLES: c.parquet", FILES, FILES) == (["c.parquet"], "No reasoning provided.")


def _payload(**fields) -> str:
    return "FINAL_PAYLOAD: " + json.dumps(fields)
