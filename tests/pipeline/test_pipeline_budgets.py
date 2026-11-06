"""Attempts are bounded, repeated rejections are caught, and rejections carry tables forward."""

from __future__ import annotations

from lakegen.pipeline.run.budgets import Attempts, DedupeGuard, selection_plan_signature
from lakegen.pipeline.run.state import RunState


def test_attempts_count_from_zero_and_mark_the_last():
    attempts = list(Attempts(3))
    assert [a.index for a in attempts] == [0, 1, 2] and [a.number for a in attempts] == [1, 2, 3]
    assert [a.last for a in attempts] == [False, False, True]
    assert [a.last for a in Attempts(1)] == [True]


def test_the_guard_recalls_a_rejected_selection_whatever_its_order_or_case():
    guard = DedupeGuard()
    assert guard.prior_rejection(["a.csv"]) is None and guard.prior_rejection([]) is None
    guard.reject_selection(
        ["B.csv", "a.csv"],
        {"category": "join_missed", "missing_requirements": ["school id"]},
        "no join",
    )
    prior = guard.prior_rejection(["a.csv", "b.csv"])
    assert prior["tables"] == ["a.csv", "b.csv"] and prior["category"] == "join_missed"
    assert prior["missing_requirements"] == ["school id"]
    assert guard.prior_rejection(["a.csv"]) is None
    guard.reject_selection(["a.csv", "b.csv"], None, "again")
    assert (
        guard.prior_rejection(["a.csv", "b.csv"])["category"] == "tables_rejected"
    )  # the latest wins
    assert guard.rejected_keys == {("a.csv", "b.csv")}


def test_the_guard_recalls_a_refused_plan_but_never_for_an_empty_selection():
    guard = DedupeGuard()
    signature = selection_plan_signature(["a.csv"], {"coder_brief": {"filters": ["year"]}})
    assert guard.repeated_plan(["a.csv"], signature) is None
    guard.reject_plan(signature, {"retry_reason": "selection_contract_invalid"})
    assert guard.repeated_plan(["a.csv"], signature) == {
        "retry_reason": "selection_contract_invalid"
    }
    assert guard.repeated_plan([], selection_plan_signature([], {})) is None


def test_carry_excludes_the_skipped_and_keeps_the_useful_with_their_metadata():
    state = RunState(solr_meta={"a.csv": {"title": "a"}, "b.csv": {"title": "b"}})
    state.carry(keep_tables=["a.csv", "c.csv"], skip_tables=["B.csv"])
    assert state.excluded_tables == {"b.csv"}
    assert state.carried_tables == ["a.csv", "c.csv"] and state.carried_metadata == {
        "a.csv": {"title": "a"}
    }
    # A table kept earlier that a later round skips leaves.
    state.carry(keep_tables=["a.csv"], skip_tables=["a.csv"])
    assert state.carried_tables == ["c.csv"] and "a.csv" in state.excluded_tables
