"""distinguishing_prefilter (experiments/retrieval_lab/arms_select2.py): drops
same-family siblings that clearly conflict with a year or phrase the question
states. Must never drop a sibling the question actually needs -- in particular
when a question needs TWO members of the same family (two years, or two
different stated phrases), both must survive, not just one.

The year path already accumulates matches across every stated year (set
intersection), so a two-year question was always safe. The phrase path used
to loop over the question's phrases and stop at the first one that narrowed
the pool -- so a question naming two different same-family members by two
different phrases would collapse to just the first phrase's match and
silently drop the second one. Fixed to accumulate like the year path does.
"""
import pytest

from experiments.retrieval_lab import arms_select2


def _patch(monkeypatch, *, families: dict[str, str], coverage: dict[str, str], docs: dict[str, dict]):
    monkeypatch.setattr(arms_select2, "family_of", lambda stem: families.get(stem))
    monkeypatch.setattr(arms_select2, "inspect", lambda stem: coverage.get(stem, ""))
    monkeypatch.setattr(arms_select2, "fetch_docs", lambda stems: {s: docs.get(s, {}) for s in stems})


FAM = "fam1"


def test_year_filter_keeps_both_members_when_question_states_both_years(monkeypatch):
    pool = ["t2020", "t2021", "t2023"]
    _patch(
        monkeypatch,
        families={s: FAM for s in pool},
        coverage={"t2020": "covers 2020", "t2021": "covers 2021", "t2023": "covers 2023"},
        docs={s: {"title": "Boundaries", "description": ""} for s in pool},
    )

    kept, trace = arms_select2.distinguishing_prefilter(
        "Compare the 2021 and 2023 boundary datasets", pool
    )

    assert set(kept) == {"t2021", "t2023"}
    assert "t2020" not in kept
    assert FAM in trace


def test_phrase_filter_keeps_both_members_when_question_states_both_phrases(monkeypatch):
    """The bug this guards: two same-family members, each matching a
    DIFFERENT phrase the question states, must BOTH survive -- not just
    whichever phrase the (old, sequential) loop happened to check first."""
    pool = ["cambridgeshire_tbl", "peterborough_tbl"]
    _patch(
        monkeypatch,
        families={s: FAM for s in pool},
        coverage={s: "" for s in pool},
        docs={
            "cambridgeshire_tbl": {"title": "Children Social Care", "tags": ["Cambridgeshire"], "description": ""},
            "peterborough_tbl": {"title": "Children Social Care", "tags": ["Peterborough"], "description": ""},
        },
    )

    kept, trace = arms_select2.distinguishing_prefilter(
        "Compare open cases in Cambridgeshire versus open cases in Peterborough",
        pool,
    )

    assert set(kept) == {"cambridgeshire_tbl", "peterborough_tbl"}


def test_three_way_family_two_needed_one_conflicting_is_dropped(monkeypatch):
    pool = ["cambridgeshire_tbl", "peterborough_tbl", "norfolk_tbl"]
    _patch(
        monkeypatch,
        families={s: FAM for s in pool},
        coverage={s: "" for s in pool},
        docs={
            "cambridgeshire_tbl": {"title": "Children Social Care", "tags": ["Cambridgeshire"], "description": ""},
            "peterborough_tbl": {"title": "Children Social Care", "tags": ["Peterborough"], "description": ""},
            "norfolk_tbl": {"title": "Children Social Care", "tags": ["Norfolk"], "description": ""},
        },
    )

    kept, trace = arms_select2.distinguishing_prefilter(
        "Compare open cases in Cambridgeshire versus open cases in Peterborough", pool
    )

    assert set(kept) == {"cambridgeshire_tbl", "peterborough_tbl"}
    assert "norfolk_tbl" not in kept


def test_phrase_filter_narrows_to_the_single_stated_member(monkeypatch):
    pool = ["cambridgeshire_tbl", "peterborough_tbl"]
    _patch(
        monkeypatch,
        families={s: FAM for s in pool},
        coverage={s: "" for s in pool},
        docs={
            "cambridgeshire_tbl": {"title": "Children Social Care", "tags": ["Cambridgeshire"], "description": ""},
            "peterborough_tbl": {"title": "Children Social Care", "tags": ["Peterborough"], "description": ""},
        },
    )

    kept, trace = arms_select2.distinguishing_prefilter(
        "How many open cases were there in Peterborough?", pool
    )

    assert kept == ["peterborough_tbl"]


def test_no_signal_leaves_the_family_untouched(monkeypatch):
    pool = ["a_tbl", "b_tbl"]
    _patch(
        monkeypatch,
        families={s: FAM for s in pool},
        coverage={s: "" for s in pool},
        docs={s: {"title": "Boundaries", "description": ""} for s in pool},
    )

    kept, trace = arms_select2.distinguishing_prefilter(
        "How many boundary areas are there in total?", pool
    )

    assert set(kept) == set(pool)
    assert trace == {}


def test_unrelated_singleton_families_pass_through_unchanged(monkeypatch):
    pool = ["only_one", "only_two"]
    _patch(
        monkeypatch,
        families={"only_one": "famA", "only_two": "famB"},
        coverage={s: "" for s in pool},
        docs={s: {"title": "X", "description": ""} for s in pool},
    )

    kept, trace = arms_select2.distinguishing_prefilter("Anything about X in 1999?", pool)

    assert set(kept) == set(pool)
