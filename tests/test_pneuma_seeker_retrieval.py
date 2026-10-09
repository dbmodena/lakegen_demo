"""The Pneuma-Seeker modality: content search and table enumeration.

The paper's retriever is "Pneuma + content search + table enumeration". These
tests pin each half separately, and pin that the plain ``pneuma`` mode is
unchanged by their presence.
"""

import pandas as pd
import pytest

from lakegen.config.experiment import RetrievalExperimentConfig
from lakegen.retrieval import (
    PneumaRetriever,
    RetrievalConfig,
    RetrievalMode,
    TableRetrievalService,
)
from lakegen.retrieval.intent import intent_entities
from lakegen.retrieval.pneuma.enumeration import _enumeration_pattern
from lakegen.retrieval.pneuma.seeker import _entity_pattern


class FakePneuma:
    """The stub used by the existing pneuma tests, kept identical in shape."""

    def __init__(self, table_ids):
        self.table_ids = list(table_ids)
        self.calls = []

    def query_index(self, index_name, queries, *, k, n, alpha):
        self.calls.append((index_name, queries, k, n, alpha))
        return {
            "status": "SUCCESS",
            "data": [{"query": queries, "retrieved_tables": self.table_ids}],
        }


def _table(directory, stem, frame):
    path = directory / f"{stem}.parquet"
    pd.DataFrame(frame).to_parquet(path)
    return str(path)


def _resolver(directory):
    """Resolve a path or stem to a document, as the Solr resolver does."""

    def resolve(table_id):
        from pathlib import Path

        stem = Path(str(table_id)).stem
        if not (directory / f"{stem}.parquet").is_file():
            return None
        return {"resource_id": stem, "dataset_id": stem, "title": stem}

    return resolve


def _seeker(directory, **overrides):
    config = RetrievalConfig(mode=RetrievalMode.PNEUMA_SEEKER, **overrides)
    return config


def test_content_search_finds_a_table_pneuma_never_returned(tmp_path):
    """The recall case: the entity is only in a table outside Pneuma's list."""
    _table(tmp_path, "harbour_traffic", {"note": ["unrelated"]})
    _table(tmp_path, "landing_log", {"place": ["ferry terminal", "ferry dock"]})
    client = FakePneuma([str(tmp_path / "harbour_traffic.parquet")])

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=client, table_dir=tmp_path
    ).retrieve("where do boats dock?", top_k=5, entities=["ferry"])

    ids = [hit.document["dataset_id"] for hit in hits]
    assert "landing_log" in ids
    found = next(hit for hit in hits if hit.document["dataset_id"] == "landing_log")
    # Content contributed; Pneuma never saw it.
    assert found.lexical_score and found.semantic_rank is None
    assert found.document["pneuma_evidence"]["term_cells"] == {"ferry": 2}


def test_a_column_name_match_scores_without_any_cell_match(tmp_path):
    """The paper scores matches in column names and cell values alike.

    Compared within the content branch, with Pneuma returning nothing, so the
    claim under test is only that a column name is evidence at all.
    """
    _table(tmp_path, "by_borough", {"borough": ["Queens"], "n": [1]})
    _table(tmp_path, "unrelated", {"note": ["nothing"]})

    hits = PneumaRetriever(
        _seeker(tmp_path),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("counts per borough", top_k=5, entities=["borough"])

    top = hits[0]
    assert top.document["dataset_id"] == "by_borough"
    assert top.document["pneuma_evidence"]["column_name_terms"] == ["borough"]
    # No cell says "borough"; the column name alone carried it.
    assert top.document["pneuma_evidence"]["term_cells"] == {}


def test_per_keyword_normalization_stops_a_frequent_entity_dominating(tmp_path):
    """Normalizing per keyword is what makes a rare entity count at all.

    Without it the table repeating a common year 5,000 times outscores the one
    that matches *both* entities, and the ranking is decided by raw volume.
    """
    _table(tmp_path, "year_dump", {"year": ["2020"] * 5000})
    _table(tmp_path, "both_terms", {"year": ["2020"], "site": ["newtown creek"]})
    client = FakePneuma([])

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=client, table_dir=tmp_path
    ).retrieve(
        "newtown creek in 2020", top_k=5, entities=["2020", "newtown creek"]
    )

    assert [hit.document["dataset_id"] for hit in hits][0] == "both_terms"


def test_log_damping_compresses_raw_match_volume(tmp_path):
    """``log(1+tf)``: a 500x difference in raw counts is not a 500x difference
    in score. The bigger table still wins -- the paper damps volume, it does not
    normalize by row count the way the grep modality does -- but by a margin
    that a handful of extra matches can no longer buy."""
    _table(tmp_path, "huge", {"t": ["ferry"] * 1000})
    _table(tmp_path, "focused", {"t": ["ferry"], "u": ["ferry"]})
    client = FakePneuma([])

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=client, table_dir=tmp_path
    ).retrieve("ferry", top_k=5, entities=["ferry"])

    scores = {hit.document["dataset_id"]: hit.score for hit in hits}
    cells = {
        hit.document["dataset_id"]: hit.document["pneuma_evidence"]["term_cells"]
        for hit in hits
    }
    assert cells["huge"]["ferry"] / cells["focused"]["ferry"] == 500
    assert scores["huge"] / scores["focused"] < 10


def test_entities_are_used_as_listed_and_none_means_no_content_search(
    tmp_path, monkeypatch
):
    """The reference implementation boosts only with entities: the question's own
    words are never scanned for."""
    from lakegen.retrieval.pneuma import seeker as pneuma_module

    _table(tmp_path, "rivers", {"place": ["East River crossing"]})
    retriever = PneumaRetriever(
        _seeker(tmp_path, pneuma_enumerate_tables=False),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
    )

    hits = retriever.retrieve("crossings", top_k=5, entities=["East River"])
    assert hits[0].document["pneuma_evidence"]["term_cells"] == {"East River": 1}

    scanned = []
    original = pneuma_module.scan_many

    def counting(paths, terms, **kwargs):
        scanned.extend(paths)
        return original(paths, terms, **kwargs)

    monkeypatch.setattr(pneuma_module, "scan_many", counting)
    assert retriever.retrieve("East River crossings", top_k=5, entities=[]) == []
    assert retriever.retrieve("East River crossings", top_k=5) == []
    assert scanned == []


def test_without_entities_the_ranking_is_pneumas_own(tmp_path):
    _table(tmp_path, "first", {"note": ["ferry"]})
    _table(tmp_path, "second", {"note": ["ferry", "ferry"]})
    client = FakePneuma(
        [str(tmp_path / "first.parquet"), str(tmp_path / "second.parquet")]
    )

    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_enumerate_tables=False),
        _resolver(tmp_path),
        client=client,
        table_dir=tmp_path,
    ).retrieve("ferry", top_k=5)

    assert [hit.document["dataset_id"] for hit in hits] == ["first", "second"]
    assert all(
        hit.document["pneuma_evidence"]["content_weight"] == 0.0 for hit in hits
    )


def test_entities_match_whole_words_not_substrings(tmp_path):
    """The reference pattern: bounded on both sides, one character between parts."""
    _table(
        tmp_path, "agencies",
        {"name": ["Department of Education"], "code": ["120201"]},
    )
    _table(tmp_path, "places", {"city": ["NEW-YORK"], "period": ["FY 2020"]})

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("anything", top_k=5, entities=["art", "2020", "New York"])

    assert [hit.document["dataset_id"] for hit in hits] == ["places"]
    assert hits[0].document["pneuma_evidence"]["term_cells"] == {
        "2020": 1, "New York": 1,
    }


def test_entities_cannot_inject_a_regular_expression(tmp_path):
    """Only an entity's letters and digits reach the pattern."""
    _table(tmp_path, "formulas", {"expr": ["a+b", "ab", "aaab"]})
    retriever = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=FakePneuma([]),
        table_dir=tmp_path,
    )

    # No letter or digit, no pattern: ".*" searches for nothing, not everything.
    assert _entity_pattern(".*") == ""
    assert retriever.retrieve("formulas", top_k=5, entities=[".*"]) == []
    # "+" stands for one separator character; as a quantifier it would match
    # "ab" and "aaab" instead.
    hits = retriever.retrieve("formulas", top_k=5, entities=["a+b"])
    assert hits[0].document["pneuma_evidence"]["term_cells"] == {"a+b": 1}


def test_partial_coverage_is_penalized_by_the_matched_fraction(tmp_path):
    """mean x coverage: one of two entities at full strength scores 0.5 x 0.5."""
    _table(tmp_path, "one_entity", {"site": ["newtown creek"]})
    _table(tmp_path, "both", {"site": ["newtown creek"], "period": ["2020"]})

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("anything", top_k=5, entities=["newtown creek", "2020"])

    content = {
        hit.document["dataset_id"]: hit.document["pneuma_evidence"]["content_score"]
        for hit in hits
    }
    assert content == {"both": pytest.approx(1.0), "one_entity": pytest.approx(0.25)}


def test_match_weights_come_from_the_config(tmp_path):
    """raw = table_name_weight x title + column_name_weight x column names
    + cell_weight x cells, then log1p. The titles here match nothing."""
    _table(tmp_path, "named", {"borough": ["x"]})
    _table(tmp_path, "cells", {"note": ["borough", "borough", "borough"]})

    def top(**weights):
        hits = PneumaRetriever(
            _seeker(tmp_path, **weights), _resolver(tmp_path),
            client=FakePneuma([]), table_dir=tmp_path,
        ).retrieve("anything", top_k=5, entities=["borough"])
        return hits[0].document["dataset_id"]

    # Upstream's 2:1 makes a column name worth two cells, so three cells win.
    assert top() == "cells"
    assert top(pneuma_column_name_weight=4.0) == "named"
    assert top(pneuma_cell_weight=0.0) == "named"


def test_match_weights_are_validated_and_reach_the_runtime_config():
    with pytest.raises(ValueError, match="pneuma_cell_weight must not be negative"):
        RetrievalConfig(mode="pneuma_seeker", pneuma_cell_weight=-1.0)
    with pytest.raises(ValueError, match="must not all be zero"):
        RetrievalConfig(
            mode="pneuma_seeker",
            pneuma_table_name_weight=0.0,
            pneuma_column_name_weight=0.0,
            pneuma_cell_weight=0.0,
        )
    runtime = RetrievalExperimentConfig(
        mode="pneuma_seeker",
        pneuma_table_name_weight=4.0,
        pneuma_column_name_weight=3.0,
        pneuma_cell_weight=0.5,
    ).to_runtime()
    assert (
        runtime.pneuma_table_name_weight,
        runtime.pneuma_column_name_weight,
        runtime.pneuma_cell_weight,
    ) == (4.0, 3.0, 0.5)


def test_a_catalog_title_match_scores_as_the_table_name(tmp_path):
    """Files are named by opaque id, so a table's name is its catalog title."""
    from pathlib import Path

    _table(tmp_path, "43nn-pn8j", {"note": ["nothing"]})
    _table(tmp_path, "99xx-ab1c", {"note": ["nothing"]})
    titles = {
        "43nn-pn8j": "Staten Island Ferry Ridership",
        "99xx-ab1c": "Street Tree Census",
    }

    def resolver(table_id):
        stem = Path(str(table_id)).stem
        if stem not in titles:
            return None
        return {"resource_id": stem, "dataset_id": stem, "title": titles[stem]}

    def run(**weights):
        return PneumaRetriever(
            _seeker(tmp_path, pneuma_enumerate_tables=False, **weights),
            resolver,
            client=FakePneuma([]),
            table_dir=tmp_path,
        ).retrieve("anything", top_k=5, entities=["Ferry"])

    hits = run()
    assert [hit.document["dataset_id"] for hit in hits] == ["43nn-pn8j"]
    evidence = hits[0].document["pneuma_evidence"]
    assert evidence["title_terms"] == ["Ferry"]
    assert evidence["term_cells"] == {} and evidence["column_name_terms"] == []
    assert hits[0].document["description"] == "Contains the searched entities: Ferry."
    # Weighted 0, a title is no evidence at all.
    assert run(pneuma_table_name_weight=0.0) == []


def test_the_content_scan_reads_past_the_grep_row_and_column_caps(tmp_path):
    """The reference implementation reads every row of every column."""
    frame = {f"c{index:02d}": ["filler"] * 60 for index in range(5)}
    frame["c05"] = ["filler"] * 59 + ["Ferry"]
    _table(tmp_path, "wide_and_long", frame)

    hits = PneumaRetriever(
        _seeker(tmp_path, grep_max_columns_per_file=2, grep_max_scan_rows_per_file=10),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("anything", top_k=5, entities=["ferry"])

    assert hits[0].document["pneuma_evidence"]["term_cells"] == {"ferry": 1}


def test_matched_entities_are_shown_in_the_description(tmp_path):
    _table(tmp_path, "boats", {"kind": ["ferry"]})
    catalog = {
        "resource_id": "boats", "dataset_id": "boats", "title": "Boats",
        "description": "Harbour boats.",
    }

    hits = PneumaRetriever(
        _seeker(tmp_path), lambda _id: catalog, client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("anything", top_k=1, entities=["ferry", "tugboat"])

    assert hits[0].document["description"] == (
        "Harbour boats. Contains the searched entities: ferry."
    )
    assert catalog["description"] == "Harbour boats."


def test_fusion_weight_boundaries(tmp_path):
    """w=0 is plain Pneuma order; w=1 is pure content order."""
    _table(tmp_path, "pneuma_favourite", {"note": ["nothing relevant"]})
    _table(tmp_path, "content_favourite", {"note": ["ferry ferry ferry"]})
    ids = [str(tmp_path / "pneuma_favourite.parquet")]

    def run(weight):
        return [
            hit.document["dataset_id"]
            for hit in PneumaRetriever(
                _seeker(tmp_path, pneuma_content_weight=weight),
                _resolver(tmp_path),
                client=FakePneuma(ids),
                table_dir=tmp_path,
            ).retrieve("ferry", top_k=5, entities=["ferry"])
        ]

    assert run(0.0)[0] == "pneuma_favourite"
    assert run(1.0)[0] == "content_favourite"


def test_a_pneuma_hit_with_no_local_file_keeps_its_rank_contribution(tmp_path):
    """A table in the index but not on disk must not vanish from the ranking."""
    _table(tmp_path, "local", {"note": ["ferry"]})
    client = FakePneuma([str(tmp_path / "local.parquet")])
    resolve = _resolver(tmp_path)

    def resolver(table_id):
        if str(table_id).endswith("remote_only.parquet"):
            return {"resource_id": "remote_only", "title": "Remote"}
        return resolve(table_id)

    client.table_ids = [
        str(tmp_path / "remote_only.parquet"),
        str(tmp_path / "local.parquet"),
    ]
    hits = PneumaRetriever(
        _seeker(tmp_path), resolver, client=client, table_dir=tmp_path
    ).retrieve("ferry", top_k=5, entities=["ferry"])

    remote = next(hit for hit in hits if hit.document["resource_id"] == "remote_only")
    assert remote.semantic_rank == 1 and remote.lexical_score is None


def test_enumeration_pulls_in_the_sibling_years(tmp_path):
    """Top-k cannot retrieve a family; the paper's enumeration is what does."""
    for year in (2019, 2020, 2021):
        _table(tmp_path, f"water_body_testing_{year}", {"ph": [7.0]})
    client = FakePneuma([str(tmp_path / "water_body_testing_2020.parquet")])

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=client, table_dir=tmp_path
    ).retrieve("water body testing", top_k=10, entities=["water body"])

    ids = [hit.document["dataset_id"] for hit in hits]
    assert set(ids) == {
        "water_body_testing_2019", "water_body_testing_2020", "water_body_testing_2021",
    }
    sibling = next(hit for hit in hits if hit.document["dataset_id"].endswith("2021"))
    assert (
        sibling.document["pneuma_evidence"]["enumerated_from"]
        == "water_body_testing_2020"
    )


def test_enumeration_is_disabled_by_config(tmp_path):
    """The paper's middle ablation arm: Pneuma + content search, no enumeration."""
    for year in (2019, 2020, 2021):
        _table(tmp_path, f"water_body_testing_{year}", {"ph": [7.0]})
    client = FakePneuma([str(tmp_path / "water_body_testing_2020.parquet")])

    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_enumerate_tables=False),
        _resolver(tmp_path),
        client=client,
        table_dir=tmp_path,
    ).retrieve("water body testing", top_k=10, entities=["zzzznotpresent"])

    assert [hit.document["dataset_id"] for hit in hits] == ["water_body_testing_2020"]


def test_an_enumerated_sibling_never_outranks_its_seed(tmp_path):
    for year in (2019, 2020):
        _table(tmp_path, f"water_body_testing_{year}", {"ph": [7.0]})
    client = FakePneuma([str(tmp_path / "water_body_testing_2020.parquet")])

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=client, table_dir=tmp_path
    ).retrieve("water body testing", top_k=10, entities=["water body"])

    assert hits[0].document["dataset_id"] == "water_body_testing_2020"


def test_enumeration_ignores_opaque_socrata_identifiers(tmp_path):
    """The NYC lake is 2,600 files named like 43nn-pn8j; none are a family.

    Stripping the digits first would join 'pn' and 'j' into a three-letter
    "word" and invent a family out of unrelated datasets, so the word test runs
    on the original stem.
    """
    for stem in ("43nn-pn8j", "99nn-pn8j"):
        _table(tmp_path, stem, {"note": ["ferry"]})
    client = FakePneuma([str(tmp_path / "43nn-pn8j.parquet")])

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=client, table_dir=tmp_path
    ).retrieve("anything", top_k=10, entities=["zzzznotpresent"])

    assert [hit.document["dataset_id"] for hit in hits] == ["43nn-pn8j"]
    assert _enumeration_pattern("43nn-pn8j") is None
    assert _enumeration_pattern("h9gi-nx95") is None
    assert _enumeration_pattern("restaurant_inspections") is None
    assert _enumeration_pattern("water_body_testing_2020") == (
        "water_body_testing_" + r"\d{4}"
    )


def test_enumeration_discards_an_over_general_pattern(tmp_path):
    """A pattern matching half the lake is a naming convention, not a series."""
    for index in range(40):
        _table(tmp_path, f"reading_{index:03d}", {"v": [index]})
    client = FakePneuma([str(tmp_path / "reading_000.parquet")])

    hits = PneumaRetriever(
        _seeker(tmp_path), _resolver(tmp_path), client=client, table_dir=tmp_path
    ).retrieve("readings", top_k=10, entities=["zzzznotpresent"])

    assert [hit.document["dataset_id"] for hit in hits] == ["reading_000"]


def test_plain_pneuma_is_untouched_by_the_new_mode(tmp_path):
    """The base arm of the experiment must not move."""
    client = FakePneuma(["/tables/b.csv", "/tables/a.csv"])
    documents = {
        "/tables/a.csv": {"resource_id": "a", "title": "A"},
        "/tables/b.csv": {"resource_id": "b", "title": "B"},
    }
    config = RetrievalConfig(
        mode="pneuma", top_k=2, alpha=0.25, candidate_multiplier=7,
        pneuma_index_name="test-index",
    )

    hits = PneumaRetriever(config, documents.get, client=client).retrieve(
        "Which tables?", top_k=2
    )

    assert [hit.document["resource_id"] for hit in hits] == ["b", "a"]
    assert [hit.score for hit in hits] == [1.0, 0.5]
    # k is the caller's top_k, not the widened Seeker pool.
    assert client.calls == [("test-index", "Which tables?", 2, 7, 0.25)]
    assert "pneuma_evidence" not in hits[0].document


def test_resolver_documents_are_never_mutated(tmp_path):
    """The resolver hands out shared dicts; evidence must not leak between queries."""
    _table(tmp_path, "boats", {"note": ["ferry"]})
    catalog = {"resource_id": "boats", "dataset_id": "boats", "title": "Boats"}
    client = FakePneuma([str(tmp_path / "boats.parquet")])

    PneumaRetriever(
        _seeker(tmp_path), lambda _id: catalog, client=client, table_dir=tmp_path
    ).retrieve("ferry", top_k=1, entities=["ferry"])

    assert "pneuma_evidence" not in catalog


def test_the_content_scan_is_bounded_by_the_file_cap(tmp_path, monkeypatch):
    from lakegen.retrieval.pneuma import seeker as pneuma_module

    for index in range(12):
        _table(tmp_path, f"t{index:02d}", {"note": ["ferry"]})
    calls = []
    original = pneuma_module.scan_many

    def counting(paths, terms, **kwargs):
        calls.extend(paths)
        return original(paths, terms, **kwargs)

    monkeypatch.setattr(pneuma_module, "scan_many", counting)
    PneumaRetriever(
        _seeker(tmp_path, grep_max_files=3),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("ferry", top_k=5, entities=["ferry"])

    assert len(calls) <= 3


def test_service_requires_a_table_dir_only_for_the_seeker(tmp_path):
    with pytest.raises(
        ValueError, match="pneuma_seeker retrieval requires a local table_dir"
    ):
        TableRetrievalService(object(), RetrievalConfig(mode="pneuma_seeker"))


def test_run_record_logs_the_seeker_mode(tmp_path):
    _table(tmp_path, "boats", {"kind": ["ferry"]})
    runs = []
    service = TableRetrievalService(
        object(),
        _seeker(tmp_path),
        observer=runs.append,
        table_dir=str(tmp_path),
        pneuma_document_resolver=_resolver(tmp_path),
        pneuma_client=FakePneuma([]),
    )
    hits = service.retrieve(
        question="ferry routes", keywords=["ferry"], top_k=1, entities=["ferry"]
    )

    assert runs[0].mode == "pneuma_seeker"
    assert runs[0].status == "succeeded"
    assert hits[0].document["dataset_id"] == "boats"


def test_intent_entities_reads_the_phase1_envelope():
    import json

    payload = {
        "status": "resolved", "concepts": ["ferries"], "entities": ["East River"],
        "measures": [], "filters": [], "time_constraints": [], "group_by": [],
        "order_by": [], "limit": None, "join_requirements": [], "missing_evidence": [],
    }
    assert intent_entities("RETRIEVAL_INTENT: " + json.dumps(payload)) == ["East River"]
    # An unusable envelope yields no entities, so no content search runs.
    assert intent_entities("not an envelope") == []
    unresolved = {**payload, "status": "unresolved", "missing_evidence": ["subject"]}
    assert intent_entities("RETRIEVAL_INTENT: " + json.dumps(unresolved)) == []


def test_content_weight_is_validated():
    for weight in (-0.1, 1.5):
        with pytest.raises(ValueError, match="pneuma_content_weight"):
            RetrievalConfig(mode="pneuma_seeker", pneuma_content_weight=weight)


def test_per_keyword_normalization_sees_every_file_by_default(tmp_path):
    """The content pool is the whole lake, so the per-keyword maximum is the real one.

    Normalizing divides by the strongest table for that entity. With a budget the
    strongest table can sit outside the pool, which silently rescales every other
    table's score -- the same defect a truncated IDF population causes in grep.
    """
    for index in range(12):
        _table(tmp_path, f"ds{index:02d}", {"note": ["filler"]})
    _table(tmp_path, "weak", {"note": ["ferry"]})
    # The strongest table for this entity sorts last, past any small budget.
    _table(tmp_path, "zz_strongest", {"note": ["ferry"] * 50})
    client = FakePneuma([])

    def content_of(stem, **overrides):
        hits = PneumaRetriever(
            _seeker(tmp_path, **overrides), _resolver(tmp_path),
            client=client, table_dir=tmp_path,
        ).retrieve("ferry", top_k=10, entities=["ferry"])
        found = {h.document["dataset_id"]: h for h in hits}
        return found[stem].document["pneuma_evidence"] if stem in found else None

    full = content_of("weak")
    assert full["truncated"] is False
    assert full["scanned_files"] == full["listed_files"] == 14
    # zz_strongest is the per-keyword maximum, so "weak" normalizes well below 1.
    assert 0.0 < full["normalized_content_score"] < 0.5


def test_zero_content_weight_never_scans_and_equals_plain_pneuma(tmp_path, monkeypatch):
    """The same rule as grep: a family weighted 0 must not do its work.

    It also makes the boundary exact -- with no content contribution the mode
    should return precisely what plain `pneuma` returns, not pneuma's results
    padded with zero-scored tables nobody can rank.
    """
    from lakegen.retrieval.pneuma import seeker as pneuma_module

    _table(tmp_path, "pneuma_hit", {"note": ["nothing"]})
    _table(tmp_path, "content_hit", {"note": ["ferry ferry"]})
    ids = [str(tmp_path / "pneuma_hit.parquet")]

    scanned = []
    original = pneuma_module.scan_many

    def counting(paths, terms, **kwargs):
        scanned.extend(paths)
        return original(paths, terms, **kwargs)

    monkeypatch.setattr(pneuma_module, "scan_many", counting)
    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_content_weight=0.0, pneuma_enumerate_tables=False),
        _resolver(tmp_path),
        client=FakePneuma(ids),
        table_dir=tmp_path,
    ).retrieve("ferry", top_k=5, entities=["ferry"])

    assert scanned == []
    assert [hit.document["dataset_id"] for hit in hits] == ["pneuma_hit"]


# ------------------------------------------------ the reference configuration


class FakeExtractor:
    def __init__(self, entities=(), error=None):
        self.result = list(entities)
        self.error = error
        self.questions = []

    def entities(self, question):
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return list(self.result)


def _ids(hits):
    return [hit.document["dataset_id"] for hit in hits]


def test_fill_keeps_pneumas_order_and_fills_only_leftover_slots(tmp_path):
    """The reference merge: content can add tables but never reorder Pneuma's."""
    _table(tmp_path, "pneuma_first", {"note": ["nothing relevant"]})
    _table(tmp_path, "pneuma_second", {"note": ["ferry"]})
    _table(tmp_path, "content_only", {"note": ["ferry", "ferry", "ferry"]})
    _table(tmp_path, "unmatched", {"note": ["nothing"]})
    ids = [str(tmp_path / f"{stem}.parquet") for stem in ("pneuma_first", "pneuma_second")]

    def run(**overrides):
        return PneumaRetriever(
            _seeker(tmp_path, pneuma_enumerate_tables=False, **overrides),
            _resolver(tmp_path),
            client=FakePneuma(ids),
            table_dir=tmp_path,
        ).retrieve("ferry", top_k=5, entities=["ferry"])

    filled = run(pneuma_content_fusion="fill")
    # A table matching no entity never fills a slot, unlike the reference's padding.
    assert _ids(filled) == ["pneuma_first", "pneuma_second", "content_only"]
    assert [hit.score for hit in filled] == [1.0, 0.5, 1.0 / 3]
    evidence = filled[2].document["pneuma_evidence"]
    assert evidence["content_fusion"] == "fill"
    assert evidence["content_weight"] is None
    assert evidence["matched_terms"] == ["ferry"]
    # The weighted sum at full content weight reorders instead.
    assert run(pneuma_content_weight=1.0)[0].document["dataset_id"] == "content_only"


def test_fill_asks_pneuma_for_exactly_top_k(tmp_path):
    """The reference queries k, not a candidate_multiplier-wide pool."""
    _table(tmp_path, "boats", {"kind": ["ferry"]})
    for fusion, expected_k in (("fill", 3), ("weighted", 15)):
        client = FakePneuma([])
        PneumaRetriever(
            _seeker(tmp_path, pneuma_content_fusion=fusion, candidate_multiplier=5),
            _resolver(tmp_path),
            client=client,
            table_dir=tmp_path,
        ).retrieve("ferry", top_k=3, entities=["ferry"])
        assert client.calls[0][2] == expected_k


def test_fill_leaves_no_slot_when_pneuma_fills_top_k(tmp_path):
    for stem in ("a", "b"):
        _table(tmp_path, stem, {"note": ["nothing"]})
    _table(tmp_path, "content_only", {"note": ["ferry"]})
    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_content_fusion="fill", pneuma_enumerate_tables=False),
        _resolver(tmp_path),
        client=FakePneuma([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")]),
        table_dir=tmp_path,
    ).retrieve("ferry", top_k=2, entities=["ferry"])

    assert _ids(hits) == ["a", "b"]


def test_fill_puts_enumerated_siblings_after_content_matches(tmp_path):
    for year in (2019, 2020):
        _table(tmp_path, f"water_body_testing_{year}", {"ph": [7.0]})
    _table(tmp_path, "lake_samples", {"site": ["Lough Neagh"]})
    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_content_fusion="fill"),
        _resolver(tmp_path),
        client=FakePneuma([str(tmp_path / "water_body_testing_2020.parquet")]),
        table_dir=tmp_path,
    ).retrieve("water quality", top_k=10, entities=["Lough Neagh"])

    assert _ids(hits) == [
        "water_body_testing_2020", "lake_samples", "water_body_testing_2019",
    ]


def test_extracted_entities_drive_content_search_when_the_caller_gives_none(tmp_path):
    _table(tmp_path, "landing_log", {"place": ["ferry terminal"]})
    extractor = FakeExtractor(["ferry"])
    retriever = PneumaRetriever(
        _seeker(tmp_path, pneuma_entity_model="fake-model"),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
        entity_extractor=extractor,
    )

    hits = retriever.retrieve("where do ferries dock?", top_k=5)
    assert _ids(hits) == ["landing_log"]
    extraction = hits[0].document["pneuma_evidence"]["entity_extraction"]
    assert extraction["model"] == "fake-model"
    assert extraction["entities"] == ["ferry"]
    assert extraction["error"] is None
    assert extractor.questions == ["where do ferries dock?"]

    # An agent's entities -- even none -- are used as given, with no extraction.
    assert retriever.retrieve("where do ferries dock?", top_k=5, entities=[]) == []
    hits = retriever.retrieve("q", top_k=5, entities=["terminal"])
    assert hits[0].document["pneuma_evidence"]["entity_extraction"] is None
    assert extractor.questions == ["where do ferries dock?"]


def test_a_failed_extraction_falls_back_to_pneumas_order(tmp_path):
    _table(tmp_path, "pneuma_hit", {"note": ["nothing"]})
    _table(tmp_path, "content_hit", {"note": ["ferry"]})
    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_entity_model="fake-model"),
        _resolver(tmp_path),
        client=FakePneuma([str(tmp_path / "pneuma_hit.parquet")]),
        table_dir=tmp_path,
        entity_extractor=FakeExtractor(error=RuntimeError("service down")),
    ).retrieve("ferry", top_k=5)

    assert _ids(hits) == ["pneuma_hit"]
    extraction = hits[0].document["pneuma_evidence"]["entity_extraction"]
    assert extraction["entities"] == []
    assert extraction["error"] == "RuntimeError: service down"


def test_no_entity_model_means_no_extraction(tmp_path):
    _table(tmp_path, "boats", {"kind": ["ferry"]})
    extractor = FakeExtractor(["ferry"])
    retriever = PneumaRetriever(
        _seeker(tmp_path),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
        entity_extractor=extractor,
    )
    assert retriever.entity_extractor is None
    assert retriever.retrieve("ferry", top_k=5) == []
    assert extractor.questions == []


def test_the_service_passes_the_extractor_through(tmp_path):
    _table(tmp_path, "boats", {"kind": ["ferry"]})
    service = TableRetrievalService(
        object(),
        _seeker(tmp_path, pneuma_entity_model="fake-model", rerank_model=None),
        table_dir=str(tmp_path),
        pneuma_document_resolver=_resolver(tmp_path),
        pneuma_client=FakePneuma([]),
        pneuma_entity_extractor=FakeExtractor(["ferry"]),
    )
    hits = service.retrieve(question="ferry routes", keywords=["routes"], top_k=1)
    assert _ids(hits) == ["boats"]


def test_parse_entities_keeps_strings_from_the_json_object():
    from lakegen.retrieval.pneuma.entities import _parse_entities

    assert _parse_entities('{"entities": ["Belfast", "2007", 3, " "]}') == [
        "Belfast", "2007",
    ]
    assert _parse_entities('Sure: {"entities": []}') == []
    with pytest.raises(ValueError):
        _parse_entities('{"names": ["Belfast"]}')


def test_fusion_and_entity_model_settings_are_validated_and_reach_the_runtime(
    monkeypatch,
):
    with pytest.raises(ValueError):
        RetrievalConfig(mode="pneuma_seeker", pneuma_content_fusion="average")
    assert RetrievalConfig(pneuma_entity_model="  ").pneuma_entity_model is None

    experiment = RetrievalExperimentConfig(
        mode="pneuma_seeker",
        pneuma_content_fusion="fill",
        pneuma_entity_model="openai.gpt-oss-120b",
    )
    runtime = experiment.to_runtime()
    assert runtime.pneuma_content_fusion == "fill"
    assert runtime.pneuma_entity_model == "openai.gpt-oss-120b"
    assert RetrievalExperimentConfig.from_runtime(runtime) == experiment

    monkeypatch.setenv("LAKEGEN_PNEUMA_CONTENT_FUSION", "fill")
    monkeypatch.setenv("LAKEGEN_PNEUMA_ENTITY_MODEL", "openai.gpt-oss-20b")
    from_env = RetrievalConfig.from_env()
    assert from_env.pneuma_content_fusion == "fill"
    assert from_env.pneuma_entity_model == "openai.gpt-oss-20b"
    monkeypatch.setenv("LAKEGEN_PNEUMA_ENTITY_MODEL", "")
    assert RetrievalConfig.from_env().pneuma_entity_model is None
    monkeypatch.delenv("LAKEGEN_PNEUMA_CONTENT_FUSION")
    assert RetrievalConfig.from_env().pneuma_content_fusion == "weighted"


def test_only_text_columns_are_searched_as_in_the_reference(tmp_path):
    """Numbers are not VARCHAR, so the reference never reads them, cells or names."""
    _table(tmp_path, "numbers", {"year_2019": [2019, 2019], "other": [1, 2]})
    _table(tmp_path, "text", {"period": ["2019"]})

    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_enumerate_tables=False),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("anything", top_k=5, entities=["2019"])

    assert [hit.document["dataset_id"] for hit in hits] == ["text"]


def test_every_occurrence_counts_as_in_the_reference(tmp_path):
    _table(tmp_path, "twice", {"note": ["ferry and ferry"]})

    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_enumerate_tables=False),
        _resolver(tmp_path),
        client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("anything", top_k=5, entities=["ferry"])

    assert hits[0].document["pneuma_evidence"]["term_cells"] == {"ferry": 2}


def test_a_title_match_counts_once_per_text_column(tmp_path):
    """The reference tests the table name on every text column's row."""
    from pathlib import Path

    _table(tmp_path, "wide", {"a": ["x"], "b": ["y"], "c": ["z"], "n": [1]})
    _table(tmp_path, "narrow", {"a": ["x"], "b": ["ferry"]})

    def resolver(table_id):
        stem = Path(str(table_id)).stem
        title = {"wide": "Ferry routes", "narrow": "Other"}.get(stem)
        if title is None:
            return None
        return {"resource_id": stem, "dataset_id": stem, "title": title}

    hits = PneumaRetriever(
        _seeker(tmp_path, pneuma_enumerate_tables=False),
        resolver,
        client=FakePneuma([]),
        table_dir=tmp_path,
    ).retrieve("anything", top_k=5, entities=["ferry"])

    # log1p(3 x 3 text columns) for the title beats log1p(1 x 1) for one cell.
    content = {
        hit.document["dataset_id"]: hit.document["pneuma_evidence"]["content_score"]
        for hit in hits
    }
    import math

    assert content["wide"] == pytest.approx(1.0)
    assert content["narrow"] == pytest.approx(math.log1p(1) / math.log1p(9))
