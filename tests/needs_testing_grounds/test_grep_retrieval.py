import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from lakegen.retrieval import grep as grep_module

from lakegen.retrieval.duckdb_agentic import _query_terms
from lakegen.retrieval.grep import _term_pattern
from lakegen.retrieval import (
    GrepRetriever,
    RetrievalConfig,
    RetrievalMode,
    TableRetrievalService,
)


def test_grep_retrieval_reports_per_column_evidence_and_samples(tmp_path):
    pd.DataFrame({
        "school_name": ["Galileo", "Newton", "Curie"],
        "connection_type": ["fiber", "dsl", "fiber"],
        "mbps": [1000, 20, 500],
    }).to_parquet(tmp_path / "school_connectivity.parquet")
    pd.DataFrame({"street": ["Main"], "trees": [4]}).to_parquet(
        tmp_path / "urban_trees.parquet"
    )

    retriever = GrepRetriever(
        RetrievalConfig(mode="grep", grep_sample_rows=1), tmp_path
    )
    hits = retriever.retrieve(
        "Which schools have fiber connectivity?", ["fiber"], top_k=5
    )

    assert [hit.document["resource_id"] for hit in hits] == [
        "school_connectivity.parquet"
    ]
    evidence = hits[0].document["grep_evidence"]
    assert evidence["term_counts"]["fiber"] == 2
    assert evidence["term_columns"]["fiber"] == {"connection_type": 2}
    assert len(evidence["samples"]) == 1
    assert hits[0].rank == hits[0].lexical_rank == 1


def test_non_string_columns_are_searched(tmp_path):
    """The DuckDB modality searches varchar only; grep casts every column."""
    pd.DataFrame({"permit": ["A1", "B2"], "issued_year": [2019, 2024]}).to_parquet(
        tmp_path / "permits.parquet"
    )
    pd.DataFrame({"note": ["unrelated"]}).to_parquet(tmp_path / "other.parquet")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "permits issued in 2024", ["2024"], top_k=5
    )

    assert hits[0].document["dataset_id"] == "permits"
    assert hits[0].document["grep_evidence"]["term_columns"]["2024"] == {
        "issued_year": 1
    }


def test_files_with_unrelated_schemas_are_each_searched(tmp_path):
    """A lake of independent tables has no common schema to unify."""
    pd.DataFrame({"city": ["Bologna"], "note": ["metro station"]}).to_parquet(
        tmp_path / "ds1.parquet"
    )
    pd.DataFrame({
        "borough": ["Queens"], "complaint": ["metro noise"], "year": [2024],
    }).to_parquet(tmp_path / "ds2.parquet")
    pd.DataFrame({"tree_id": [7], "species": ["oak"]}).to_parquet(
        tmp_path / "ds3.parquet"
    )

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "metro", ["metro"], top_k=5
    )

    assert {hit.document["dataset_id"] for hit in hits} == {"ds1", "ds2"}


def test_ranking_uses_density_not_raw_match_volume(tmp_path):
    pd.DataFrame({
        "t": ["metro" if index % 1000 == 0 else "filler" for index in range(50_000)]
    }).to_parquet(tmp_path / "huge_sparse.parquet")
    pd.DataFrame({"t": ["metro"] * 5}).to_parquet(tmp_path / "tiny_focused.parquet")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "metro", ["metro"], top_k=5
    )

    assert [hit.document["dataset_id"] for hit in hits] == [
        "tiny_focused",
        "huge_sparse",
    ]
    counts = {
        hit.document["dataset_id"]: hit.document["grep_evidence"]["term_counts"]["metro"]
        for hit in hits
    }
    # The lower-ranked file still has ten times as many raw matches.
    assert counts["huge_sparse"] == 50 and counts["tiny_focused"] == 5


def test_idf_downweights_terms_present_in_every_table(tmp_path):
    for name in ("a", "b", "c"):
        pd.DataFrame({
            "borough": ["Queens"], "topic": ["parking" if name == "a" else "other"],
        }).to_parquet(tmp_path / f"{name}.parquet")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "parking in a borough", ["parking", "borough"], top_k=5
    )

    # "borough" is in all three tables and cannot discriminate; "parking" is in one.
    assert hits[0].document["dataset_id"] == "a"


def test_uncastable_columns_are_skipped_without_losing_the_file(tmp_path):
    pl.DataFrame({
        "tags": [["alpha", "beta"]],
        "label": ["ferry terminal"],
    }).write_parquet(tmp_path / "nested.parquet")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "ferry", ["ferry"], top_k=5
    )

    assert hits[0].document["dataset_id"] == "nested"
    evidence = hits[0].document["grep_evidence"]
    assert evidence["uncastable_columns"] == 1
    assert evidence["scanned_columns"] == ["label"]


def test_struct_columns_are_searched(tmp_path):
    """Structs render as text, so a match nested inside one still counts."""
    pl.DataFrame({"payload": [{"id": 1, "kind": "ferry"}]}).write_parquet(
        tmp_path / "structs.parquet"
    )

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "ferry", ["ferry"], top_k=5
    )

    evidence = hits[0].document["grep_evidence"]
    assert evidence["term_columns"]["ferry"] == {"payload": 1}
    assert evidence["uncastable_columns"] == 0


def test_probe_rescues_a_value_only_match_cut_by_the_prefilter(tmp_path):
    """The prefilter sees names and metadata only; values need a prefix read."""
    pd.DataFrame({"col": ["unrelated"]}).to_parquet(tmp_path / "aaa.parquet")
    pd.DataFrame({"col": ["ferry"]}).to_parquet(tmp_path / "zzz.parquet")

    config = RetrievalConfig(mode="grep", grep_max_files=1, grep_probe_files=5)
    retriever = GrepRetriever(config, tmp_path)

    # Neither filename, schema, nor metadata mentions the term, so the prefilter
    # cannot separate the two and orders them by name: zzz falls past the cut.
    ranked = [entry.path.name for entry in retriever._candidates(["ferry"])]
    assert ranked.index("zzz.parquet") >= config.grep_max_files

    hits = retriever.retrieve("ferry routes", ["ferry"], top_k=5)
    assert [hit.document["dataset_id"] for hit in hits] == ["zzz"]


def test_concepts_never_reach_the_engine_as_patterns(tmp_path):
    """Two layers keep an agent from supplying a regex: tokenizing, then escaping."""
    # The shared tokenizer keeps alphanumeric runs only, so metacharacters are
    # already gone by the time a term exists...
    assert _query_terms("match .* everywhere", [".*"]) == (["match", "everywhere"], [])
    # ...and any term that does survive is escaped rather than compiled as syntax.
    assert _term_pattern("a+b") == "(?i)a\\+b"

    pd.DataFrame({"expr": ["a+b", "ab"]}).to_parquet(tmp_path / "formulas.parquet")
    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "formula table", [".*", "a+b"], top_k=5
    )
    assert [hit.document["dataset_id"] for hit in hits] == ["formulas"]


def test_scan_row_cap_is_respected(tmp_path):
    pd.DataFrame({
        "t": ["ferry"] * 10 + ["filler"] * 90,
    }).to_parquet(tmp_path / "capped.parquet")

    hits = GrepRetriever(
        RetrievalConfig(mode="grep", grep_max_scan_rows_per_file=5), tmp_path
    ).retrieve("ferry", ["ferry"], top_k=1)

    evidence = hits[0].document["grep_evidence"]
    assert evidence["term_counts"]["ferry"] == 5
    assert evidence["rows_scanned"] == 5 and evidence["total_rows"] == 100


def test_no_match_returns_no_hits(tmp_path):
    pd.DataFrame({"t": ["unrelated"]}).to_parquet(tmp_path / "x.parquet")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "zzzznotpresent", ["zzzznotpresent"], top_k=5
    )

    assert hits == []


def test_service_requires_table_dir_and_does_not_call_solr(tmp_path):
    config = RetrievalConfig(mode=RetrievalMode.GREP)
    with pytest.raises(ValueError, match="grep retrieval requires a local table_dir"):
        TableRetrievalService(object(), config)

    pd.DataFrame({"incident_kind": ["collision"]}).to_parquet(
        tmp_path / "road_events.parquet"
    )
    service = TableRetrievalService(object(), config, table_dir=str(tmp_path))
    hits = service.retrieve(
        question="road collision incidents", keywords=["collision"], top_k=1
    )
    assert hits[0].document["dataset_id"] == "road_events"


def test_scans_do_not_use_the_streaming_engine():
    """Guard against reintroducing a crash that takes the whole app down.

    A lake-wide scan issues thousands of small collects. Under polars 1.40
    ``engine="streaming"`` segfaults intermittently at that volume (4 of 5 runs
    over the 2,673-file NYC lake), killing the hosting process rather than
    raising. The row cap and one-column-at-a-time reads already bound memory --
    the streaming engine measured the same peak RSS -- so there is nothing to
    win back by restoring it. A segfault cannot be caught, so this is asserted
    at the source level.
    """
    from lakegen import cell_scan

    for module in (grep_module, cell_scan):  # the queries themselves live in cell_scan
        source = Path(module.__file__).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        _, _, body = code.partition('"""')
        _, _, body = body.partition('"""')  # drop the module docstring, which names it
        assert 'engine="streaming"' not in body, module.__name__


def test_catalog_title_outweighs_an_incidental_column_name(tmp_path):
    """Socrata ids and column codes are opaque; the title carries the meaning.

    The on-topic table is named ``43nn-pn8j`` with columns like ``CAMIS``/``BORO``
    and matches only in its title. The distractor matches in more places -- its
    description, its catalogued column list, and a real column name -- so a flat
    per-field bonus ranks the distractor first and buries the table the question
    is actually about.
    """
    tables = tmp_path / "datasets" / "parquet"
    tables.mkdir(parents=True)
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()

    pd.DataFrame({"CAMIS": [1], "BORO": ["Brooklyn"]}).to_parquet(
        tables / "43nn-pn8j.parquet"
    )
    pd.DataFrame({"restaurant_inspection_visits": [3]}).to_parquet(
        tables / "9zzz-zzzz.parquet"
    )

    (meta_dir / "metadata.json").write_text(json.dumps([
        # Both terms in the title only.
        {"resource": {"id": "43nn-pn8j",
                      "name": "DOHMH New York City Restaurant Inspection Results",
                      "description": "Citations issued during visits.",
                      "columns_name": ["CAMIS", "BORO"]},
         "classification": {"domain_tags": [], "domain_category": "Health"}},
        # Both terms in description + catalogued columns + a real column name.
        {"resource": {"id": "9zzz-zzzz", "name": "Citywide Mobility Survey",
                      "description": "How often respondents visit a restaurant, "
                                     "and inspection of travel diaries.",
                      "columns_name": ["restaurant_inspection_visits"]},
         "classification": {"domain_tags": [], "domain_category": "Transportation"}},
    ]), encoding="utf-8")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tables).retrieve(
        "restaurant inspections", ["restaurant", "inspection"], top_k=5
    )

    assert [hit.document["dataset_id"] for hit in hits] == ["43nn-pn8j", "9zzz-zzzz"]


def test_local_file_modes_are_flagged_for_table_dir():
    assert RetrievalMode.GREP.requires_table_dir
    assert RetrievalMode.DUCKDB_AGENTIC.requires_table_dir
    # Only the Seeker augmentation scans cells; plain pneuma stays service-only.
    assert RetrievalMode.PNEUMA_SEEKER.requires_table_dir
    assert not RetrievalMode.KEYWORD.requires_table_dir
    assert not RetrievalMode.SEMANTIC.requires_table_dir
    assert not RetrievalMode.HYBRID.requires_table_dir
    assert not RetrievalMode.PNEUMA.requires_table_dir


def test_run_record_logs_grep_mode(tmp_path):
    pd.DataFrame({"kind": ["ferry"]}).to_parquet(tmp_path / "boats.parquet")
    runs = []
    service = TableRetrievalService(
        object(),
        RetrievalConfig(mode=RetrievalMode.GREP),
        observer=runs.append,
        table_dir=str(tmp_path),
    )
    service.retrieve(question="ferry routes", keywords=["ferry"], top_k=1)

    assert runs[0].mode == "grep"
    assert runs[0].status == "succeeded"
    assert runs[0].hits[0].document["dataset_id"] == "boats"


def _values_config(**overrides) -> RetrievalConfig:
    return RetrievalConfig(mode=RetrievalMode.GREP_VALUES, **overrides)


def test_values_only_ignores_a_match_that_is_not_in_any_cell(tmp_path):
    """Filename, column name, and catalog title must all stop counting."""
    tables = tmp_path / "datasets" / "parquet"
    tables.mkdir(parents=True)
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()

    # Every non-value signal points at this file: its name, a column name, and
    # its catalogued title and description all say "ferry".
    pd.DataFrame({"ferry_route": ["red"], "count": [3]}).to_parquet(
        tables / "ferry_counts.parquet"
    )
    # ...while only this file's cells do.
    pd.DataFrame({"a1b2": ["ferry terminal"]}).to_parquet(tables / "9zzz-zzzz.parquet")

    (meta_dir / "metadata.json").write_text(json.dumps([
        {"resource": {"id": "ferry_counts", "name": "Ferry ridership",
                      "description": "Daily ferry counts.",
                      "columns_name": ["ferry_route", "count"]},
         "classification": {"domain_tags": ["ferry"], "domain_category": "Transport"}},
    ]), encoding="utf-8")

    hits = GrepRetriever(_values_config(), tables).retrieve(
        "ferry ridership", ["ferry"], top_k=5
    )

    assert [hit.document["dataset_id"] for hit in hits] == ["9zzz-zzzz"]
    evidence = hits[0].document["grep_evidence"]
    assert evidence["values_only"] is True
    assert evidence["term_columns"]["ferry"] == {"a1b2": 1}
    assert evidence["filename_terms"] == []
    assert evidence["schema_terms"] == []
    assert evidence["metadata_terms"] == []


def test_the_default_modality_still_ranks_on_every_signal(tmp_path):
    """The same lake, the same question: only the modality differs."""
    tables = tmp_path / "datasets" / "parquet"
    tables.mkdir(parents=True)
    pd.DataFrame({"ferry_route": ["red"]}).to_parquet(tables / "ferry_counts.parquet")
    pd.DataFrame({"a1b2": ["ferry terminal"]}).to_parquet(tables / "9zzz-zzzz.parquet")

    ranked = {
        mode: [
            hit.document["dataset_id"]
            for hit in GrepRetriever(RetrievalConfig(mode=mode), tables).retrieve(
                "ferry ridership", ["ferry"], top_k=5
            )
        ]
        for mode in (RetrievalMode.GREP, RetrievalMode.GREP_VALUES)
    }

    # grep sees the name and the column and ranks that file first; grep_values
    # cannot see either, so the file is not a hit at all.
    assert ranked[RetrievalMode.GREP][0] == "ferry_counts"
    assert ranked[RetrievalMode.GREP_VALUES] == ["9zzz-zzzz"]


def test_values_only_still_carries_the_catalog_title_for_display(tmp_path):
    """Ranking ignores the catalog; the hit still reads like every other hit.

    Only the ranking is meant to differ between the two modalities. If a hit
    came back titled by its opaque filename, a downstream agent would be
    choosing from worse cards rather than from a different ranking.
    """
    tables = tmp_path / "datasets" / "parquet"
    tables.mkdir(parents=True)
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    pd.DataFrame({"a1b2": ["ferry terminal"]}).to_parquet(tables / "43nn-pn8j.parquet")
    (meta_dir / "metadata.json").write_text(json.dumps([
        {"resource": {"id": "43nn-pn8j", "name": "Harbor Landings",
                      "description": "Dock arrivals.", "columns_name": ["a1b2"]},
         "classification": {"domain_tags": ["boats"], "domain_category": "Transport"}},
    ]), encoding="utf-8")

    hit = GrepRetriever(_values_config(), tables).retrieve(
        "ferry", ["ferry"], top_k=1
    )[0]

    assert hit.document["title"] == "Harbor Landings"
    assert hit.document["tags"] == ["boats", "Transport"]
    assert hit.document["description"].startswith("Dock arrivals.")
    # Read for display only: the catalog contributed nothing to the ranking.
    assert hit.document["grep_evidence"]["metadata_terms"] == []


def test_values_only_ranking_still_uses_density_and_idf(tmp_path):
    pd.DataFrame({
        "t": ["metro" if index % 1000 == 0 else "filler" for index in range(50_000)]
    }).to_parquet(tmp_path / "huge_sparse.parquet")
    pd.DataFrame({"t": ["metro"] * 5}).to_parquet(tmp_path / "tiny_focused.parquet")

    hits = GrepRetriever(_values_config(), tmp_path).retrieve(
        "metro", ["metro"], top_k=5
    )

    assert [hit.document["dataset_id"] for hit in hits] == [
        "tiny_focused",
        "huge_sparse",
    ]


def test_a_lake_under_the_file_cap_is_scanned_whole_without_probing(tmp_path, monkeypatch):
    """No prefilter can cut a small lake, so the prefix read is pure cost."""
    pd.DataFrame({"col": ["unrelated"] * 5}).to_parquet(tmp_path / "aaa.parquet")
    # The match sits past the probe prefix, so a probe-gated path would lose it.
    pd.DataFrame({"col": ["filler"] * 5 + ["ferry"]}).to_parquet(tmp_path / "zzz.parquet")

    retriever = GrepRetriever(
        _values_config(grep_max_files=2, grep_probe_rows_per_file=2), tmp_path
    )
    monkeypatch.setattr(
        retriever, "_probe_counts", lambda *_a: pytest.fail("probed a lake under the cap")
    )
    hits = retriever.retrieve("ferry routes", ["ferry"], top_k=5)

    assert [hit.document["dataset_id"] for hit in hits] == ["zzz"]


def test_over_the_file_cap_values_decide_which_files_are_scanned(tmp_path):
    """Past the cap there is no free signal left, so the probe orders the lake."""
    for name in ("aaa", "bbb", "ccc"):
        pd.DataFrame({"col": ["unrelated"]}).to_parquet(tmp_path / f"{name}.parquet")
    pd.DataFrame({"col": ["ferry"]}).to_parquet(tmp_path / "zzz.parquet")

    retriever = GrepRetriever(_values_config(grep_max_files=1), tmp_path)

    # Nothing outside the cells distinguishes the four files: the candidate list
    # is in name order, and zzz sorts last, well past a max_files=1 cut.
    assert [entry.path.stem for entry in retriever._candidates(["ferry"])][0] == "aaa"

    hits = retriever.retrieve("ferry routes", ["ferry"], top_k=5)
    assert [hit.document["dataset_id"] for hit in hits] == ["zzz"]


def test_values_only_mode_is_flagged_for_table_dir_and_as_grep(tmp_path):
    assert RetrievalMode.GREP_VALUES.requires_table_dir
    assert RetrievalMode.GREP_VALUES.is_grep and RetrievalMode.GREP.is_grep
    assert not RetrievalMode.DUCKDB_AGENTIC.is_grep

    with pytest.raises(
        ValueError, match="grep_values retrieval requires a local table_dir"
    ):
        TableRetrievalService(object(), _values_config())


def test_run_record_logs_the_values_only_mode(tmp_path):
    pd.DataFrame({"kind": ["ferry"]}).to_parquet(tmp_path / "boats.parquet")
    runs = []
    service = TableRetrievalService(
        object(), _values_config(), observer=runs.append, table_dir=str(tmp_path)
    )
    hits = service.retrieve(question="ferry routes", keywords=["ferry"], top_k=1)

    assert runs[0].mode == "grep_values"
    assert runs[0].status == "succeeded"
    assert hits[0].document["dataset_id"] == "boats"


def _lake_of(tmp_path, count, term_in_last):
    """A lake bigger than any cap under test, with the term only at the end."""
    for index in range(count):
        value = term_in_last if index == count - 1 else "filler"
        pd.DataFrame({"note": [value]}).to_parquet(tmp_path / f"ds{index:03d}.parquet")


def test_the_whole_lake_is_scanned_by_default(tmp_path):
    """No cap by default: a match in the alphabetically-last file is still found.

    The old default scanned the 250 files the metadata prefilter ranked highest,
    which for a term with no metadata signal anywhere was just the files that
    sorted first by name.
    """
    _lake_of(tmp_path, 30, "ferry")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "ferry", ["ferry"], top_k=5
    )

    assert [hit.document["dataset_id"] for hit in hits] == ["ds029"]
    evidence = hits[0].document["grep_evidence"]
    assert evidence["files_in_lake"] == evidence["files_scanned"] == 30
    assert evidence["truncated"] is False


def test_an_explicit_cap_still_truncates_and_says_so(tmp_path):
    """A cap remains available, but a capped run must be self-identifying."""
    _lake_of(tmp_path, 30, "ferry")

    hits = GrepRetriever(
        RetrievalConfig(mode="grep", grep_max_files=5, grep_probe_files=1), tmp_path
    ).retrieve("ferry", ["ferry"], top_k=5)

    # The term sits past both the cut and the promotion window, so it is lost --
    # which is exactly why a cap is no longer the default.
    assert hits == []


def test_parallel_and_sequential_scans_agree_exactly(tmp_path):
    """The map is parallel; the ranking it feeds must not depend on that."""
    for index in range(12):
        pd.DataFrame({
            "note": ["ferry"] * (index + 1) + ["filler"] * 5,
        }).to_parquet(tmp_path / f"ds{index:02d}.parquet")

    def run(workers):
        hits = GrepRetriever(
            RetrievalConfig(mode="grep", scan_workers=workers), tmp_path
        ).retrieve("ferry terminals", ["ferry"], top_k=10)
        return [(h.document["dataset_id"], h.rank, h.score) for h in hits]

    assert run(1) == run(4) == run(8)


def test_a_cap_distorts_idf_which_is_why_it_is_not_the_default(tmp_path):
    """Pins the bug the unlimited default now avoids.

    IDF is computed across the files that were evaluated, so truncating the scan
    changes every term's document frequency -- and with it the scores, and even
    the winner, among the tables that did survive. A table's rank must not depend
    on how many other files happened to be read.
    """
    # "ferry" in 3 of 12 files and "dock" in the other 9: a cut that keeps only
    # the first few files changes both document frequencies.
    for index in range(12):
        term = "ferry" if index < 3 else "dock"
        pd.DataFrame({"note": [term]}).to_parquet(tmp_path / f"ds{index:02d}.parquet")

    def top(**overrides):
        config = RetrievalConfig(mode="grep", **overrides)
        hit = GrepRetriever(config, tmp_path).retrieve(
            "ferry dock", ["ferry", "dock"], top_k=1
        )[0]
        return hit.document["dataset_id"], round(hit.score, 4)

    full = top()
    capped = top(grep_max_files=4, grep_probe_files=1)
    assert full[0] != capped[0] and full[1] != capped[1]


def _weighted(tmp_path, monkeypatch, **overrides):
    """Retrieve while counting how many files the cell scanner actually read."""
    scanned = []
    original = grep_module.scan_many

    def counting(paths, terms, **kwargs):
        scanned.extend(paths)
        return original(paths, terms, **kwargs)

    monkeypatch.setattr(grep_module, "scan_many", counting)
    hits = GrepRetriever(RetrievalConfig(mode="grep", **overrides), tmp_path).retrieve(
        "ferry", ["ferry"], top_k=5
    )
    return hits, scanned


def _mixed_lake(tmp_path):
    """One table findable only by its cells, one only by its column name."""
    pd.DataFrame({"note": ["ferry"]}).to_parquet(tmp_path / "cells_only.parquet")
    pd.DataFrame({"ferry_id": [1]}).to_parquet(tmp_path / "name_only.parquet")


def test_zero_value_weight_never_reads_a_cell(tmp_path, monkeypatch):
    """Weighted 0, the cell scan cannot change a ranking, so it must not run."""
    _mixed_lake(tmp_path)

    hits, scanned = _weighted(tmp_path, monkeypatch, grep_value_weight=0.0)

    assert scanned == []
    # Only the metadata family remains, so the value-only table is not a hit.
    assert [hit.document["dataset_id"] for hit in hits] == ["name_only"]
    evidence = hits[0].document["grep_evidence"]
    assert evidence["value_weight"] == 0.0 and evidence["metadata_weight"] == 1.0


def test_zero_metadata_weight_never_matches_a_name_or_catalog_field(tmp_path, monkeypatch):
    """The mirror case: no filename, column name, or catalog field is matched."""
    _mixed_lake(tmp_path)

    hits, scanned = _weighted(tmp_path, monkeypatch, grep_metadata_weight=0.0)

    assert scanned  # the cell scan is the only family left, so it does run
    assert [hit.document["dataset_id"] for hit in hits] == ["cells_only"]
    hit = hits[0].document["grep_evidence"]
    assert hit["filename_terms"] == [] and hit["schema_terms"] == []
    assert hit["metadata_terms"] == []


def test_both_families_run_by_default(tmp_path, monkeypatch):
    _mixed_lake(tmp_path)

    hits, scanned = _weighted(tmp_path, monkeypatch)

    assert scanned
    assert {hit.document["dataset_id"] for hit in hits} == {"cells_only", "name_only"}


def test_weights_scale_the_two_families_against_each_other(tmp_path):
    """Halving one family's weight must move it down, not off."""
    _mixed_lake(tmp_path)

    def ranking(**overrides):
        return [
            hit.document["dataset_id"]
            for hit in GrepRetriever(
                RetrievalConfig(mode="grep", **overrides), tmp_path
            ).retrieve("ferry", ["ferry"], top_k=5)
        ]

    # The column-name match outscores a single cell by default...
    assert ranking()[0] == "name_only"
    # ...and stops doing so once metadata is weighted down.
    assert ranking(grep_metadata_weight=0.05)[0] == "cells_only"


def test_a_weightless_configuration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not both be zero"):
        RetrievalConfig(mode="grep", grep_value_weight=0.0, grep_metadata_weight=0.0)
    with pytest.raises(ValueError, match="must not be negative"):
        RetrievalConfig(mode="grep", grep_value_weight=-1.0)


def test_values_only_mode_still_overrides_the_metadata_weight(tmp_path):
    """The mode is the stronger statement; a stray config cannot re-enable it."""
    _mixed_lake(tmp_path)

    hits = GrepRetriever(
        RetrievalConfig(mode="grep_values", grep_metadata_weight=1.0), tmp_path
    ).retrieve("ferry", ["ferry"], top_k=5)

    assert [hit.document["dataset_id"] for hit in hits] == ["cells_only"]
    assert hits[0].document["grep_evidence"]["metadata_weight"] == 0.0


def test_retrieval_never_loads_polars_in_the_calling_process(tmp_path):
    """A native polars crash must cost a worker process, not the application.

    Polars segfaulted inside the Chainlit app during a grep_values search, and a
    segfault cannot be caught, so the defence is keeping polars queries out of
    the process that serves users. The retrieval path is held to the stronger
    bar of never importing polars at all; the full app does import it, through
    the valentine schema matcher, which runs on pandas frames. Checked in a
    fresh interpreter, because this test module imports polars itself.
    """
    import subprocess
    import sys
    import textwrap

    pd.DataFrame({"note": ["ferry terminal"]}).to_parquet(tmp_path / "boats.parquet")
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(f"""
        import sys
        from lakegen.retrieval import GrepRetriever, RetrievalConfig
        for mode in ("grep", "grep_values"):
            config = RetrievalConfig(mode=mode, scan_workers=2)
            hits = GrepRetriever(config, {str(tmp_path)!r}).retrieve(
                "ferry", ["ferry"], top_k=1
            )
            assert hits[0].document["grep_evidence"]["samples"], mode
        assert "polars" not in sys.modules, "polars loaded in the calling process"
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={**__import__("os").environ, "PYTHONPATH": f"{root / 'src'}:{root}"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


def test_grep_values_searches_exactly_the_listed_values(tmp_path):
    """Only what the model listed is searched, never words from the question.

    The question's words used to be added as extra terms, so "more" or "total"
    could match cells nobody asked about.
    """
    pd.DataFrame({"season": ["2016-17"]}).to_parquet(tmp_path / "listed.parquet")
    pd.DataFrame({"note": ["more total"]}).to_parquet(
        tmp_path / "question_words.parquet"
    )

    hits = GrepRetriever(_values_config(), tmp_path).retrieve(
        "more total districts", ["2016-17"], top_k=5
    )

    assert [hit.document["dataset_id"] for hit in hits] == ["listed"]
    evidence = hits[0].document["grep_evidence"]
    assert evidence["terms"] == ["2016-17"] and evidence["secondary_terms"] == []


def test_grep_values_keeps_each_value_whole(tmp_path):
    """A value is one pattern: "East River" is not "east" plus "river"."""
    pd.DataFrame({"place": ["East River crossing"]}).to_parquet(
        tmp_path / "whole.parquet"
    )
    pd.DataFrame({
        "place": ["east bank", "river road"], "year": ["2017", "170"],
    }).to_parquet(tmp_path / "pieces.parquet")

    hits = GrepRetriever(_values_config(), tmp_path).retrieve(
        "crossings", ["East River", "2016-17"], top_k=5
    )

    # "pieces" would match if values were split into words or "17".
    assert [hit.document["dataset_id"] for hit in hits] == ["whole"]
    assert hits[0].document["grep_evidence"]["term_counts"]["east river"] == 1


def test_grep_values_does_not_fall_back_to_the_question(tmp_path):
    """No listed values means nothing to search, not a search of the question."""
    pd.DataFrame({"note": ["ferry"]}).to_parquet(tmp_path / "boats.parquet")

    assert GrepRetriever(_values_config(), tmp_path).retrieve(
        "ferry", [], top_k=5
    ) == []


def test_grep_values_matches_literally_without_aliases(tmp_path):
    """Concept search widens "borough" to "boro"; a listed value stays exact."""
    pd.DataFrame({"boro": ["x"], "note": ["BORO 3"]}).to_parquet(
        tmp_path / "alias.parquet"
    )

    assert GrepRetriever(_values_config(), tmp_path).retrieve(
        "borough", ["borough"], top_k=5
    ) == []
    assert GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "borough", ["borough"], top_k=5
    )


def test_listed_values_are_normalized_deduplicated_and_capped():
    values = [" East   River ", "east river", "A", *[f"v{i}" for i in range(12)]]

    assert grep_module._search_values(values) == [
        "east river", *[f"v{i}" for i in range(7)]
    ]


def test_the_concept_mode_still_adds_question_terms(tmp_path):
    """Only grep_values changed: grep keeps its tokenizer and question terms."""
    pd.DataFrame({"note": ["ferry"]}).to_parquet(tmp_path / "boats.parquet")

    hits = GrepRetriever(RetrievalConfig(mode="grep"), tmp_path).retrieve(
        "ferry routes", ["ferry"], top_k=1
    )

    assert hits[0].document["grep_evidence"]["secondary_terms"] == ["route"]
