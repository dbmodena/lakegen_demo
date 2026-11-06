"""The MemGPT selector's evidence profile and portal requirement on small tables (experiments/profile_lab)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lakegen.pipeline.discovery.selectors.memgpt.agent import MemGPTAgent
from lakegen.pipeline.discovery.selectors.memgpt.evidence import (
    OTHER_HEADER,
    evidence_profile,
    searched,
    step,
    year_values,
)
from lakegen.pipeline.discovery.selectors.memgpt.memory import (
    CARD_CHARS,
    Card,
    Memory,
    Requirement,
    card_line,
)
from lakegen.pipeline.discovery.selectors.memgpt.tables import table_profile
from lakegen.pipeline.discovery.selectors.memgpt.text import only_place, portal_places


@pytest.fixture()
def tables(tmp_path: Path) -> Path:
    pd.DataFrame(
        {
            "Traveller Name": [f"Person {i}" for i in range(20)] + ["Debra Coupar"] * 3,
            "Travel Type": ["Train"] * 23,
            "Total": [10.5 + i for i in range(23)],
        }
    ).to_parquet(tmp_path / "travel.parquet")
    pd.DataFrame(
        {
            "Ciutat": ["València", "Madrid"] * 3,
            "Any": [2021, 2021, 2022, 2022, 2023, 2023],
            "Preu": [1592.0, 3288.7, 1600.1, 3300.0, 1700.0, 3400.0],
        }
    ).to_parquet(tmp_path / "prices.parquet")
    times = pd.date_range("2026-08-01", periods=400, freq="15min", tz="UTC").strftime(
        "%Y-%m-%dT%H:%M:%S+00:00"
    )
    pd.DataFrame({"data": times, "moto": range(400)}).to_parquet(tmp_path / "gate.parquet")
    dates = [f"{month}/1{month % 3}/12" for month in range(1, 12)] + [
        "1/05/11"
    ]  # more than a category sample
    pd.DataFrame(
        {"Column Date": dates, "Columns Title": [f"Municipal Law - {i}" for i in range(12)]}
    ).to_parquet(tmp_path / "columns.parquet")
    pd.DataFrame(
        {
            "ref": [f"r{i}" for i in range(20)],
            "kind": ["park", "garden"] * 10,
            "amount": [float(i) for i in range(20)],
            "geo_shape": [f'{{"coordinates": [[[2.3{i}, 48.8]]]}}' for i in range(20)],
            "OBJECTID": range(20),
            "created_user": ["gis"] * 20,
            "last_edited_date": [f"2024-03-{d:02d}" for d in range(1, 21)],
        }
    ).to_parquet(tmp_path / "gis.parquet")
    return tmp_path


def agent(
    tables: Path, question: str, candidates: list[str], portal: str = "", cls=MemGPTAgent
) -> MemGPTAgent:
    return cls(
        question=question,
        candidates=candidates,
        metadata={},
        table_dir=tables,
        llm=None,
        prompts=None,
        portal=portal,
    )


def test_geometry_ids_and_audit_columns_take_one_line_and_an_id_column_has_no_samples(tables: Path):
    profile = evidence_profile(tables, "gis.parquet", [])
    assert f"{OTHER_HEADER}geo_shape, OBJECTID, created_user, last_edited_date" in profile
    assert (
        "- geo_shape" not in profile
        and "- OBJECTID" not in profile
        and "coordinates" not in profile
    )
    assert (
        "- ref (str; 20 distinct, one per row)" in profile and "most frequent: 'r0'" not in profile
    )
    assert (
        "- kind (Category sample): ['park', 'garden']" in profile
        and "- amount (float64; 0 to 19)" in profile
    )
    assert (
        "last_edited_date" not in profile.split("Columns (")[0]
    )  # when a row was edited is not the table's period
    m = Memory("How much amount per kind?", ["gis.parquet"], {}, 10)
    m.requirements = [
        Requirement("R1", "measure", "amount"),
        Requirement("R2", "dimension", "kind"),
    ]
    m.record_profile(1, profile)
    card = m.cards[1]
    assert card.columns == [
        "ref",
        "kind",
        "amount",
        "geo_shape",
        "OBJECTID",
        "created_user",
        "last_edited_date",
    ]
    assert card.facts[0].startswith("7 columns: ref, kind, amount, geo_shape")
    assert card.shows == {"R1": "columns (amount)", "R2": "columns (kind)"}


def test_a_card_says_each_requirement_once_and_never_cuts_a_fact(tables: Path):
    meta = {
        "travel.parquet": {
            "title": "Travel 2021",
            "description": "",
            "columns.name": ["Traveller Name", "Travel Type", "Total"],
        }
    }
    m = Memory(
        "How much did Debra Coupar spend on train travel in 2021?", ["travel.parquet"], meta, 10
    )
    m.requirements = [
        Requirement("R1", "dimension", "traveller"),
        Requirement("R2", "filter", "train"),
        Requirement("R3", "period", "2021"),
    ]
    m.record_profile(1, evidence_profile(tables, "travel.parquet", ["coupar", "debra"]))
    line = m._card_lines()[1]
    assert line.startswith(
        '  #1 "Travel 2021" -- 23 rows; shows R1 in columns (Traveller Name); R2 in values (Travel Type); '
        "R3 in title year | 3 columns: Traveller Name, Travel Type, Total | 2021: not in any inspected value or date range"
    )
    assert (
        "words in column names" not in line
        and "words in values" not in line
        and "words found nowhere: " in line
    )
    assert any(
        f.startswith("words in values (search): coupar in Traveller Name") for f in m.cards[1].facts
    )  # the data stays

    class Positive(Memory):
        card_negatives = False

    m.__class__ = Positive
    line = m._card_lines()[1]
    assert (
        "not in any inspected" not in line
        and "words found nowhere" not in line
        and "3 columns:" in line
    )
    card = Card(1, "x.parquet", "t", rows="5", facts=["a" * 300, "b" * 300, "c" * 50])
    assert card_line(card) == "5 rows; shows none of the requirements | " + "a" * 300 + " | ..."
    assert len(card_line(card)) <= CARD_CHARS + 60


def test_the_candidates_block_can_show_inspected_tables_by_title_only(tables: Path):
    meta = {
        n: {
            "title": f"Table {n}",
            "description": "A long description of the table " * 3,
            "columns.name": ["c1", "c2"],
        }
        for n in ("travel.parquet", "prices.parquet")
    }
    m = Memory("q", ["travel.parquet", "prices.parquet"], meta, 2)
    m.record_profile(1, evidence_profile(tables, "travel.parquet", []))
    assert m.render_candidates().count("indexed columns: c1, c2") == 2

    class Slim(Memory):
        slim_inspected = True

    m.__class__ = Slim
    lines = m.render_candidates().splitlines()
    assert lines[0] == "#1 [inspected] Table travel.parquet" and lines[1].startswith(
        "#2 [not inspected] Table prices.parquet"
    )
    assert m.render_candidates().count("indexed columns: c1, c2") == 1


def test_year_values_only_for_columns_of_years():
    assert year_values(pd.Series([2021, 2022, 2021])) == [2021, 2022]
    assert year_values(pd.Series([1, 2019, 2020])) is None
    assert year_values(pd.Series(["2021", "2022"])) is None


def test_step_names_the_gap_between_dates():
    assert (
        step(pd.Series(pd.date_range("2026-01-01", periods=50, freq="15min").astype(str)))
        == "every 15 minutes"
    )
    assert (
        step(pd.Series(pd.date_range("2020-01-01", periods=24, freq="MS").astype(str))) == "monthly"
    )


def test_profile_shows_the_name_the_sample_hides(tables: Path):
    profile = evidence_profile(tables, "travel.parquet", ["coupar", "debra", "accommodation"])
    assert searched(profile) == {"coupar": "Traveller Name", "debra": "Traveller Name"}
    assert "- Traveller Name (str; 21 distinct; most frequent: 'Debra Coupar'" in profile
    assert "- Total (float64; 10.50 to 32.50)" in profile
    assert "Debra Coupar" not in table_profile(tables, "travel.parquet")


def test_profile_lists_years_held_as_integers(tables: Path):
    profile = evidence_profile(tables, "prices.parquet", [])
    assert "- Any (Category sample): ['2021', '2022', '2023']" in profile
    assert "- Preu (float64; 1,592 to 3,400)" in profile


def test_profile_gives_the_step_and_reads_short_dates(tables: Path):
    assert "every 15 minutes)" in evidence_profile(tables, "gate.parquet", []).split("Columns (")[0]
    head = evidence_profile(tables, "columns.parquet", []).split("Columns (")[0]
    assert "- Column Date: 2011-01-05 to 2012-11-12 (m/d/yy dates parsed from the values)" in head


def _card(tables: Path, name: str, question: str, requirement: str):
    m = Memory(question, [name], {name: {"title": name}}, 1)
    m.requirements = [
        Requirement("R1", "subject", requirement),
        Requirement("R2", "period", "2021"),
    ]
    m.record_profile(1, evidence_profile(tables, name, ["coupar", "debra"]))
    return m.cards[1]


def test_cards_read_the_search_as_values(tables: Path):
    card = _card(
        tables, "travel.parquet", "How much did Debra Coupar spend in 2021?", "Debra Coupar"
    )
    assert card.shows["R1"] == "values (Traveller Name)"
    assert not any("coupar" in f for f in card.facts if f.startswith("words found nowhere"))
    assert any(
        f.startswith("words in values (search): ") and "coupar in Traveller Name" in f
        for f in card.facts
    )


def test_cards_without_a_search_line_are_unchanged(tables: Path):
    m = Memory(
        "How much did Debra Coupar spend?",
        ["travel.parquet"],
        {"travel.parquet": {"title": "t"}},
        1,
    )
    m.requirements = [Requirement("R1", "subject", "Debra Coupar")]
    m.record_profile(1, table_profile(tables, "travel.parquet"))
    assert "R1" not in m.cards[1].shows
    assert any(f.startswith("words found nowhere: ") and "coupar" in f for f in m.cards[1].facts)


def test_cards_find_the_year_in_an_integer_column(tables: Path):
    card = _card(tables, "prices.parquet", "Price in València in 2021?", "price")
    assert "2021: in Any values" in card.facts
    assert card.shows["R2"].startswith("dates (Any")


def test_inspect_uses_the_evidence_profile_unless_switched_off(tables: Path):
    on = agent(tables, "How much did Debra Coupar spend?", ["travel.parquet"])
    on.define_requirements(["subject: Debra Coupar"])
    assert "Values matching the question" in on.inspect([1])

    class Plain(MemGPTAgent):
        evidence_profiles = False

    off = agent(tables, "How much did Debra Coupar spend?", ["travel.parquet"], cls=Plain)
    off.define_requirements(["subject: Debra Coupar"])
    assert off.inspect([1]) == f"[#1]\n{table_profile(tables, 'travel.parquet')}"


def test_portal_places_come_from_open_data_titles():
    assert portal_places("New York City Open Data portal") == {"new", "york", "city", "nyc"}
    assert portal_places("City of Chicago Open Data portal") == {"chicago"}
    assert (
        portal_places(
            "KramaBench data lake (archaeology, astronomy, biomedical, environment, legal, wildfire)"
        )
        == set()
    )
    assert portal_places("BIRD databases (TARGET benchmark)") == set()
    valencia = portal_places("Valencia Open Data portal")
    for text in (
        "Valencia",
        "location: Valencia",
        "ubicación = València",
        "municipio de València",
        "ciudad=Valencia",
    ):
        assert only_place(text, valencia), text
    for text in ("incidencias en Valencia", "province = Valencia", "Barcelona", "city"):
        assert not only_place(text, valencia), text
    assert not only_place("city", portal_places("New York City Open Data portal"))


def test_define_requirements_waives_the_portal_place(tables: Path):
    a = agent(
        tables,
        "Price of housing in Valencia in 2021?",
        ["prices.parquet"],
        portal="Valencia Open Data portal",
    )
    reply = a.define_requirements(["measure: price", "filter: location Valencia", "period: 2021"])
    waived = {r.id: r.waived for r in a.memory.requirements if r.waived}
    assert list(waived) == ["R2"] and "portal's own place" in waived["R2"]
    assert reply.endswith(
        "The system waived R2: the portal's own place, which every table of the Valencia Open Data portal covers."
    )
    assert [r.id for r in a.memory.open_requirements()] == ["R1", "R3"]

    class Keep(MemGPTAgent):
        waive_portal_place = False

    kept = agent(
        tables,
        "Price of housing in Valencia in 2021?",
        ["prices.parquet"],
        portal="Valencia Open Data portal",
        cls=Keep,
    )
    kept.define_requirements(["measure: price", "filter: location Valencia", "period: 2021"])
    assert not any(r.waived for r in kept.memory.requirements)
