"""A table's catalogue entry: whole titles, the dataset's own title from the lake's catalogue, who published it."""

from __future__ import annotations

import json

from lakegen.retrieval.solr.catalogue import (
    ckan_titles,
    single_text,
    solr_metadata_from_doc,
    with_dataset_title,
)
from lakegen.retrieval.solr.client import KNN_METADATA_FIELDS


def test_the_semantic_branch_fetches_the_publisher_and_the_second_titles():
    assert {"publisher", "resource_name", "responsible_entity"} <= set(KNN_METADATA_FIELDS)


def test_list_valued_catalogue_text_is_joined_not_spelled_as_a_list():
    assert (
        single_text(["Pressupost d'ingressos", "", "Pressupost d'ingressos", "<b>Llistat</b>"])
        == "Pressupost d'ingressos / Llistat"
    )
    assert single_text("Presupuesto") == "Presupuesto" and single_text(None) == ""


def test_the_entry_keeps_the_titles_the_publisher_and_the_responsible_entity_whole():
    title = "Niveles sonoros registrados por el sensor de ruido de la calle Carles Cervera, Chaflán Reina Doña María"
    meta = solr_metadata_from_doc(
        {
            "title": title,
            "resource_name": ["Nivells sonors del sensor del carrer Carles Cervera"],
            "publisher": "Ajuntament de València",
            "responsible_entity": ["JCDecaux"],
            "description": "x",
        }
    )
    assert (
        meta["title"] == title
        and meta["resource_name"] == "Nivells sonors del sensor del carrer Carles Cervera"
    )
    assert (meta["publisher"], meta["responsible_entity"]) == ("Ajuntament de València", "JCDecaux")
    plain = solr_metadata_from_doc({"title": "CSV"})
    assert (plain["resource_name"], plain["publisher"], plain["responsible_entity"]) == (
        "CSV",
        "",
        "",
    )


def test_dataset_titles_come_from_the_lakes_ckan_package_list(tmp_path):
    path = tmp_path / "metadata_retrieved_cleaned.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "ds-1",
                    "title": "Organogram of Staff Roles &amp; Salaries",
                    "organization": {"title": "HM Treasury"},
                },
                {
                    "id": "ds-2",
                    "title": "Obras ejecutadas",
                    "organization": {"name": "ajuntament-de-valencia"},
                },
                {"id": "", "title": "no id"},
                "not a package",
            ]
        ),
        encoding="utf-8",
    )
    titles = ckan_titles(path)
    assert titles == {
        "ds-1": ("Organogram of Staff Roles & Salaries", "HM Treasury"),
        "ds-2": ("Obras ejecutadas", "ajuntament-de-valencia"),
    }
    assert ckan_titles(tmp_path / "missing.json") == {}
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    assert ckan_titles(tmp_path / "broken.json") == {}

    # a UK file indexed under its own name gets the dataset's title and, lacking one, the publisher
    uk = with_dataset_title(
        solr_metadata_from_doc({"title": "CSV", "dataset_id": "ds-1"}),
        {"dataset_id": "ds-1"},
        titles,
    )
    assert (uk["dataset_title"], uk["publisher"]) == (
        "Organogram of Staff Roles & Salaries",
        "HM Treasury",
    )
    # a Valencia entry whose indexed title is the dataset's gets no second copy of it, and keeps its own publisher
    es = with_dataset_title(
        solr_metadata_from_doc(
            {
                "title": "Obras ejecutadas",
                "publisher": "Ajuntament de València",
                "dataset_id": "ds-2",
            }
        ),
        {"dataset_id": "ds-2"},
        titles,
    )
    assert "dataset_title" not in es and es["publisher"] == "Ajuntament de València"
    assert "dataset_title" not in with_dataset_title(
        {"title": "x"}, {"dataset_id": "unknown"}, titles
    )
    assert with_dataset_title({"title": "x"}, {"dataset_id": "ds-1"}, {}) == {"title": "x"}
