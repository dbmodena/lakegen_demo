"""Naming rules shared by everything that reads catalogue metadata.

CKAN keeps a title on the *package* and a name on each *resource*. Which one a
table should be called is not obvious, and getting it wrong is quiet: the index
still builds, the queries still run, the tables are just unfindable.

On data.gov.uk the resource name of an ArcGIS-harvested file is the literal
format label ("CSV"), so taking the resource name first leaves ~23% of the UK
lake titled "CSV" with the real name unused in the package title. Taking only
the package title is no better: 1,426 UK tables share the package title
"Organogram of Staff Roles & Salaries", and the resource name ("2021-12-31
Organogram (Junior)") is the only thing telling them apart.

So a table is named by both, in that order — the family it belongs to, then the
file's own name within it — with format noise removed from each. In this lake
that noise takes three forms, all of which put a useless "csv" token into the
index: a name that is only a format label and filler ("CSV Download"), a
trailing file extension ("2022 NI Water Results.csv"), and a format token
sitting inside an otherwise real name ("Organogram - Senior CSV data").
"""

from __future__ import annotations

import re

# A "name" made only of these says nothing about the table.
FORMAT_LABELS = frozenset({
    "csv", "tsv", "xls", "xlsx", "ods", "json", "geojson", "xml", "zip", "pdf",
    "txt", "kml", "kmz", "api", "html", "htm", "rdf", "wfs", "wms", "parquet",
    "shp", "shapefile", "doc", "docx", "rss", "atom",
})

# Words that describe the act of publishing rather than the data, plus the
# function words that can pad them out ("Download the data file"). Used only to
# decide whether a name says nothing at all; never removed from a real name.
FILLER_WORDS = frozenset({
    "download", "downloads", "file", "files", "data", "link", "links",
    "resource", "open", "format", "export", "view", "here", "click",
    "dataset", "table", "attachment", "preview",
    "the", "a", "an", "of", "for", "in", "on", "and", "or", "to", "as", "this",
})

_EXTENSION = re.compile(
    r"\.(?:" + "|".join(sorted(FORMAT_LABELS)) + r")\s*$", re.IGNORECASE)
_FORMAT_TOKEN = re.compile(
    r"(?<![0-9a-z])(?:" + "|".join(sorted(FORMAT_LABELS)) + r")(?![0-9a-z])",
    re.IGNORECASE)
# Brackets are deliberately absent: stripping them would turn "Organogram
# (Junior)" into "Organogram (Junior". A pair left empty by token removal is
# collapsed instead.
_SEPARATORS = " -–—_,;:/|"
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")


def _words(value: str) -> list[str]:
    return [token for token in re.split(r"[^0-9a-zA-Z]+", value or "") if token]


def is_uninformative(value: object) -> bool:
    """Is this "name" only a file format and publishing filler?

    ``"CSV"``, ``"CSV Download"`` and ``"Download the data file"`` are; a name
    keeping any word of its own, such as ``"Organogram - Senior CSV data"``, is
    not.
    """
    return not [word for word in _words(str(value or ""))
                if word.casefold() not in FORMAT_LABELS
                and word.casefold() not in FILLER_WORDS]


def strip_format_noise(value: object) -> str:
    """Drop a trailing file extension and any standalone format token.

    Filler words are left alone: they decide whether a name is worthless, but
    removing them from a real name would change what it says ("Land Registry
    Price Paid Data" should keep its "Data").
    """
    text = _EXTENSION.sub("", str(value or "").strip())
    text = _FORMAT_TOKEN.sub(" ", text)
    text = _EMPTY_BRACKETS.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(_SEPARATORS)
    return text


def catalogue_title(package_title: object, resource_name: object, *,
                    fallback: object = "") -> str:
    """The title a table should be indexed and shown under.

    Joins the package title and the resource name, dropping either when it is
    empty or says nothing but its format, stripping format noise from what is
    left, and skipping the second when it repeats the first. ``fallback`` is
    used only when nothing usable remains.

    >>> catalogue_title("Regions (December 2024) Boundaries EN BFC", "CSV")
    'Regions (December 2024) Boundaries EN BFC'
    >>> catalogue_title("Organogram of Staff Roles & Salaries", "Organogram - Senior CSV data")
    'Organogram of Staff Roles & Salaries Organogram - Senior data'
    """
    parts: list[str] = []
    for value in (package_title, resource_name):
        text = str(value or "").strip()
        if not text or is_uninformative(text):
            continue
        text = strip_format_noise(text)
        if text and text not in parts:
            parts.append(text)
    return " ".join(parts) or strip_format_noise(fallback)
