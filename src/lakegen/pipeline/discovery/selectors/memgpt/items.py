"""Named items of a question and the evidence that a table holds one.

A question that asks about several things one by one -- five sensors, four years, three party
lists, June and August -- needs every one of them in the selection. Written as one requirement
("dimension: sensor location"), one table "covered" all five. Here every item becomes its own
requirement, and a table covers an item only with evidence:

  periods      found by the system: years and month-years the question names; a range ("between
               2021 and 2025", "del 1 al 7 September 2026", "2018-2020") is one item. Periods are
               items only when the question names two or more.
  named items  listed by the agent in define_requirements(named_items=...), in whatever shape the
               model gave them (a list, a string, a dict of lists), kept only when they occur in the
               question, side by side as a list, and only when there are two or more. A name is
               matched at its tightest stretch ("Cádiz 3" in "Cádiz 16, Cádiz 3"), and a month word
               inside a name is a period only with a day or a year beside it ("Salvador Abril" is a
               street).

Evidence for an item in a table: its tokens close together in the table's title, description,
tags, file name or sampled values, or (for periods) its dates; ``scan_values`` looks through the
text columns themselves when the cheap evidence is not there.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from lakegen.pipeline.discovery.selectors.memgpt.text import NOISE, fold

logger = logging.getLogger(__name__)

MONTHS = {
    1: "january enero janvier gennaio jan ene janv gen",
    2: "february febrero fevrier febbraio feb fev",
    3: "march marzo mars mar",
    4: "april abril avril aprile apr abr avr",
    5: "may mayo mai maggio",
    6: "june junio juin giugno jun",
    7: "july julio juillet luglio jul juil lug",
    8: "august agosto aout aug ago",
    9: "september septiembre setiembre septembre settembre sep sept set",
    10: "october octubre octobre ottobre oct ott",
    11: "november noviembre novembre nov",
    12: "december diciembre decembre dicembre dec dic",
}
MONTH_OF = {name: number for number, names in MONTHS.items() for name in names.split()}
LONG_MONTH = re.compile(
    r"\b(" + "|".join(sorted((n for n in MONTH_OF if len(n) > 3), key=len, reverse=True)) + r")\b"
)
YEAR = re.compile(
    r"(?<![\d.])((?:19|20)\d{2})(?![\d.]|s\b|'s\b)"
)  # "the 2020s" is a decade, not a year
# The range keyword must lead straight into the first period: only an article, a day number and "de" between.
RANGE_BEFORE = re.compile(
    r"\b(?:entre|between|from|du|dal|dalla|tra|fra|desde|del|depuis|since)\s+(?:el\s+|le\s+|il\s+|the\s+|la\s+|l')?"
    r"(?:\d{1,2}(?:er|st|nd|rd|th)?\s+(?:de\s+)?)?$"
)
RANGE_BETWEEN = re.compile(
    r"^\s*(?:y|e|et|and|to|au|al|a|hasta|until|fino al|-)\s+(?:el\s+|le\s+|il\s+|the\s+|la\s+)?(?:\d{1,2}(?:er|st|nd|rd|th)?\s+(?:de\s+)?)?$"
)
ITEM_NOISE = NOISE | set(
    "el la los las le les il lo gli the of de del della di du des en in a sensor sensore capteur calle via rue street".split()
)
MAX_SCAN_ROWS = 200_000
MAX_UNIQUES = 60_000
PROXIMITY = 40


@dataclass(frozen=True)
class Item:
    """One period or named item of the question, which a table covers only with evidence.

    Attributes:
        label: The item as the question names it.
        kind: "period" or "named".
        years: The years a period covers.
        month: The month of a month-year period; 0 for none.
        span: For a range: (first year, last year).
        tokens: The words a named item is matched by.
    """

    label: str
    kind: str
    years: tuple[int, ...] = ()
    month: int = 0
    span: tuple[int, int] = (0, 0)
    tokens: tuple[str, ...] = ()

    @property
    def requirement_text(self) -> str:
        """The item as a requirement: ``period: <label>`` or ``item: <label>``."""
        return f"{'period' if self.kind == 'period' else 'item'}: {self.label}"


def _period_mentions(question: str) -> list[tuple[int, int, str, int, int]]:
    """(start, end, label, year, month) for every month-year and bare year, in order."""
    text = fold(question)
    found: list[tuple[int, int, str, int, int]] = []
    taken: list[tuple[int, int]] = []
    for match in LONG_MONTH.finditer(text):
        month = MONTH_OF[match.group(1)]
        after = YEAR.search(text, match.end(), match.end() + PROXIMITY)
        before = None
        if after is None:
            for m in YEAR.finditer(text, max(0, match.start() - 15), match.start()):
                before = m
        year_match = after or before
        if year_match is None:
            continue
        year = int(year_match.group(1))
        found.append(
            (
                match.start(),
                max(match.end(), year_match.end()) if after else match.end(),
                f"{match.group(1)} {year}",
                year,
                month,
            )
        )
        taken.append(year_match.span())
    for match in YEAR.finditer(text):
        if any(a <= match.start() < b for a, b in taken):
            continue
        # a full date written with digits (2026-09-15) is one period of its own
        iso = re.match(r"(?:19|20)\d{2}-(\d{2})-\d{2}", text[match.start() :])
        month = int(iso.group(1)) if iso else 0
        found.append(
            (
                match.start(),
                match.end() + (6 if iso else 0),
                match.group(1) + (f"-{iso.group(1)}" if iso else ""),
                int(match.group(1)),
                month,
            )
        )
    return sorted(found)


def question_periods(question: str) -> list[Item]:
    """The periods the question names, ranges merged; empty when it names fewer than two."""
    text = fold(question)
    mentions = _period_mentions(question)
    items: list[Item] = []
    i = 0
    while i < len(mentions):
        start, end, label, year, month = mentions[i]
        if i + 1 < len(mentions):
            nstart, nend, nlabel, nyear, nmonth = mentions[i + 1]
            between = text[end:nstart]
            before = text[max(0, start - 25) : start]
            hyphen = re.fullmatch(r"\s*[-/]\s*", between) is not None
            if hyphen or (RANGE_BEFORE.search(before) and RANGE_BETWEEN.match(between)):
                lo, hi = sorted((year, nyear))
                items.append(
                    Item(f"{label} to {nlabel}", "period", tuple(range(lo, hi + 1)), 0, (lo, hi))
                )
                i += 2
                continue
        items.append(Item(label, "period", (year,), month))
        i += 1
    unique = list({item.label: item for item in items}.values())
    return unique if len(unique) >= 2 else []


def item_tokens(label: str) -> tuple[str, ...]:
    """The words of an item's label that identify it: numbers and words of two letters or more, noise words aside."""
    words = re.findall(r"[a-z0-9]+", fold(label))  # "t248655" stays one token
    return tuple(w for w in words if (w.isdigit() or len(w) > 1) and w not in ITEM_NOISE)


# What may stand between two items of a list: separators, conjunctions, articles and a repeated head noun
# ("el de Cadiz 16, el de cuba 3 y el de General prim"; "the sensor at X, the sensor at Y").
LIST_GAP = re.compile(
    r"^[\s,;/()'-]*(?:(?:y|e|et|and|o|or|ou|und|ni)\b)?[\s,()'-]*"
    r"(?:(?:el|la|los|las|le|les|l|il|lo|gli|the|de|del|della|di|du|des|d|of|en|in|at|a|al|on|"
    r"sensor|sensore|capteur|sensors|calle|via|rue|street|lista|liste|list|varco|gate|station|estacion|"
    r"dataset|datasets|datos|dati|donnees|conjunto|conjuntos|jeu|jeux|tabla|table|tables|quartiere|zona|zone)"
    r"\b[\s,()'-]*)*$"
)


def listed_names(named: Any) -> list[str]:
    """The names the model listed, whatever shape it gave them: a list, a string, a dict of lists, nested lists."""
    if named is None:
        return []
    if isinstance(named, str):
        return [part.strip() for part in re.split(r"[;,\n]+", named) if part.strip()]
    if isinstance(named, dict):
        return [name for value in named.values() for name in listed_names(value)]
    if isinstance(named, (list, tuple)):
        return [
            name
            for value in named
            for name in (
                listed_names(value)
                if isinstance(value, (list, tuple, dict))
                else [str(value).strip()]
            )
            if name
        ]
    return [str(named).strip()]


def is_period_label(label: str) -> bool:
    """A year, or a month word with a day or a year beside it; a street named after a month is not a period."""
    folded = fold(label)
    if YEAR.search(folded):
        return True
    for match in LONG_MONTH.finditer(folded):
        if re.search(r"\d", folded[max(0, match.start() - 12) : match.end() + 12]):
            return True
    return False


def tightest_span(question: str, tokens: tuple[str, ...]) -> tuple[int, int] | None:
    """Where the item's tokens occur together, in order, in the (folded) question: the shortest such stretch.

    The shortest, so that "cadiz 3" in "cadiz 16, cadiz 3" is the second one,
    not the stretch from the first "cadiz".

    Args:
        question: The folded question.
        tokens: The item's tokens.

    Returns:
        The (start, end) of the stretch, or None when the tokens do not occur together.
    """
    pattern = _sequence(tuple(tokens))
    head = re.compile(r"(?<![a-z0-9])" + re.escape(tokens[0]) + r"(?![a-z0-9])")
    best = None
    for start in (m.start() for m in head.finditer(question)):
        match = pattern.match(question, start)
        if match and (best is None or match.end() - match.start() < best[1] - best[0]):
            best = match.span()
    return best


def over_label_words(question: str, span: tuple[int, int], label: str) -> tuple[int, int]:
    """The span widened over the label's own noise words that follow it in the question.

    "salvador" -> "salvador abril", so that such a word does not break the
    list between the item and the next one.

    Args:
        question: The folded question.
        span: Where the item's tokens occur.
        label: The item's label.

    Returns:
        The widened (start, end).
    """
    words = set(re.findall(r"[a-z0-9]+", fold(label))) - set(item_tokens(label))
    start, end = span
    while True:
        match = re.match(r"[^a-z0-9]{1,3}([a-z0-9]+)", question[end:])
        if not match or match.group(1) not in words:
            return start, end
        end += match.end()


def named_items(question: str, listed: Any) -> list[Item]:
    """The agent's named items that form a list in the question (two or more, side by side).

    An item must occur in the question, must not be a period (periods are found by the system), and must
    stand in a list with the others: only separators, conjunctions, articles or a repeated head noun between
    neighbours. Concepts the agent names ("secteur administratif", "Jeux Olympiques 2024") are not a list.

    Args:
        question: The user's question.
        listed: The named items as the model gave them (a list, a string, a dict of lists).

    Returns:
        The longest list of items side by side in the question; empty when it has fewer than two.
    """
    q = fold(question)
    found: dict[str, tuple[tuple[int, int], Item]] = {}
    for raw in listed_names(listed):
        label = " ".join(str(raw).split())[:80]
        tokens = item_tokens(label)
        if not tokens or is_period_label(label):
            continue
        span = tightest_span(q, tokens)
        if span is not None:
            found.setdefault(
                " ".join(tokens),
                (over_label_words(q, span, label), Item(label, "named", tokens=tokens)),
            )
    ordered = sorted(found.values(), key=lambda pair: pair[0])
    best: list[Item] = []
    chain: list[tuple[tuple[int, int], Item]] = []
    for span, item in ordered:
        gap = q[chain[-1][0][1] : span[0]] if chain else ""
        # neighbours of a list are separated by a comma, a dash or a conjunction, not just "a" or "de la"
        separated = bool(re.search(r"[,;/]|\s-\s|\b(?:y|e|et|and|o|or|ou|und|ni)\b", gap))
        if chain and separated and LIST_GAP.match(gap) and span[0] - chain[-1][0][1] <= 45:
            chain.append((span, item))
        else:
            chain = [(span, item)]
        if len(chain) > len(best):
            best = [it for _, it in chain]
    return best if len(best) >= 2 else []


@lru_cache(maxsize=4096)
def _sequence(tokens: tuple[str, ...]) -> re.Pattern:
    """The tokens in order, at most two other words between neighbours ("calle cadiz, 16"; "cadiz 3")."""
    gap = r"[^a-z0-9]{1,6}(?:[a-z0-9]+[^a-z0-9]{1,6}){0,2}"
    return re.compile(r"(?<![a-z0-9])" + gap.join(re.escape(t) for t in tokens) + r"(?![a-z0-9])")


def tokens_near(text: str, tokens: tuple[str, ...], window: int = PROXIMITY) -> bool:
    """The item's tokens occur in order, close together (``window`` kept for the callers' signature)."""
    return bool(tokens) and _sequence(tuple(tokens)).search(text) is not None


_RANGE_LINE = re.compile(
    r"^- (.+?): (\d{4})-(\d{2})(?:-\d{2})?\S* to (\d{4})-(\d{2})", re.MULTILINE
)
_SAMPLE_LINE = re.compile(r"^- (.+?) \(Category sample\): (.*)$", re.MULTILINE)


def _dates_cover(item: Item, profile: str) -> str | None:
    """The date column of the profile whose measured range holds the period."""
    for column, y1, m1, y2, m2 in _RANGE_LINE.findall(profile):
        lo, hi = (int(y1), int(m1)), (int(y2), int(m2))
        if item.span != (0, 0):
            if lo[0] <= item.span[1] and hi[0] >= item.span[0]:
                return f"dates ({column})"
        elif item.month:
            if lo <= (item.years[0], item.month) <= hi:
                return f"dates ({column})"
        elif lo[0] <= item.years[0] <= hi[0]:
            return f"dates ({column})"
    return None


def period_evidence(item: Item, label_text: str, profile: str) -> str | None:
    """How a table shows a period: its dates, its values, or its title / file name."""
    dates = _dates_cover(item, profile)
    if dates:
        return dates
    folded = fold(label_text)
    values = " ".join(v for _, v in _SAMPLE_LINE.findall(profile))
    for year in item.years:
        if not re.search(rf"(?<!\d){year}(?!\d)", folded + " " + values):
            continue
        if item.month:
            names = MONTHS[item.month].split()
            if any(re.search(rf"\b{n}\b", folded) for n in names) or re.search(
                rf"{year}[-/]0?{item.month}\b|\b0?{item.month}[-/]{year}", folded + " " + values
            ):
                return "title/description"
            continue
        return "title/description" if re.search(rf"(?<!\d){year}(?!\d)", folded) else "values"
    if item.month and item.years:
        # a month with a two-digit year: "June 19", "Sept-20", "06/19"
        yy = f"{item.years[0] % 100:02d}"
        names = "|".join(MONTHS[item.month].split())
        if re.search(
            rf"\b(?:{names})\b[\s'’-]{{0,3}}{yy}(?!\d)|(?<!\d)0?{item.month}[-/]{yy}(?!\d)", folded
        ):
            return "title/description"
    return None


def named_evidence(item: Item, label_text: str, profile: str) -> str | None:
    """How a table shows a named item: its title / file name or its sampled values; None when it does not."""
    if tokens_near(fold(label_text), item.tokens):
        return "title/description"
    values = fold(" ".join(v for _, v in _SAMPLE_LINE.findall(profile)))
    if tokens_near(values, item.tokens, window=20):
        return "values"
    return None


@lru_cache(maxsize=512)
def _text_values(path: str, mtime: int) -> tuple[str, ...]:
    """The distinct text values of a table's string columns, folded (bounded)."""
    try:
        schema = pq.read_schema(path)
    except Exception:
        logger.debug("No parquet schema for %s; no values to scan", path, exc_info=True)
        return ()
    columns = [
        f.name for f in schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)
    ]
    if not columns:
        return ()
    try:
        frame = pq.ParquetFile(path).iter_batches(batch_size=MAX_SCAN_ROWS, columns=columns[:40])
        batch = next(frame).to_pandas()
    except Exception:
        logger.debug("Text values of %s not read; no values to scan", path, exc_info=True)
        return ()
    values: set[str] = set()
    for column in batch.columns:
        uniques = pd.Series(batch[column].dropna().astype(str).unique()[:MAX_UNIQUES])
        values.update(fold(v) for v in uniques if len(v) <= 200)
    return tuple(values)


def scan_values(path: Path, item: Item) -> str | None:
    """Whether a cell of the table's text columns holds the item (named items and years)."""
    try:
        values = _text_values(str(path), int(path.stat().st_mtime_ns))
    except OSError:
        return None
    if item.kind == "named":
        return (
            "values (scanned)"
            if any(tokens_near(v, item.tokens, window=20) for v in values)
            else None
        )
    for year in item.years:
        if item.month:
            names = "|".join(MONTHS[item.month].split())
            pattern = re.compile(
                rf"{year}[-/.]0?{item.month}(?!\d)|(?<!\d)0?{item.month}[-/.]{year}"
                rf"|\b(?:{names})\b[^0-9]{{0,12}}{year}|{year}[^0-9]{{0,12}}\b(?:{names})\b"
            )
        else:
            pattern = re.compile(rf"(?<!\d){year}(?!\d)")
        if any(pattern.search(v) for v in values):
            return "values (scanned)"
    return None
