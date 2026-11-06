"""The working memory of one MemGPT selection, and how each turn shows it.

Three blocks, as in MemGPT (Packer et al. 2023):

  read-only   the instructions, the tools, the question and, once defined, the requirements R1..Rn
              (the prompt ``memgpt_selector/system_prompt.md``; never edited after definition)
  STATE       kept by the system from the agent's tool calls: the selection (each table with the
              requirements it covers), the requirements still open or waived, one card per
              inspected table (with which requirement each one shows), the relationships measured
              between selected tables, the ranked queue of tables still worth inspecting, the
              budgets, the next step
  history     a FIFO of the last ``KEEP_RECENT`` exchanges verbatim, preceded by one summary that
              absorbs every exchange the FIFO evicts (one line each, written by the system)

The queue is what prevents early stoppage: every uninspected candidate whose catalogue entry
mentions an open requirement (or matches the requirements as well as the selection does) is
listed with the reason, and the confirm and reject gates (``agent``) refuse while it is not empty.
A period or named item of the question (``items``) is a requirement of its own, which a table
covers only with evidence. A profile's search line (``evidence``: the question's words found in the
table's values) counts as values on the card.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lakegen.pipeline.discovery.selectors.memgpt.evidence import OTHER_HEADER, searched
from lakegen.pipeline.discovery.selectors.memgpt.items import (
    named_evidence,
    period_evidence,
    tokens_near,
)
from lakegen.pipeline.discovery.selectors.memgpt.text import (
    YEAR,
    content_words,
    fold,
    mentions,
    question_words,
)
from lakegen.pipeline.evidence import requirement_ledger as ledger

KEEP_RECENT = 3
MAX_FACTS_WORDS = 5
MAX_CARD_COLUMNS = 14
"""Column names kept on a card, so a select can bind columns without re-reading the profile."""
CARD_CHARS = 600
"""The budget of a card's line in STATE: whole facts are dropped past it, never cut mid-word."""
PERIOD_NEGATIVE = ": not in any inspected value or date range"
WORD_FACTS = ("words in values", "words in column names", "words in title/description")
"""The question's words found in the profile: on the card they are said once, per requirement, by ``shows``."""
_ROWS = re.compile(r"^Rows: (.+)$", re.MULTILINE)
_COVERAGE = re.compile(r"^- (.+?): (\S+) to (\S+)", re.MULTILINE)
_SAMPLE = re.compile(r"^- (.+?) \(Category sample\): (.*)$", re.MULTILINE)
_COLUMN = re.compile(r"^- (.+?)(?: \(|: )", re.MULTILINE)


def _period_facts(question: str, profile: str) -> list[str]:
    facts: list[str] = []
    for period in ledger.question_periods(question):
        evidence = ledger.inspected_period_evidence(profile, period)
        if evidence is None:
            facts.append(f"{period}: not in any inspected value or date range")
        elif evidence[1] == "values":
            facts.append(f"{period}: in {evidence[0]} values")
        else:
            facts.append(f"{period}: within {evidence[0]} date range")
    coverage = [f"{col}: {start}..{end}" for col, start, end in _COVERAGE.findall(profile)[:2]]
    if coverage and not ledger.question_periods(question):
        facts.append("dates " + "; ".join(coverage))
    return facts


def _word_facts(
    question: str, columns: list[str], sampled: dict[str, str], metadata_text: str
) -> list[str]:
    column_names = " ".join(columns).casefold()
    found: dict[str, list[str]] = {"values": [], "columns": [], "metadata": [], "missing": []}
    for word in question_words(question):
        column = next((c for c, values in sampled.items() if mentions(values, word)), None)
        if column is not None:
            found["values"].append(f"{word} in {column}")
        elif mentions(column_names, word):
            found["columns"].append(word)
        elif mentions(metadata_text.casefold(), word):
            found["metadata"].append(word)
        else:
            found["missing"].append(word)
    facts = []
    for key, label, limit in (
        ("values", "words in values", MAX_FACTS_WORDS),
        ("columns", "words in column names", MAX_FACTS_WORDS),
        ("metadata", "words in title/description", MAX_FACTS_WORDS),
        ("missing", "words found nowhere", MAX_FACTS_WORDS + 1),
    ):
        if found[key]:
            facts.append(f"{label}: " + ", ".join(found[key][:limit]))
    return facts


def profile_columns(profile: str) -> list[str]:
    """The columns a profile lists: its "- name" lines, then the ones its ``OTHER_HEADER`` line only names."""
    # The column lines follow the "Columns (...)" header; the temporal-coverage lines above it share the "- name:" shape.
    section = profile.split("Columns (", 1)[-1] if "Columns (" in profile else profile
    columns = list(dict.fromkeys(_COLUMN.findall(section)))
    other = next((ln for ln in profile.splitlines() if ln.startswith(OTHER_HEADER)), "")
    return columns + [
        c.strip()
        for c in other[len(OTHER_HEADER) :].split(",")
        if c.strip() and c.strip() not in columns
    ]


def card_facts(question: str, profile: str, metadata_text: str = "") -> tuple[str, list[str]]:
    """What a profile says that matters for the question: rows, coverage of its periods, where its words are."""
    rows = _ROWS.search(profile)
    facts = _period_facts(question, profile)
    columns = profile_columns(profile)
    sampled = {column: values.casefold() for column, values in _SAMPLE.findall(profile)}
    facts += _word_facts(question, columns, sampled, metadata_text)
    shown = ", ".join(str(c)[:28] for c in columns[:MAX_CARD_COLUMNS])
    more = (
        f" (+{len(columns) - MAX_CARD_COLUMNS} more: recall for the full list)"
        if len(columns) > MAX_CARD_COLUMNS
        else ""
    )
    facts.insert(0, f"{len(columns)} columns: {shown}{more}" if columns else "no columns read")
    return (rows.group(1) if rows else ""), facts


@dataclass
class Requirement:
    """One requirement R1..Rn of the question, as the agent defined it or the system added it.

    Attributes:
        id: ``R1``, ``R2``, ...
        kind: The kind the agent wrote before the colon (``data`` when none), ``period`` or ``item``.
        text: What the requirement asks for.
        waived: Why the agent says no data is needed for it (empty when not waived).
        item: The named item or period (``items.Item``) this requirement is about; covered only with evidence.
    """

    id: str
    kind: str
    text: str
    waived: str = ""
    item: Any = None

    @property
    def label(self) -> str:
        """The requirement as STATE lists it: ``R1 [kind] text``."""
        return f"{self.id} [{self.kind}] {self.text}"


@dataclass
class Card:
    """What STATE keeps of an inspected table.

    Attributes:
        number: The candidate's number.
        name: The table file.
        title: The table's heading.
        rows: The row count the profile states.
        facts: What the profile says that matters for the question.
        shows: Requirement id -> where the profile shows it (columns, values, dates, title).
        columns: The columns the profile lists.
    """

    number: int
    name: str
    title: str
    rows: str = ""
    facts: list[str] = field(default_factory=list)
    shows: dict[str, str] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)


@dataclass
class Exchange:
    """One turn of the history: the agent's call and the system's result.

    Attributes:
        turn: The turn's number.
        call: The call as the history shows it.
        result: The tool's result.
        summary: One line that replaces the exchange once the FIFO evicts it.
    """

    turn: int
    call: str
    result: str
    summary: str


def _names(text: str, item) -> bool:
    """Whether a requirement's text names the item (a period's first year and month, or the item's tokens)."""
    if item.kind == "period":
        return all(re.search(rf"(?<!\d){y}(?!\d)", text) for y in item.years[:1]) and (
            not item.month or item.label.split()[0] in fold(text)
        )
    return tokens_near(fold(text), item.tokens)


class Memory:
    """The working memory of one selection: the requirements, STATE and the history, and how a turn renders them.

    Candidates are numbered from 1 in retrieval order; the first ``visible``
    are shown, the rest only once expanded.

    Attributes:
        card_negatives: Cards say which of the question's periods and words the table does not hold;
            off, only what it holds.
        slim_inspected: The candidates block repeats an inspected table's description and indexed
            columns; on, its title only (its card in STATE holds the real columns).
        question: The user's question.
        candidates: The candidate table files, in retrieval order.
        metadata: Catalogue metadata by table file.
        visible: How many candidates are shown.
        requirements: The requirements, once defined.
        cards: Candidate number -> the card of an inspected table.
        profiles: Candidate number -> the profile of an inspected table.
        selection: Candidate number -> requirement ids it covers.
        unselected: Candidate number -> why it left the selection.
        relations: Pair of selected candidates -> how they join and union.
        scanned: Item evidence found by scanning a table's text values.
        recent: The last exchanges, verbatim.
        summary: One line per exchange the FIFO evicted.
        last_refusal: The system's last refusal, shown in STATE.

    Args:
        question: The user's question.
        candidates: The candidate table files, in retrieval order.
        metadata: Catalogue metadata by table file.
        visible: How many candidates are shown first.
    """

    card_negatives = True
    slim_inspected = False

    def __init__(
        self,
        question: str,
        candidates: list[str],
        metadata: dict[str, dict[str, Any]],
        visible: int,
    ) -> None:
        self.question = question
        self.candidates = candidates
        self.metadata = metadata
        self.visible = visible
        self.requirements: list[Requirement] = []
        self.cards: dict[int, Card] = {}
        self.profiles: dict[int, str] = {}
        self.selection: dict[int, list[str]] = {}
        self.unselected: dict[int, str] = {}
        self.relations: dict[tuple[int, int], str] = {}
        self.scanned: dict[tuple[int, str], str] = {}
        self.recent: list[Exchange] = []
        self.summary: list[str] = []
        self.last_refusal = ""

    # ---------------------------------------------------------------- catalogue

    def _field(self, number: int, key: str) -> str:
        return " ".join(
            str(self.metadata.get(self.candidates[number - 1], {}).get(key) or "").split()
        )

    def title(self, number: int) -> str:
        """The indexed title, whole; the file name when the entry has none."""
        return self._field(number, "title") or self.candidates[number - 1]

    def dataset_title(self, number: int) -> str:
        """The title of the dataset the file belongs to, when it is not the indexed title (data.gov.uk files)."""
        return self._field(number, "dataset_title")

    def other_title(self, number: int) -> str:
        """The file's own name when it says something the title does not (Valencia's Valencian names)."""
        name = self._field(number, "resource_name")
        return (
            ""
            if name.casefold()
            in (self.title(number).casefold(), self.dataset_title(number).casefold())
            else name
        )

    def publisher(self, number: int) -> str:
        """Who stands behind the table: the publisher, and the responsible entity when the portal names one."""
        publisher, entity = (
            self._field(number, "publisher"),
            self._field(number, "responsible_entity"),
        )
        if entity and entity.casefold() != publisher.casefold():
            return (
                f"{publisher}; responsible entity: {entity}"
                if publisher
                else f"responsible entity: {entity}"
            )
        return publisher

    def heading(self, number: int) -> str:
        """The title as STATE and the contrast check name a table: whole, with the dataset's title when it has one."""
        dataset = self.dataset_title(number)
        return f"{self.title(number)} (dataset: {dataset})" if dataset else self.title(number)

    def catalogue_lines(self, number: int) -> list[tuple[str, str]]:
        """The labelled lines under a candidate's title: the dataset, its other name, the publisher."""
        lines = [
            ("dataset", self.dataset_title(number)),
            ("also titled", self.other_title(number)),
            ("publisher", self.publisher(number)),
        ]
        return [(label, value) for label, value in lines if value]

    def _titles(self, number: int) -> list[str]:
        return [
            t
            for t in (
                self.title(number),
                self.dataset_title(number),
                self.other_title(number),
                self.publisher(number),
            )
            if t
        ]

    def catalogue_text(self, number: int) -> str:
        """Titles, publisher, description, tags, indexed columns and file name, lowercased, for word matching."""
        name = self.candidates[number - 1]
        meta = self.metadata.get(name, {})
        columns = " ".join(str(c) for c in (meta.get("columns.name") or []))
        tags = " ".join(map(str, meta.get("tags") or []))
        return " ".join(
            str(p)
            for p in (*self._titles(number), meta.get("description"), tags, columns, name)
            if p
        ).casefold()

    def label_text(self, number: int) -> str:
        """Titles, publisher, description, tags and file name: where a table names what it is about."""
        name = self.candidates[number - 1]
        meta = self.metadata.get(name, {})
        return " ".join(
            str(p)
            for p in (
                *self._titles(number),
                meta.get("description"),
                " ".join(map(str, meta.get("tags") or [])),
                name,
            )
            if p
        )

    def item_evidence(self, number: int, requirement: Requirement) -> str | None:
        """How the table shows the requirement's item, from its catalogue entry and, once inspected, its profile."""
        item = requirement.item
        profile = self.profiles.get(number, "")
        if item.kind == "period":
            # a year mentioned in passing in a description ("Paris Vote depuis 2017") is not the table's period
            name = self.candidates[number - 1]
            meta = self.metadata.get(name, {})
            label = " ".join(
                str(p)
                for p in (
                    self.title(number),
                    self.dataset_title(number),
                    self.other_title(number),
                    " ".join(map(str, meta.get("tags") or [])),
                    name,
                )
                if p
            )
            return period_evidence(item, label, profile)
        return named_evidence(item, self.label_text(number), profile)

    def matches(self, number: int, requirement: Requirement) -> list[str]:
        """The requirement's words (and years) the candidate's catalogue entry mentions."""
        if requirement.item is not None:
            return [requirement.item.label] if self.item_evidence(number, requirement) else []
        text = self.catalogue_text(number)
        found = [w for w in content_words(requirement.text) if mentions(text, w)]
        found += [y for y in YEAR.findall(requirement.text) if y in text]
        return list(dict.fromkeys(found))

    def plausibility(self, number: int) -> int:
        """How many words of the requirements not waived the candidate's catalogue entry mentions."""
        return sum(len(self.matches(number, r)) for r in self.requirements if not r.waived)

    # ---------------------------------------------------------------- requirements

    def define(self, items: list[str], named: list[Any] = ()) -> list[str]:
        """Fix the requirements, and add one per named item or period that has none of its own.

        Args:
            items: The requirements the agent wrote, ``kind: text`` or text alone.
            named: The question's named items and periods (``items.Item``).

        Returns:
            The labels of the requirements the system added.
        """
        reqs = []
        for item in items:
            text = " ".join(str(item).split())
            if not text:
                continue
            kind, _, body = text.partition(":")
            if body and len(kind) <= 14 and " " not in kind.strip():
                kind, text = kind.strip().casefold(), body.strip()
            else:
                kind = "data"
            reqs.append(Requirement(f"R{len(reqs) + 1}", kind, text[:160]))
        added: list[str] = []
        for item in named:
            own = [
                r
                for r in reqs
                if _names(r.text, item) and sum(_names(r.text, other) for other in named) == 1
            ]
            if own:
                own[0].item = item
                continue
            reqs.append(
                Requirement(
                    f"R{len(reqs) + 1}",
                    "period" if item.kind == "period" else "item",
                    item.label[:120],
                    item=item,
                )
            )
            added.append(reqs[-1].label)
        self.requirements = reqs[:20]
        return added

    def requirement(self, ref: Any) -> Requirement | None:
        """The requirement a reference names by its number (``R2``, ``2``); None for none."""
        match = re.search(r"\d+", str(ref))
        if not match:
            return None
        rid = f"R{int(match.group())}"
        return next((r for r in self.requirements if r.id == rid), None)

    def covered_by(self) -> dict[str, list[int]]:
        """Requirement id -> the selected candidates that cover it."""
        out: dict[str, list[int]] = {r.id: [] for r in self.requirements}
        for number, rids in self.selection.items():
            for rid in rids:
                out.setdefault(rid, []).append(number)
        return out

    def open_requirements(self) -> list[Requirement]:
        """The requirements neither waived nor covered by the selection."""
        covered = self.covered_by()
        return [r for r in self.requirements if not r.waived and not covered.get(r.id)]

    # ---------------------------------------------------------------- inspection

    def record_profile(self, number: int, profile: str) -> None:
        """Keep an inspected table's profile and write its card."""
        name = self.candidates[number - 1]
        self.profiles[number] = profile
        card = Card(number, name, self.heading(number))
        if profile.startswith("Error"):
            card.facts = [profile[:120]]
        else:
            card.rows, card.facts = card_facts(self.question, profile, self.catalogue_text(number))
            card.facts = _with_search(card.facts, searched(profile))
            card.columns = profile_columns(profile)
        self.cards[number] = card
        self.refresh_shows(number)

    def refresh_shows(self, number: int) -> None:
        """Recompute which requirements the card of an inspected table shows, and where."""
        card, profile = self.cards.get(number), self.profiles.get(number, "")
        if card is None:
            return
        samples = {c: v.casefold() for c, v in _SAMPLE.findall(profile)}
        title = f"{card.title} {card.name} {self.catalogue_text(number)}".casefold()
        found = searched(profile)
        card.shows = {}
        for r in self.requirements:
            if r.item is not None:
                evidence = self.item_evidence(number, r) or self.scanned.get((number, r.id))
                where = [evidence] if evidence else []
            else:
                where = _shown_by(r, card.columns, samples, title, profile, found)
            if where:
                card.shows[r.id] = ", ".join(where)

    def inspected(self) -> list[int]:
        """The numbers of the tables inspected without an error."""
        return sorted(
            n for n, c in self.cards.items() if not (c.facts and c.facts[0].startswith("Error"))
        )

    # ---------------------------------------------------------------- the queue

    def queue(self) -> list[tuple[int, str]]:
        """Uninspected visible candidates still worth inspecting, best first, with the reason.

        A candidate is queued when its catalogue entry mentions an open requirement, or when it matches the
        requirements at least half as well as the best-matching table of the selection (an alternative the
        agent has not looked at). Hidden candidates are not queued; ``hidden_matches`` reports them.

        Returns:
            (candidate number, reason) pairs, best first.
        """
        if not self.requirements:
            return []
        opened = self.open_requirements()
        best_selected = max((self.plausibility(n) for n in self.selection), default=0)
        entries = []
        for number in range(1, self.visible + 1):
            if number in self.cards:
                continue
            reasons = []
            for r in opened:
                words = self.matches(number, r)
                if words:
                    reasons.append(f"{r.id} '{', '.join(words[:3])}'")
            score = self.plausibility(number)
            if not reasons and best_selected and score >= max(1, best_selected // 2):
                reasons.append(
                    f"an alternative: matches the requirements ({score}) about as well as the selection ({best_selected})"
                )
            if not reasons and not self.selection and score:
                reasons.append(f"matches the requirements ({score})")
            if reasons:
                entries.append(
                    (
                        score + 10 * len([x for x in reasons if x.startswith("R")]),
                        number,
                        "; ".join(reasons),
                    )
                )
        entries.sort(key=lambda e: (-e[0], e[1]))
        return [(number, reason) for _, number, reason in entries]

    def partitions_to_add(self) -> list[tuple[int, list[str]]]:
        """Inspected, unselected tables whose card shows an open item requirement."""
        opened = {r.id for r in self.open_requirements() if r.item is not None}
        out = []
        for n in self.inspected():
            if n in self.selection:
                continue
            rids = [rid for rid in self.cards[n].shows if rid in opened]
            if rids:
                out.append((n, rids))
        return out

    def alternatives(self) -> list[tuple[int, int, list[str]]]:
        """Inspected, unselected tables that show every requirement a selected table covers and shows."""
        adding = {n for n, _ in self.partitions_to_add()}
        out = []
        for s, rids in sorted(self.selection.items()):
            shown = [rid for rid in rids if rid in self.cards[s].shows]
            if not shown:
                continue
            for n in self.inspected():
                if n in self.selection or n in self.unselected or n in adding:
                    continue
                if all(rid in self.cards[n].shows for rid in shown):
                    out.append((n, s, shown))
        return out

    def hidden_matches(self) -> list[tuple[int, list[str]]]:
        """Hidden candidates whose catalogue mentions an open requirement (expand reveals them)."""
        out = []
        for number in range(self.visible + 1, len(self.candidates) + 1):
            rids = [r.id for r in self.open_requirements() if self.matches(number, r)]
            if rids:
                out.append((number, rids))
        return out

    # ---------------------------------------------------------------- history

    def push(self, exchange: Exchange) -> None:
        """Add an exchange to the FIFO; the ones it evicts become summary lines."""
        self.recent.append(exchange)
        while len(self.recent) > KEEP_RECENT:
            old = self.recent.pop(0)
            self.summary.append(f"turn {old.turn}: {old.summary}")

    # ---------------------------------------------------------------- rendering

    def render_requirements(self) -> str:
        """The requirements block of the read-only part, once defined."""
        if not self.requirements:
            return "REQUIREMENTS: not defined yet -- your first call is define_requirements."
        return "REQUIREMENTS (read-only, defined at turn 1):\n" + "\n".join(
            f"  {r.label}" for r in self.requirements
        )

    def render_turn(self, status: list[str], note: str = "") -> str:
        """The user message of a turn: the candidates, the history, STATE and the system's note."""
        parts = [
            "CANDIDATES (retrieval order; #N is the number to use):\n" + self.render_candidates(),
            self.render_history(),
            self.render_state(status),
        ]
        if note:
            parts.append(note)
        parts.append("Answer with exactly one tool call.")
        return "\n\n".join(parts)

    def render_candidates(self) -> str:
        """The visible candidates, each with its mark, catalogue lines, description and indexed columns."""
        lines = []
        for number in range(1, self.visible + 1):
            name = self.candidates[number - 1]
            meta = self.metadata.get(name, {})
            mark = (
                "SELECTED"
                if number in self.selection
                else "inspected"
                if number in self.cards
                else "not inspected"
            )
            description = " ".join(str(meta.get("description") or "").split())[:150]
            columns = ", ".join(str(c)[:24] for c in (meta.get("columns.name") or [])[:8])
            if self.slim_inspected and number in self.cards:
                description, columns = "", ""
            lines.append(
                f"#{number} [{mark}] {self.title(number)}"
                + "".join(
                    f"\n    {label}: {value}" for label, value in self.catalogue_lines(number)
                )
                + (f"\n    {description}" if description else "")
                + (f"\n    indexed columns: {columns}" if columns else "")
            )
        hidden = len(self.candidates) - self.visible
        if hidden:
            lines.append(f"({hidden} more candidates are hidden; expand reveals 5 at a time)")
        return "\n".join(lines)

    def render_state(self, status: list[str]) -> str:
        """STATE: requirements, selection, cards, pointers, the agent's ``status`` lines and the last refusal."""
        lines = [
            "[STATE -- kept by the system from your tool calls; it is current, trust it over the history]"
        ]
        lines += self._requirement_lines()
        lines += self._selection_lines()
        lines += self._card_lines()
        lines += self._pointer_lines()
        lines.extend(status)
        if self.last_refusal:
            lines.append("Last refusal: " + self.last_refusal[:400])
        return "\n".join(lines)

    def _requirement_lines(self) -> list[str]:
        if not self.requirements:
            return []
        covered = self.covered_by()
        lines = ["Requirements:"]
        for r in self.requirements:
            if r.waived:
                mark = f"waived ({r.waived[:80]})"
            elif covered.get(r.id):
                mark = "covered by " + ", ".join(f"#{n}" for n in covered[r.id])
            else:
                mark = "OPEN"
            lines.append(f"  {r.label} -- {mark}")
        return lines

    def _selection_lines(self) -> list[str]:
        if self.selection:
            lines = [
                "Selection: "
                + "; ".join(
                    f"#{n} {self.heading(n)} covers {', '.join(rids) or 'nothing yet'}"
                    for n, rids in sorted(self.selection.items())
                )
            ]
        else:
            lines = ["Selection: empty"]
        if self.unselected:
            lines.append(
                "Removed from the selection: "
                + "; ".join(f"#{n} ({why[:60]})" for n, why in self.unselected.items())
            )
        return lines

    def _card_lines(self) -> list[str]:
        if not self.cards:
            return []
        lines = ["Inspected tables (full profiles: recall(<number>)):"]
        for n in sorted(self.cards):
            c = self.cards[n]
            lines.append(f'  #{n} "{c.title}" -- ' + card_line(c, self.card_negatives))
        return lines

    def _pointer_lines(self) -> list[str]:
        """What STATE points the agent to: partitions to add, alternatives, relationships, the queue, hidden matches."""
        lines = []
        partitions = self.partitions_to_add()
        if partitions:
            lines.append(
                "Partitions to add (inspected, not selected, showing an OPEN requirement): "
                + "; ".join(f"#{n} shows {', '.join(rids)}" for n, rids in partitions[:8])
                + " -- select them for those requirements"
            )
        alternatives = self.alternatives()
        if alternatives:
            lines.append(
                "Alternatives to compare (inspected, not selected, showing as many requirements as a selected table): "
                + "; ".join(
                    f"#{n} vs #{s} (both show {', '.join(rids)})" for n, s, rids in alternatives[:4]
                )
                + " -- keep the one whose columns and values match the question; unselect the other with the reason"
            )
        if self.relations:
            lines.append(
                "Relationships between selected tables: "
                + " | ".join(f"#{a}-#{b}: {s}" for (a, b), s in sorted(self.relations.items()))
            )
        queue = self.queue()
        if queue:
            lines.append(
                "Still to inspect (ranked): " + "; ".join(f"#{n} ({why})" for n, why in queue[:8])
            )
        hidden = self.hidden_matches()
        if hidden:
            lines.append(
                "Hidden candidates matching open requirements: "
                + ", ".join(f"#{n} ({'/'.join(rids)})" for n, rids in hidden[:6])
                + " -- expand reveals them"
            )
        return lines

    def render_history(self) -> str:
        """The summary of the evicted turns, then the recent turns verbatim."""
        parts = []
        if self.summary:
            parts.append(
                "SUMMARY OF EARLIER TURNS (written by the system):\n"
                + "\n".join(self.summary[-30:])
            )
        if self.recent:
            parts.append(
                "RECENT TURNS (oldest first):\n"
                + "\n\n".join(
                    f"[turn {e.turn}] you called {e.call}\nresult:\n{e.result}" for e in self.recent
                )
            )
        return "\n\n".join(parts) or "No turns yet."


def card_line(card: Card, negatives: bool = True) -> str:
    """A card in STATE: what the table shows per requirement, its columns, its periods.

    Whole facts within ``CARD_CHARS``, the words a requirement matched said
    once (by ``shows``, not again as word facts).

    Args:
        card: The card.
        negatives: Whether to keep the facts on what the table does not hold.

    Returns:
        The card's line.
    """
    shows = (
        "; ".join(f"{rid} in {where}" for rid, where in card.shows.items())
        or "none of the requirements"
    )
    parts = [f"shows {shows}"]
    for fact in card.facts:
        if fact.startswith(WORD_FACTS):
            continue
        if not negatives and (
            fact.endswith(PERIOD_NEGATIVE) or fact.startswith("words found nowhere")
        ):
            continue
        parts.append(fact)
    kept, used = [], 0
    for part in parts:
        if used + len(part) + 3 > CARD_CHARS and kept:
            kept.append("...")
            break
        kept.append(part)
        used += len(part) + 3
    return (f"{card.rows} rows; " if card.rows else "") + " | ".join(kept)


def _with_search(facts: list[str], found: dict[str, str]) -> list[str]:
    """The card's facts with the words the search found in the values moved out of "words found nowhere"."""
    if not found or not facts:
        return facts
    out = []
    for fact in facts:
        if fact.startswith("words found nowhere: "):
            rest = [w for w in fact.split(": ", 1)[1].split(", ") if w not in found]
            fact = "words found nowhere: " + ", ".join(rest) if rest else ""
        if fact:
            out.append(fact)
    at = next((i + 1 for i, f in enumerate(out) if f.startswith("words in values")), len(out))
    out.insert(
        at,
        "words in values (search): " + ", ".join(f"{w} in {c}" for w, c in list(found.items())[:6]),
    )
    return out


def _shown_by(
    r: Requirement,
    columns: list[str],
    samples: dict[str, str],
    title: str,
    profile: str,
    found: dict[str, str],
) -> list[str]:
    """Where an inspected table shows a requirement that is not an item, naming the column.

    Columns, values, dates, title; ``found`` is the profile's search (word ->
    column) for words the samples do not show.

    Args:
        r: The requirement.
        columns: The table's columns.
        samples: Sampled values by column, lowercased.
        title: The table's titles and catalogue text, lowercased.
        profile: The table's profile.
        found: The profile's search: word -> column.

    Returns:
        Where the table shows the requirement; empty when it does not.
    """
    words = content_words(r.text)
    where = []
    column = next((c for w in words for c in columns if mentions(c.casefold(), w)), None)
    if column is not None:
        where.append(f"columns ({column[:28]})")
    column = next((c for w in words for c, v in samples.items() if mentions(v, w)), None)
    if column is not None:
        where.append(f"values ({column[:28]})")
    for period in ledger.question_periods(r.text) or YEAR.findall(r.text):
        evidence = ledger.inspected_period_evidence(profile, period)
        if evidence:
            where.append(f"dates ({evidence[0]})")
            break
        if period in title:
            where.append("title year")
            break
    if not where and words and any(mentions(title, w) for w in words):
        where.append("title/description")
    searched_in = [found[w] for w in words if w in found]
    if not where and searched_in:
        where.append(f"values ({searched_in[0][:28]})")
    return where
