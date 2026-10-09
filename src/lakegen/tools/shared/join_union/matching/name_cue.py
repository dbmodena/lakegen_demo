"""Foreign keys that column names give away, whatever Valentine scores them.

A foreign key is named after what it points to: `alignment_id` in `superhero` points at the `id` of
`alignment`, `UserId` in `postHistory` at the `Id` of `users`, `link_to_member` at `member_id`. COMA scores
`alignment_id` against `id` low (the names share almost nothing) and, keeping only the best partner of
each column, drops the pair; the value pass skips columns that already have a name partner. On the
TARGET BIRD benchmark that left 31% of the join conditions unproposed.

This module proposes such pairs from the names alone, without looking at a value:

  * the two columns share a word (`link_to_member` and `member_id` share `member`), or
  * one column is a generic key (`id`, `key`, `code`) and the other contains the stem of the table the
    generic key belongs to (`alignment_id` and the table `superhero_alignment`).

A proposal is only a lead. `key_search` measures it on the whole columns and keeps it only when the
values confirm it. Generic words (`id`, `key`, `link`, `to`) are ignored, so two unrelated tables whose
keys are both called `...id` are not paired (`raceId` against `disp_id`).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

GENERIC_KEYS = frozenset({"id", "key", "code", "pk", "uuid", "guid"})
# words that carry no meaning in a key's name
GENERIC_WORDS = frozenset({"id", "key", "code", "link", "to", "fk", "pk", "no", "num", "number", "ref", "of", "the", "by", "ids"})
# most pairs one table pair may propose
MAX_CUE_PAIRS = 12


def _squash(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).casefold())


def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def column_words(name: object) -> set[str]:
    """The meaningful lower-case words of a column name: `link_to_member` -> {member}, `UserId` -> {user}."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    words = {_singular(w.casefold()) for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}
    return {w for w in words if w and w not in GENERIC_WORDS}


def is_generic_key(name: object) -> bool:
    return _squash(name) in GENERIC_KEYS


def table_stem(file_name: str) -> str:
    """The word naming a table: `superhero_alignment.parquet` -> `alignment`, `users` -> `user`.

    Empty for names that say nothing: a UUID, a Socrata id (`4p5c-cbgn`) or a word with digits among letters.
    """
    stem = Path(file_name).stem.casefold()
    if re.fullmatch(r"[a-z0-9]{4}-[a-z0-9]{4}", stem):
        return ""
    last = re.split(r"[_\-\s.]+", stem)[-1]
    if len(last) >= 8 and re.search(r"\d", last) or re.fullmatch(r"[0-9a-f-]{12,}", last):
        return ""
    return _singular(last)


def cue_strength(left: str, right: str, left_table: str, right_table: str) -> int:
    """0 when the names give no cue; 2 when one column names the table of the other; 1 when they share a word."""
    left_stem, right_stem = table_stem(left_table), table_stem(right_table)
    if is_generic_key(right) and not is_generic_key(left) and right_stem and right_stem in column_words(left):
        return 2
    if is_generic_key(left) and not is_generic_key(right) and left_stem and left_stem in column_words(right):
        return 2
    if column_words(left) & column_words(right):
        return 1
    return 0


def foreign_key_pairs(
    left_columns: Sequence[str], right_columns: Sequence[str], left_table: str, right_table: str
) -> list[tuple[str, str]]:
    """Column pairs whose names suggest a foreign key between the two tables, strongest cue first."""
    scored = [
        (cue_strength(str(l), str(r), left_table, right_table), str(l), str(r))
        for l in left_columns
        for r in right_columns
    ]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(key=lambda item: -item[0])
    return [(l, r) for _, l, r in scored[:MAX_CUE_PAIRS]]
