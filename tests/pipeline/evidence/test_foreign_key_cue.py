"""Foreign keys the column names give away, confirmed on the values at call time."""

import pandas as pd

from lakegen.pipeline.evidence.join_union.join_keys import JOIN
from lakegen.pipeline.evidence.join_union.key_search import find_join_keys, settle_foreign_keys
from lakegen.pipeline.evidence.join_union.matching.name_cue import (
    cue_strength,
    foreign_key_pairs,
    is_generic_key,
    table_stem,
)


def test_table_stem_reads_the_last_word_and_ignores_opaque_names():
    assert table_stem("superhero_alignment.parquet") == "alignment"
    assert table_stem("users.parquet") == "user"
    assert table_stem("4p5c-cbgn.parquet") == ""
    assert table_stem("0b8a6f1e-2c3d-4e5f-9a0b-1c2d3e4f5a6b.parquet") == ""


def test_generic_key_names():
    assert is_generic_key("Id") and is_generic_key("UUID")
    assert not is_generic_key("alignment_id")


def test_a_generic_key_pairs_with_a_column_named_after_its_table():
    pairs = foreign_key_pairs(
        ["name", "alignment_id"],
        ["id", "alignment"],
        "superhero.parquet",
        "superhero_alignment.parquet",
    )
    assert ("alignment_id", "id") in pairs
    assert cue_strength("alignment_id", "id", "superhero", "superhero_alignment") == 2


def test_a_shared_word_pairs_columns_whatever_their_affixes():
    pairs = foreign_key_pairs(["link_to_member"], ["member_id"], "attendance", "member")
    assert pairs == [("link_to_member", "member_id")]


def test_unrelated_ids_are_not_paired():
    assert foreign_key_pairs(["raceId"], ["disp_id"], "results", "disp") == []


def _tables():
    alignment = pd.DataFrame({"id": range(1, 4), "label": ["good", "bad", "neutral"]})
    hero = pd.DataFrame({"name": [f"h{i}" for i in range(60)], "alignment_id": [1, 2, 3] * 20})
    return hero, alignment


def test_cue_pairs_are_measured_and_kept_when_the_values_confirm_them():
    hero, alignment = _tables()
    cue = foreign_key_pairs(
        list(hero.columns), list(alignment.columns), "superhero", "superhero_alignment"
    )
    keys = find_join_keys(hero, alignment, [], cue_pairs=cue)  # Valentine proposed nothing
    kept = settle_foreign_keys(keys)
    found = [ev for ev in kept if ev.left_cols == ("alignment_id",) and ev.right_cols == ("id",)]
    assert found and found[0].by_cue and found[0].verdict == JOIN


def test_cue_pairs_the_values_do_not_confirm_are_dropped():
    hero, alignment = _tables()
    hero["alignment_id"] = range(1000, 1060)  # same name, no value in common
    cue = foreign_key_pairs(
        list(hero.columns), list(alignment.columns), "superhero", "superhero_alignment"
    )
    keys = find_join_keys(hero, alignment, [], cue_pairs=cue)
    assert not [ev for ev in settle_foreign_keys(keys) if ev.by_cue]
