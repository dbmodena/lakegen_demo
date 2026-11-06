"""The starter questions the chat offers come from files the repository holds."""

from __future__ import annotations

import json

from lakegen.config import paths as core_config
from lakegen.config.registry import lakes
from lakegen.entrypoints.webapp import starters


def test_every_lake_names_question_files_that_exist():
    for lake in lakes():
        assert lake.starters_path is not None and lake.starters_path.is_file(), lake.name


def test_a_lake_without_default_starters_takes_them_from_its_benchmark():
    questions = starters.starter_questions_for_core("chicago")
    assert len(questions) == starters.STARTER_LIMIT
    cases = json.loads(
        (core_config.BASE_DIR / "benchmark" / "100q_chicago.json").read_text(encoding="utf-8")
    )["cases"]
    asked = {" ".join(case["question"].split()) for case in cases}
    assert all(question in asked for _label, question in questions)
    assert all(0 < len(label) <= starters.LABEL_LIMIT for label, _question in questions)
