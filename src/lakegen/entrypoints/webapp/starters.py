"""The starter questions the app offers for each lake."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

import chainlit as cl

from lakegen.config.registry import lakes

logger = logging.getLogger(__name__)

STARTER_LIMIT = 3
LABEL_LIMIT = 72

CORE_QUERY_FILES: dict[str, Path] = {
    lake.name: lake.starters_path for lake in lakes() if lake.starters_path
}


DEFAULT_QUESTIONS = {
    "bologna": [
        (
            "Donne votanti alle elezioni del 2022",
            "Quali sono i 3 quartieri con più donne votanti alle elezioni del senato del 2022?",
        ),
        (
            "Confronto del totale dei votanti maschi nel 2022",
            "Chi ha avuto più votanti maschi nelle elezioni del 2022? Il quartiere di porto saragozza o navile?",
        ),
        (
            "Referendum details",
            "What is the total number of 'no' votes for the referendum question 4 in each zone, considering only the referendum of 2022, and how many 'no' votes are there for the referendum question 5 in each zone?",
        ),
    ],
    "nyc": [
        (
            "Districts and student removals and suspensions",
            "What are the 3 districts with more total students removals and suspensions in the 2016-2017 academic year?",
        ),
        (
            "Restaurants information about inspections and seating applications",
            "What are the restaurants that have been inspected by the hyghien department and that have applied for outdoor seating in brooklyn?",
        ),
    ],
    "valencia": [
        (
            "High PM10 levels dates",
            "Looking at the most recent data, which dates recorded PM10 levels exceeding the critical threshold of 50 µg/m³?",
        ),
        (
            "Sensors loud level",
            "Were there any days where the Cuba 3 sensor was louder than the Cadis 3 sensor during the night?",
        ),
    ],
    "paris": [
        (
            "Economic activity codes with a protected green space",
            "Quels sont les codes d'activité économique uniques dans l'économie sociale et solidaire qui ont un espace vert protégé?",
        )
    ],
}


def _iter_questions(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        question = value.get("question")
        if isinstance(question, str):
            normalized = " ".join(question.split())
            if normalized:
                yield normalized
        for nested in value.values():
            yield from _iter_questions(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_questions(nested)


def _starter_label(question: str) -> str:
    if len(question) <= LABEL_LIMIT:
        return question
    return f"{question[: LABEL_LIMIT - 3].rstrip()}..."


@lru_cache(maxsize=len(CORE_QUERY_FILES))
def starter_questions_for_core(core: str) -> tuple[tuple[str, str], ...]:
    """The lake's starter questions with their labels.

    The lake's default questions come first, then questions from its
    benchmark file until there are ``STARTER_LIMIT``. A benchmark file that
    cannot be read or parsed is logged and skipped.

    Args:
        core: The lake's name (its Solr core).

    Returns:
        ``(label, question)`` pairs.
    """
    defaults = DEFAULT_QUESTIONS.get(core, [])
    labels = [label for label, _ in defaults]
    questions = [question for _, question in defaults]

    query_file = CORE_QUERY_FILES.get(core)
    if query_file is not None and query_file.exists():
        try:
            payload = json.loads(query_file.read_text(encoding="utf-8"))
            seen: set[str] = set(questions)
            for question in _iter_questions(payload):
                if question in seen:
                    continue
                seen.add(question)
                labels.append(_starter_label(question))
                questions.append(question)
                if len(questions) >= STARTER_LIMIT:
                    break
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read the starter questions in %s", query_file, exc_info=True)
    return tuple(zip(labels, questions, strict=True))


def starters_for_core(core: str) -> list[cl.Starter]:
    """The lake's starter questions as Chainlit starters."""
    return [
        cl.Starter(label=label, message=question)
        for label, question in starter_questions_for_core(core)
    ]
