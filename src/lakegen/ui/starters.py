from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import chainlit as cl

from lakegen.config import BASE_DIR


STARTER_LIMIT = 3
LABEL_LIMIT = 72

CORE_QUERY_FILES: dict[str, Path] = {
    "nyc": BASE_DIR / "queries/generated_queries_new_york.json",
    "valencia": BASE_DIR / "queries/generated_queries_valencia.json",
    "bologna": BASE_DIR / "queries/generated_queries_bologna.json",
    "paris": BASE_DIR / "queries/generated_queries_paris.json",
}


DEFAULT_QUESTIONS = {
    "bologna": [
        (
            "Donne votanti alle elezioni del 2022",
            "Quali sono i 3 quartieri con più donne votanti alle elezioni del senato del 2022?"
        ),
        (
            "Confronto del totale dei votanti maschi nel 2022",
            "Chi ha avuto più votanti maschi nelle elezioni del 2022? Il quartiere di porto saragozza o navile?"
        ),
        (
            "Referendum details",
            "What is the total number of 'no' votes for the referendum question 4 in each zone, considering only the referendum of 2022, and how many 'no' votes are there for the referendum question 5 in each zone?"
        )
    ],
    "nyc": [
        (
            "Districts and student removals and suspensions",
            "What are the 3 districts with more total students removals and suspensions in the 2016-2017 academic year?",
        ),
        (
            "Restaurants information about inspections and seating applications",
            "What are the restaurants that have been inspected by the hyghien department and that have applied for outdoor seating in brooklyn?"
        )
    ],
    "valencia": [
        (
            "High PM10 levels dates",
            "Looking at the most recent data, which dates recorded PM10 levels exceeding the critical threshold of 50 µg/m³?",
        ),
        (
            "Sensors loud level",
            "Were there any days where the Cuba 3 sensor was louder than the Cadis 3 sensor during the night?"
        )
    ],
    "paris": [
        (
            "Economic activity codes with a protected green space",
            "Quels sont les codes d'activité économique uniques dans l'économie sociale et solidaire qui ont un espace vert protégé?"
        )
    ]
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
def starter_questions_for_core(core: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    query_file = CORE_QUERY_FILES.get(core)
    if query_file is None or not query_file.exists():
        return ()

    try:
        payload = json.loads(query_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    seen: set[str] = set()

    labels, questions = zip(*DEFAULT_QUESTIONS[core])
    labels = list(labels)
    questions = list(questions)

    for question in _iter_questions(payload):
        if question in seen:
            continue
        seen.add(question)
        labels.append(_starter_label(question))
        questions.append(question)
        if len(questions) >= STARTER_LIMIT:
            break
    return tuple(zip(labels, questions))


def starters_for_core(core: str) -> list[cl.Starter]:
    return [
        cl.Starter(label=label, message=question)
        for label, question in starter_questions_for_core(core)
    ]
