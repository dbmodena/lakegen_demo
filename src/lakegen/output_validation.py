"""Deterministic quality gates for generated and synthesized answers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class AnswerDisposition(StrEnum):
    VALID = "valid"
    REJECTED = "rejected"
    EMPTY = "empty"


@dataclass(frozen=True)
class AnswerValidation:
    disposition: AnswerDisposition
    reason: str = ""


_REJECTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bREJECT_TABLES\b",
        r"\bno tables? (?:were )?provided\b",
        r"\b(?:data|tables?|datasets?|information) (?:provided )?(?:do(?:es)? not|doesn't|do not) contain\b",
        r"\b(?:required|necessary|needed) (?:data|information|columns?|tables?) (?:is|are) (?:not available|missing|absent)\b",
        r"\b(?:cannot|can't|could not|unable to|not possible to) (?:answer|determine|calculate|resolve)\b",
        r"\binsufficient (?:data|information)\b",
        r"\bnon (?:è|e) possibile (?:rispondere|determinare|calcolare)\b",
        r"\bi dati (?:forniti )?non contengono\b",
        r"\bdati (?:necessari|richiesti) (?:non disponibili|mancanti)\b",
    )
)


def validate_answer(*texts: str | None) -> AnswerValidation:
    """Classify an answer without treating an explicit refusal as success.

    Both execution output and synthesized text should be supplied. A refusal in
    either stage wins over otherwise non-empty prose because synthesis can turn
    a missing-data message into a fluent but still non-answer response.
    """

    non_empty = [str(text).strip() for text in texts if text is not None and str(text).strip()]
    if not non_empty:
        return AnswerValidation(AnswerDisposition.EMPTY, "Answer is empty.")
    for text in non_empty:
        for pattern in _REJECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                excerpt = re.sub(r"\s+", " ", match.group(0)).strip()
                return AnswerValidation(
                    AnswerDisposition.REJECTED,
                    f"Answer declares the question unresolvable: {excerpt}",
                )
    return AnswerValidation(AnswerDisposition.VALID)
