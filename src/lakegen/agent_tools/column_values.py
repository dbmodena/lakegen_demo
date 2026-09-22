"""Value normalisation shared by the join-key and union-mapping evidence.

Open-data tables store the same identifier in different shapes: text with thousands separators
('2,012,938') next to a float (2012938.0), padded or differently-cased text, numbers stored as
text. Comparing raw values would report "no overlap" for tables that do share their keys.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd

_NUMBER_TEXT = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")

# A text column counts as numeric when this share of a bounded sample parses as a number.
NUMERIC_TEXT_SHARE = 0.9
NUMERIC_TEXT_SAMPLE = 2000


def map_uniques(series: pd.Series, fn: Callable[[object], str | None]) -> pd.Series:
    """Apply ``fn`` once per distinct value (fast on big, repetitive columns).

    Missing values, and values ``fn`` maps to None, stay missing.
    """
    codes, uniques = pd.factorize(series)
    mapped = np.empty(len(uniques), dtype=object)
    for index, value in enumerate(uniques):
        mapped[index] = fn(value)
    out = np.full(len(series), None, dtype=object)
    present = codes >= 0
    out[present] = mapped[codes[present]]
    return pd.Series(out, index=series.index, dtype="object")


def as_number(value: object) -> float | None:
    """A number, or a text that is one -- including '2,012,938' (thousands separators)."""
    if isinstance(value, str):
        text = value.strip()
        return float(text.replace(",", "")) if _NUMBER_TEXT.match(text) else None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def format_number(value: object) -> str | None:
    number = as_number(value)
    return None if number is None else repr(number)


def format_text(value: object) -> str | None:
    text = str(value).strip().casefold()
    return text or None


def format_datetime(value: object) -> str | None:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")  # type: ignore[arg-type]
    except Exception:
        return None


def column_kind(series: pd.Series) -> str:
    """'num', 'dt' or 'str' (booleans and everything else are compared as text)."""
    if pd.api.types.is_bool_dtype(series):
        return "str"
    if pd.api.types.is_numeric_dtype(series):
        return "num"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "dt"
    return "str"


def mostly_numeric(series: pd.Series) -> bool:
    """True when a text column holds numbers ('2,012,938', '007') in a bounded sample."""
    sample = series.head(NUMERIC_TEXT_SAMPLE * 10).dropna().head(NUMERIC_TEXT_SAMPLE)
    if sample.empty:
        return False
    parsed = sum(as_number(value) is not None for value in sample)
    return parsed / len(sample) >= NUMERIC_TEXT_SHARE
