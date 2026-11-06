"""Value normalisation shared by the join-key and union-mapping evidence.

Open-data tables store the same identifier in different shapes: text with thousands separators
('2,012,938') next to a float (2012938.0), padded or differently-cased text, numbers stored as
text. Comparing raw values would report "no overlap" for tables that do share their keys.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from lakegen.utils.table_io import TableProfile

logger = logging.getLogger(__name__)

_NUMBER_TEXT = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")
# a number written in any locale: '1.234,56', "1'234.5", '12 345', '7%'
_LOOSE_NUMBER_TEXT = re.compile(r"^[+-]?\d[\d.,'\s]*%?$")
# an amount: thousands groups of three digits and one or two decimals ('1.234,56'), or plain decimals ('12.5')
_DECIMAL_TEXT = re.compile(r"^[+-]?(?:\d{1,3}(?:[.,'\s]\d{3})*|\d+)[.,]\d{1,2}$")

# A text column counts as numeric when this share of a bounded sample parses as a number.
NUMERIC_TEXT_SHARE = 0.9
NUMERIC_TEXT_SAMPLE = 2000
# a text column is a column of amounts when this share of its values is written with decimals
DECIMAL_TEXT_SHARE = 0.3


def map_uniques(series: pd.Series, fn: Callable[[object], str | None]) -> pd.Series:
    """Apply ``fn`` once per distinct value (fast on big, repetitive columns).

    Missing values, and values ``fn`` maps to None, stay missing.

    Args:
        series: The column.
        fn: The mapping of one value.

    Returns:
        The mapped column, of object dtype, on the same index.
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
    """A number in one text form (``repr`` of the float), or None for a value that is not one."""
    number = as_number(value)
    return None if number is None else repr(number)


def format_text(value: object) -> str | None:
    """A value stripped and casefolded, or None when nothing is left."""
    text = str(value).strip().casefold()
    return text or None


def format_datetime(value: object) -> str | None:
    """A date as YYYY-MM-DD, or None for a value that is not one."""
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")  # type: ignore[arg-type]
    except Exception:
        logger.debug("Not a date, compared as missing: %r", value)
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


def _share_matching(series: pd.Series, pattern: re.Pattern[str]) -> float:
    sample = series.head(NUMERIC_TEXT_SAMPLE * 10).dropna().head(NUMERIC_TEXT_SAMPLE)
    if sample.empty:
        return 0.0
    return float(sample.astype(str).str.strip().str.match(pattern).mean())


def looks_numeric(series: pd.Series) -> bool:
    """A number column, or a text column that writes numbers (in any locale: '1.234,56', '12 345')."""
    if column_kind(series) == "num":
        return True
    return (
        column_kind(series) == "str"
        and _share_matching(series, _LOOSE_NUMBER_TEXT) >= NUMERIC_TEXT_SHARE
    )


def mostly_numeric(series: pd.Series) -> bool:
    """True when a text column holds numbers ('2,012,938', '007') in a bounded sample."""
    sample = series.head(NUMERIC_TEXT_SAMPLE * 10).dropna().head(NUMERIC_TEXT_SAMPLE)
    if sample.empty:
        return False
    parsed = sum(as_number(value) is not None for value in sample)
    return parsed / len(sample) >= NUMERIC_TEXT_SHARE


def looks_like_row_numbers(series: pd.Series, profile: TableProfile | None = None) -> bool:
    """Consecutive integers (1..N / 0..N-1).

    Overlap between two such columns is guaranteed by construction and says nothing about a
    relationship. A random sample of such a column is NOT consecutive (a 2,000-row sample of 1..458,000 has gaps
    everywhere), so on a sampled table the check would miss it and report the row numbers as a
    1:1 join key. When ``profile`` says the frame is a sample of a larger table, the table's own
    footer statistics decide instead: the column starts at 0 or 1 and spans exactly the table's
    non-null row count, and the sample must hold no repeated value.

    Args:
        series: The column, or a sample of it.
        profile: The footer facts of the table, when ``series`` is a sample.

    Returns:
        Whether the column holds row numbers.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna()
    if len(values) < 3 or not (values % 1 == 0).all():
        return False
    distinct = values.nunique()
    if distinct != len(values):
        return False
    whole = (
        profile.ranges.get(str(series.name))
        if profile is not None and profile.rows > len(series)
        else None
    )
    if whole is not None:
        low, high, nulls = whole
        return low in (0, 1) and high % 1 == 0 and (high - low + 1) == profile.rows - nulls
    low, high = values.min(), values.max()
    return low in (0, 1) and (high - low + 1) == distinct


def has_fractions(series: pd.Series) -> bool:
    """A column of fractions is a measurement, not an identifier.

    A float with non-integral values, or text that writes amounts ('1.234,56', '12.5').

    Args:
        series: The column.

    Returns:
        Whether the column holds fractions.
    """
    if pd.api.types.is_float_dtype(series):
        values = series.head(100_000).dropna().head(10_000)
        return bool(len(values)) and bool((values % 1 != 0).any())
    if column_kind(series) == "str":
        return _share_matching(series, _DECIMAL_TEXT) >= DECIMAL_TEXT_SHARE
    return False


# whole numbers are only evidence of a shared domain when the domain is wide: ids, codes, years
# overlap by design, while two small counts overlap by chance
MIN_DISTINCT_TEXT = 3
MIN_DISTINCT_NUMBERS = 20


def can_match_on_values(series: pd.Series) -> bool:
    """Whether the values of this column can say it corresponds to a column of another table.

    Text and dates with some variety can, and so can whole numbers over a wide domain. Amounts and
    coordinates (fractions), row numbers, flags with one or two values, and nested cells cannot.

    Args:
        series: The column.

    Returns:
        Whether its values can be matched.
    """
    try:
        distinct = int(series.nunique(dropna=True))
    except TypeError:  # unhashable cells: lists, dicts
        return False
    if looks_numeric(series):
        if has_fractions(series) or looks_like_row_numbers(series):
            return False
        return distinct >= MIN_DISTINCT_NUMBERS
    return distinct >= MIN_DISTINCT_TEXT
